#!/usr/bin/env python3
# Harness: time -- the agent schedules its own future work.
"""
s14_cron_scheduler.py - Cron / Scheduled Tasks

The agent can schedule prompts for future execution using standard cron
expressions. When a schedule matches the current time, it pushes a
notification back into the main conversation loop.

    Cron expression: 5 fields
    +-------+-------+-------+-------+-------+
    | min   | hour  | dom   | month | dow   |
    | 0-59  | 0-23  | 1-31  | 1-12  | 0-6   |
    +-------+-------+-------+-------+-------+
    Examples:
      "*/5 * * * *"   -> every 5 minutes
      "0 9 * * 1"     -> Monday 9:00 AM
      "30 14 * * *"   -> daily 2:30 PM

    Two persistence modes:
    +--------------------+-------------------------------+
    | session-only       | In-memory list, lost on exit  |
    | durable            | .claude/scheduled_tasks.json  |
    +--------------------+-------------------------------+

    Two trigger modes:
    +--------------------+-------------------------------+
    | recurring          | Repeats until deleted or      |
    |                    | 7-day auto-expiry             |
    | one-shot           | Fires once, then auto-deleted |
    +--------------------+-------------------------------+

    Jitter: recurring tasks can avoid exact minute boundaries.

    Architecture:
    +-------------------------------+
    |  Background thread            |
    |  (checks every 1 second)      |
    |                               |
    |  for each task:               |
    |    if cron_matches(now):      |
    |      enqueue notification     |
    +-------------------------------+
              |
              v
    [notification_queue]
              |
         (drained at top of agent_loop)
              |
              v
    [injected as user messages before LLM call]

Key idea: scheduling remembers future work, then hands it back to the
same main loop when the time arrives.

中文注解：

本章讲解 Cron / 定时任务系统：agent 可以使用标准 cron 表达式，
把某个 prompt 安排到未来某个时间执行。当当前时间匹配某个计划任务时，
调度器会把一条通知推回主对话循环。

Cron 表达式包含 5 个字段：

- min：分钟，范围 0-59
- hour：小时，范围 0-23
- dom：day of month，月份中的日期，范围 1-31
- month：月份，范围 1-12
- dow：day of week，星期几，范围 0-6

示例：

- "*/5 * * * *"：每 5 分钟执行一次
- "0 9 * * 1"：每周一上午 9:00 执行
- "30 14 * * *"：每天 14:30 执行

这里有两种持久化模式：

- session-only：只保存在内存里，程序退出后丢失
- durable：写入 .claude/scheduled_tasks.json，重启后仍然存在

这里有两种触发模式：

- recurring：重复触发，直到被删除，或者达到 7 天自动过期
- one-shot：只触发一次，触发后自动删除

Jitter 表示“抖动”：重复任务可以避开精确的分钟边界，
避免很多任务都在同一秒集中触发。

整体架构：

后台线程每 1 秒检查一次所有计划任务。
如果某个任务的 cron 表达式匹配当前时间，就把通知加入 notification_queue。
主 agent_loop 在每次调用 LLM 前清空这个队列，
然后把通知作为 user message 注入到模型上下文里。

核心思想：
调度器负责记住“未来要做的事”；
等时间到了，再把这件事交还给同一个主循环继续处理。
"""

import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 用来保留持久化任务的文件路径
SCHEDULED_TASKS_FILE = WORKDIR / ".claude" / "scheduled_tasks.json" 
# 防止多个会话同时触发同一个 cron 任务，这个锁文件保存的是运行这个 cron 任务的进程 ID。
# 通过检查锁文件是否存在，以及锁文件中的进程 ID 是否存活，来判断是否可以触发 cron 任务。
CRON_LOCK_FILE = WORKDIR / ".claude" / "cron.lock"
# 让重复定时任务最多保留 7 天，超过 7 天后自动过期并被删除。
AUTO_EXPIRY_DAYS = 7
# 避免重复定时任务在同一分钟触发，JITTER_MINUTES 列表中的分钟数不会被触发。
JITTER_MINUTES = [0, 30]  # avoid these exact minutes for recurring tasks
# 抖动范围，抖动范围在 1-4 分钟之间。
JITTER_OFFSET_MAX = 4     # offset range in minutes
# Teaching version: use a simple 1-4 minute offset when needed.


class CronLock:
    """
    PID-file-based lock to prevent multiple sessions from firing the same cron job.
    """

    def __init__(self, lock_path: Path = None):
        self._lock_path = lock_path or CRON_LOCK_FILE # 锁文件路径，默认是当前工作目录下的 .claude 目录下的 cron.lock 文件。

    def acquire(self) -> bool:
        """
        Try to acquire the cron lock. Returns True on success.

        If a lock file exists, check whether the PID inside is still alive.
        If the process is dead the lock is stale and we can take over.
        """
        if self._lock_path.exists():
            try:
                stored_pid = int(self._lock_path.read_text().strip())
                # PID liveness probe: send signal 0 (no-op) to check existence
                os.kill(stored_pid, 0)
                # Process is alive -- lock is held by another session
                return False # 如果锁文件存在，并且进程 ID 仍然存活，则返回 False，表示锁已经被另一个会话持有。
            except (ValueError, ProcessLookupError, PermissionError, OSError):
                # Stale lock (process dead or PID unparseable) -- remove it
                pass # 如果锁文件存在，并且进程 ID 已经不存在，则删除锁文件。
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.write_text(str(os.getpid())) # 写入当前进程 ID 到锁文件。
        return True # 返回 True，表示锁已经被当前会话持有。

    def release(self):
        """Remove the lock file if it belongs to this process."""
        try:
            if self._lock_path.exists():
                stored_pid = int(self._lock_path.read_text().strip()) # 读取锁文件中的进程 ID。
                if stored_pid == os.getpid():
                    self._lock_path.unlink() # 删除锁文件。
        except (ValueError, OSError):
            pass


def cron_matches(expr: str, dt: datetime) -> bool:
    """
    Check if a 5-field cron expression matches a given datetime.

    Fields: minute hour day-of-month month day-of-week
    Supports: * (any), */N (every N), N (exact), N-M (range), N,M (list)

    No external dependencies -- simple manual matching.
    """
    fields = expr.strip().split() # 分割 cron 表达式，得到 5 个字段。
    if len(fields) != 5:
        return False # 如果 cron 表达式不是 5 个字段，则返回 False。

    values = [dt.minute, dt.hour, dt.day, dt.month, dt.weekday()] # 得到当前时间的 5 个字段。
    # Python weekday: 0=Monday; cron: 0=Sunday. Convert.
    cron_dow = (dt.weekday() + 1) % 7 # 将当前时间的星期几转换为 cron 表达式中的星期几。
    values[4] = cron_dow # 将当前时间的星期几赋值给 values 列表的第 5 个元素。
    ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)] # 定义 5 个字段的范围。

    for field, value, (lo, hi) in zip(fields, values, ranges): # 遍历 5 个字段，检查当前字段是否匹配。
        # 30 14 * * *
        if not _field_matches(field, value, lo, hi): # 检查当前字段是否匹配。
            return False
    return True


def _field_matches(field: str, value: int, lo: int, hi: int) -> bool:
    """Match a single cron field against a value."""
    # field 是 cron 表达式中的单个字段，例如分钟字段 "*"、"*/5"、"10-20"、"1,15,30"。
    # value 是当前时间对应字段的实际值，例如当前分钟是 30，则 value=30。
    # lo/hi 是该字段允许的最小值和最大值，例如分钟字段是 0-59。
    if field == "*":  # "*" 表示任意值都匹配。
        return True

    for part in field.split(","):  # 支持逗号列表，例如 "1,15,30" 会拆成三个候选片段。
        # Handle step: */N or N-M/S
        step = 1  # 默认步长是 1，表示每个值都算匹配。
        if "/" in part:  # 支持步长语法，例如 "*/5" 或 "10-30/5"。
            part, step_str = part.split("/", 1)  # 拆成范围部分和步长部分。
            step = int(step_str)  # 把步长字符串转成整数，例如 "5" -> 5。

        if part == "*":  # 处理 "*/N"，例如 "*/5" 表示每隔 5 个单位匹配一次。
            # */N -- check if value is on the step grid
            if (value - lo) % step == 0:  # 从字段最小值 lo 开始，检查当前值是否落在步长网格上。
                return True
        elif "-" in part:  # 处理范围语法，例如 "10-20" 或 "10-20/2"。
            # Range: N-M
            start, end = part.split("-", 1)  # 拆出范围起点和终点。
            start, end = int(start), int(end)  # 转成整数后才能做大小比较。
            if start <= value <= end and (value - start) % step == 0:  # 同时满足范围和步长才匹配。
                return True
        else:
            # Exact value
            if int(part) == value:  # 处理精确值，例如 "30" 只匹配 value=30。
                return True

    return False  # 所有候选片段都不匹配时，当前 cron 字段不匹配。


class CronScheduler:
    """
    Manage scheduled tasks with background checking.

    Teaching version keeps only the core pieces: schedule records, a
    minute checker, optional persistence, and a notification queue.
    """

    def __init__(self):
        self.tasks = []        # list of task dicts
        self.queue = Queue()   # notification queue
        self._stop_event = threading.Event() # 停止事件，用于停止后台线程。
        self._thread = None
        self._last_check_minute = -1  # avoid double-firing within same minute

    def start(self):
        """Load durable tasks and start the background check thread."""
        self._load_durable() # 加载定时任务
        self._thread = threading.Thread(target=self._check_loop, daemon=True) # 启动一个后台线程，每秒检查一次定时任务。
        self._thread.start() # 启动后台线程。
        count = len(self.tasks) # 获取定时任务数量
        if count:
            print(f"[Cron] Loaded {count} scheduled tasks")

    def stop(self):
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def create(self, cron_expr: str, prompt: str,
               recurring: bool = True, durable: bool = False) -> str:
        """Create a new scheduled task. Returns the task ID."""
        task_id = str(uuid.uuid4())[:8]
        now = time.time()

        task = {
            "id": task_id,
            "cron": cron_expr,
            "prompt": prompt,
            "recurring": recurring,
            "durable": durable,
            "createdAt": now,
        }

        # Jitter for recurring tasks: if the cron fires on :00 or :30,
        # note it so we can offset the check slightly
        if recurring:
            task["jitter_offset"] = self._compute_jitter(cron_expr) # 计算抖动偏移量。

        self.tasks.append(task)
        if durable:
            self._save_durable()

        mode = "recurring" if recurring else "one-shot"
        store = "durable" if durable else "session-only" # 存储模式，持久化或直接存储到内存中的 tasks 列表中。
        return f"Created task {task_id} ({mode}, {store}): cron={cron_expr}"

    def delete(self, task_id: str) -> str:
        """Delete a scheduled task by ID.
        这里的任务删除采用的是原地删除，直接覆盖原来的任务列表，而不是删除任务。
        """
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        if len(self.tasks) < before:
            self._save_durable() 
            return f"Deleted task {task_id}"
        return f"Task {task_id} not found"

    def list_tasks(self) -> str:
        """List all scheduled tasks."""
        if not self.tasks:
            return "No scheduled tasks."
        lines = []
        for t in self.tasks:
            mode = "recurring" if t["recurring"] else "one-shot"
            store = "durable" if t["durable"] else "session"
            age_hours = (time.time() - t["createdAt"]) / 3600
            lines.append(
                f"  {t['id']}  {t['cron']}  [{mode}/{store}] "
                f"({age_hours:.1f}h old): {t['prompt'][:60]}"
            )
        return "\n".join(lines)

    def drain_notifications(self) -> list[str]:
        """Drain all pending notifications from the queue."""
        notifications = []
        while True:
            try:
                # get_nowait 是 Queue 类的一个方法，用于从队列中获取一个元素，如果队列为空，则抛出 Empty 异常。
                notifications.append(self.queue.get_nowait()) 
            except Empty:
                break
        return notifications # 返回通知列表。

    def _compute_jitter(self, cron_expr: str) -> int:
        """If cron targets :00 or :30, return a small offset (1-4 minutes)."""
        fields = cron_expr.strip().split()
        if len(fields) < 1:
            return 0
        minute_field = fields[0] # 获取分钟字段。
        try:
            minute_val = int(minute_field)
            if minute_val in JITTER_MINUTES:
                # Deterministic jitter based on the expression hash
                return (hash(cron_expr) % JITTER_OFFSET_MAX) + 1 # 计算抖动偏移量。
        except ValueError:
            pass
        return 0 # 返回 0，表示没有抖动偏移量。

    def _check_loop(self):
        """Background thread: check every second if any task is due."""
        while not self._stop_event.is_set(): # 检查当前定时任务时间是否被设置为停止。
            now = datetime.now()
            current_minute = now.hour * 60 + now.minute

            # Only check once per minute to avoid double-firing
            if current_minute != self._last_check_minute: # 如果当前分钟与上次检查分钟不同，则检查定时任务。
                self._last_check_minute = current_minute # 更新上次检查分钟。
                self._check_tasks(now)

            self._stop_event.wait(timeout=1) # 等待 1 秒，然后继续检查定时任务。

    def _check_tasks(self, now: datetime):
        """Check all tasks against current time, fire matches."""
        expired = [] # 过期任务 ID 列表。
        fired_oneshots = [] # 已触发一次任务 ID 列表。

        for task in self.tasks: # 遍历所有定时任务。
            # Auto-expiry: recurring tasks older than 7 days
            age_days = (time.time() - task["createdAt"]) / 86400 # 计算任务创建时间与当前时间的差值，单位为天。
            if task["recurring"] and age_days > AUTO_EXPIRY_DAYS: # 如果任务是重复任务，并且创建时间超过 7 天，则将任务 ID 添加到过期任务 ID 列表。
                expired.append(task["id"]) # 将任务 ID 添加到过期任务 ID 列表。
                continue

            # Apply jitter offset for the match check
            check_time = now # 检查时间等于当前时间。
            jitter = task.get("jitter_offset", 0) # 获取任务的抖动偏移量。
            if jitter: # 如果抖动偏移量不为 0，则将检查时间减去抖动偏移量。
                check_time = now - timedelta(minutes=jitter) # 将检查时间减去抖动偏移量。

            if cron_matches(task["cron"], check_time): # 如果当前时间与任务的 cron 表达式匹配，则发送通知。
                notification = (
                    f"[Scheduled task {task['id']}]: {task['prompt']}"
                ) # 生成通知消息，准备发送给主循环。
                self.queue.put(notification) # 将通知消息添加到通知队列。
                task["last_fired"] = time.time() # 更新任务的最后触发时间。
                print(f"[Cron] Fired: {task['id']}")

                if not task["recurring"]:
                    fired_oneshots.append(task["id"]) # 如果任务是一次性任务，则将任务 ID 添加到已触发一次任务 ID 列表。

        # Clean up expired and one-shot tasks
        if expired or fired_oneshots:
            remove_ids = set(expired) | set(fired_oneshots) # 计算需要删除的任务 ID 集合。
            self.tasks = [t for t in self.tasks if t["id"] not in remove_ids]
            for tid in expired:
                print(f"[Cron] Auto-expired: {tid} (older than {AUTO_EXPIRY_DAYS} days)") # 打印过期任务信息。
            for tid in fired_oneshots:
                print(f"[Cron] One-shot completed and removed: {tid}") # 打印一次性任务完成并删除信息。
            self._save_durable() # 保存定时任务。

    def _load_durable(self):
        """Load durable tasks from .claude/scheduled_tasks.json."""
        if not SCHEDULED_TASKS_FILE.exists():
            return
        try:
            data = json.loads(SCHEDULED_TASKS_FILE.read_text())
            # Only load durable tasks
            self.tasks = [t for t in data if t.get("durable")] # 加载定时任务
        except Exception as e:
            print(f"[Cron] Error loading tasks: {e}")

    def detect_missed_tasks(self) -> list[dict]:
        """
        On startup, check each durable task's last_fired time.

        If a task should have fired while the session was closed (i.e.
        the gap between last_fired and now contains at least one cron match),
        flag it as missed. The caller can then let the user decide whether
        to run or discard each missed task.

        中文注解：
        启动时检查每个持久化任务的 last_fired 时间。
        如果 agent 关闭期间本该触发某个任务，也就是 last_fired 到 now
        之间至少有一次匹配 cron 表达式的时间点，就把它标记为 missed。
        调用方之后可以让用户决定：现在补跑这个任务，还是直接丢弃它。

        """
        now = datetime.now()
        missed = []
        for task in self.tasks:
            last_fired = task.get("last_fired")
            if last_fired is None:
                continue
            last_dt = datetime.fromtimestamp(last_fired) # 将最后触发时间转换为 datetime 对象。
            # Walk forward minute-by-minute from last_fired to now (cap at 24h)
            check = last_dt + timedelta(minutes=1) # 从最后触发时间开始，每分钟检查一次。
            cap = min(now, last_dt + timedelta(hours=24)) # 检查时间不能超过 24 小时。
            while check <= cap:
                if cron_matches(task["cron"], check):
                    missed.append({
                        "id": task["id"],
                        "cron": task["cron"],
                        "prompt": task["prompt"],
                        "missed_at": check.isoformat(), # 将检查时间转换为 ISO 格式。
                    })
                    break  # one miss is enough to flag it
                check += timedelta(minutes=1)
        return missed

    def _save_durable(self):
        """Save durable tasks to disk."""
        durable = [t for t in self.tasks if t.get("durable")]
        SCHEDULED_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCHEDULED_TASKS_FILE.write_text(
            json.dumps(durable, indent=2) + "\n"
        ) # 将定时任务保存到磁盘。


# Global scheduler
scheduler = CronScheduler()


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 新增定时任务相关的工具
    "cron_create": lambda **kw: scheduler.create(
        kw["cron"], kw["prompt"], kw.get("recurring", True), kw.get("durable", False)),
    "cron_delete": lambda **kw: scheduler.delete(kw["id"]),
    "cron_list":   lambda **kw: scheduler.list_tasks(),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    # 新增定时任务相关的工具
    {"name": "cron_create", "description": "Schedule a recurring or one-shot task with a cron expression.",
     "input_schema": {"type": "object", "properties": {
         "cron": {"type": "string", "description": "5-field cron expression: 'min hour dom month dow'"},
         "prompt": {"type": "string", "description": "The prompt to inject when the task fires"},
         "recurring": {"type": "boolean", "description": "true=repeat, false=fire once then delete. Default true."},
         "durable": {"type": "boolean", "description": "true=persist to disk, false=session-only. Default false."},
     }, "required": ["cron", "prompt"]}},
    {"name": "cron_delete", "description": "Delete a scheduled task by ID.",
     "input_schema": {"type": "object", "properties": {
         "id": {"type": "string", "description": "Task ID to delete"},
     }, "required": ["id"]}},
    {"name": "cron_list", "description": "List all scheduled tasks.",
     "input_schema": {"type": "object", "properties": {}}},
]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks.\n\nYou can schedule future work with cron_create. Tasks fire automatically and their prompts are injected into the conversation."


def agent_loop(messages: list):
    """
    Cron-aware agent loop.

    Before each LLM call, drain the notification queue and inject any
    fired task prompts as user messages. This is how the agent "wakes up"
    to handle scheduled work.
    """
    while True:
        # Drain scheduled task notifications
        notifications = scheduler.drain_notifications()
        for note in notifications:
            print(f"[Cron notification] {note[:100]}")
            messages.append({"role": "user", "content": note})

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            handler = TOOL_HANDLERS.get(block.name)
            try:
                output = handler(**(block.input or {})) if handler else f"Unknown: {block.name}"
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {block.name}: {str(output)[:200]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(output),
            })

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    scheduler.start()
    print("[Cron scheduler running. Background checks every second.]")
    print("[Commands: /cron to list tasks, /test to fire a test notification]")

    history = []
    while True:
        try:
            query = input("\033[36ms14 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            scheduler.stop()
            break
        if query.strip().lower() in ("q", "exit", ""):
            scheduler.stop()
            break

        if query.strip() == "/cron":
            print(scheduler.list_tasks())
            continue

        if query.strip() == "/test":
            # Manually enqueue a test notification for demonstration
            scheduler.queue.put("[Scheduled task test-0000]: This is a test notification.")
            print("[Test notification enqueued. It will be injected on your next message.]")
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
