#!/usr/bin/env python3
# Harness: background execution -- the model thinks while the harness waits.
"""
s13_background_tasks.py - Background Tasks

Run slow commands in background threads. Before each LLM call, the loop
drains a notification queue and hands finished results back to the model.

    Main thread                Background thread
    +-----------------+        +-----------------+
    | agent loop      |        | task executes   |
    | ...             |        | ...             |
    | [LLM call] <---+------- | enqueue(result) |
    |  ^drain queue   |        +-----------------+
    +-----------------+

    Timeline:
    Agent ----[spawn A]----[spawn B]----[other work]----
                 |              |
                 v              v
              [A runs]      [B runs]
                 |              |
                 +-- notification queue --> [results injected]

Background tasks here are runtime execution slots, not the durable task-board
records introduced in s12.

中文注解：

本章讲解后台任务：把耗时命令放到后台线程执行，让主 agent loop
可以继续处理其它工作。每次调用 LLM 之前，主循环都会清空通知队列，
把已经完成的后台任务结果注入回模型上下文。

主线程负责运行 agent loop；后台线程负责执行耗时任务。
后台任务完成后，不会直接打断主循环，而是把结果放入 notification queue。
主循环在下一次调用模型前读取这个队列，再把结果交给模型继续推理。

时间线含义：
Agent 可以先启动后台任务 A，再启动后台任务 B，然后继续做其它工作。
A 和 B 在后台并行执行；它们完成后，结果会通过通知队列统一注入回来。

注意：
这里的 background task 是运行时执行槽位，只表示“某个命令正在后台跑”。
它不是 s12 中的持久化任务记录；s12 的 task-board record 是写在磁盘上的工作项。
"""

import os
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
RUNTIME_DIR = WORKDIR / ".runtime-tasks" # 运行时任务目录，默认是当前工作目录下的 .runtime-tasks 目录。
RUNTIME_DIR.mkdir(exist_ok=True)
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."

STALL_THRESHOLD_S = 45  # seconds before a task is considered stalled # 后台任务运行多久还没结束，就认为它可能停滞


class NotificationQueue:
    """
    Priority-based notification queue with same-key folding.
    Folding means a newer message can replace an older message with the
    same key, so the context is not flooded with stale updates.

    基于优先级的通知队列，支持相同 key 的折叠。
    折叠的意思是：如果新消息和旧消息拥有相同的 key，
    新消息可以替换旧消息，这样上下文就不会被过期更新刷屏。
    """

    PRIORITIES = {"immediate": 0, "high": 1, "medium": 2, "low": 3}

    def __init__(self):
        self._queue = []  # list of (priority, key, message)
        self._lock = threading.Lock()

    def push(self, message: str, priority: str = "medium", key: str = None):
        """Add a message to the queue, folding if key matches an existing entry."""
        with self._lock:
            if key:
                # Fold: replace existing message with same key
                self._queue = [(p, k, m) for p, k, m in self._queue if k != key]
            self._queue.append((self.PRIORITIES.get(priority, 2), key, message))
            self._queue.sort(key=lambda x: x[0])

    def drain(self) -> list[str]:
        """Return all pending messages in priority order and clear the queue."""
        with self._lock:
            messages = [m for _, _, m in self._queue]
            self._queue.clear()
            return messages


# -- BackgroundManager: threaded execution + notification queue --
class BackgroundManager:
    def __init__(self):
        self.dir = RUNTIME_DIR # 后台任务的工作目录，默认是当前工作目录下的 .runtime-tasks 目录。
        self.tasks = {}  # task_id -> {status, result, command, started_at} # 后台任务列表，task_id 到任务信息的映射。
        self._notification_queue = []  # completed task results # 通知队列，completed task results 表示已经完成的后台任务结果。
        self._lock = threading.Lock() # 锁，用于保护任务列表和通知队列的并发访问。

    def _record_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.json" # 任务记录文件路径，默认是当前工作目录下的 .runtime-tasks 目录下的 task_id.json 文件。

    def _output_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.log" # 任务输出文件路径，默认是当前工作目录下的 .runtime-tasks 目录下的 task_id.log 文件。

    def _persist_task(self, task_id: str):
        record = dict(self.tasks[task_id])
        # 覆盖之前的任务状态信息
        self._record_path(task_id).write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        ) # 持久化任务，将任务信息写入任务记录文件。

    def _preview(self, output: str, limit: int = 500) -> str:
        compact = " ".join((output or "(no output)").split())
        return compact[:limit] # 生成任务输出的简短摘要。

    def run(self, command: str) -> str:
        """Start a background thread, return task_id immediately.""" # 启动一个后台线程，立即返回任务 ID。
        task_id = str(uuid.uuid4())[:8]
        output_file = self._output_path(task_id)
        self.tasks[task_id] = { # 添加任务到任务列表。
            "id": task_id,
            "status": "running",
            "result": None,
            "command": command,
            "started_at": time.time(),
            "finished_at": None,
            "result_preview": "", # 任务输出的简短摘要。
            "output_file": str(output_file.relative_to(WORKDIR)), # 任务输出文件路径。
        }
        self._persist_task(task_id) # 持久化任务状态
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True # 启动一个后台线程，执行任务。daemon=True 表示这是一个守护线程，当主线程退出时，这个线程也会退出。
        )
        thread.start() # 启动后台线程。
        return (
            f"Background task {task_id} started: {command[:80]} "
            f"(output_file={output_file.relative_to(WORKDIR)})"
        )

    def _execute(self, task_id: str, command: str):
        """Thread target: run subprocess, capture output, push to queue."""
        try:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"
        final_output = output or "(no output)"
        preview = self._preview(final_output) # 生成任务输出的简短摘要。
        output_path = self._output_path(task_id)
        output_path.write_text(final_output) # 将任务输出写入任务输出文件。
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = final_output
        self.tasks[task_id]["finished_at"] = time.time()
        self.tasks[task_id]["result_preview"] = preview
        self._persist_task(task_id) # 持久化任务状态
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "preview": preview,
                "output_file": str(output_path.relative_to(WORKDIR)),
            }) # 将任务结果添加到通知队列。

    def check(self, task_id: str = None) -> str:
        """Check status of one task or list all."""
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            visible = {
                "id": t["id"],
                "status": t["status"],
                "command": t["command"],
                "result_preview": t.get("result_preview", ""),
                "output_file": t.get("output_file", ""),
            } # 返回任务信息。
            return json.dumps(visible, indent=2, ensure_ascii=False)
        lines = []
        for tid, t in self.tasks.items():
            lines.append(
                f"{tid}: [{t['status']}] {t['command'][:60]} "
                f"-> {t.get('result_preview') or '(running)'}" # 返回任务信息，只有执行完了，才会有 result_preview。
            )
        return "\n".join(lines) if lines else "No background tasks." # 返回所有任务信息。

    def drain_notifications(self) -> list:
        """Return and clear all pending completion notifications."""
        with self._lock: # 获取锁，防止并发访问通知队列。
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs # 返回通知队列。

    def detect_stalled(self) -> list[str]:
        """
        Return task IDs that have been running longer than STALL_THRESHOLD_S.
        检测停滞任务，返回运行时间超过 STALL_THRESHOLD_S 的任务 ID 列表。
        """
        now = time.time() # 当前时间。
        stalled = [] # 停滞任务 ID 列表。
        for task_id, info in self.tasks.items(): # 遍历所有任务。
            if info["status"] != "running": # 如果任务不是运行状态，则跳过。
                continue
            elapsed = now - info.get("started_at", now) # 计算任务运行时间。
            if elapsed > STALL_THRESHOLD_S: # 如果任务运行时间超过 STALL_THRESHOLD_S，则认为它可能停滞。
                stalled.append(task_id) # 将任务 ID 添加到停滞任务 ID 列表。
        return stalled # 返回停滞任务 ID 列表。


BG = BackgroundManager()


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
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "background_run":   lambda **kw: BG.run(kw["command"]), # 启动一个后台线程，执行任务。
    "check_background": lambda **kw: BG.check(kw.get("task_id")), # 检查后台任务状态。
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command (blocking).",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    # 新增后台任务相关的工具
    {"name": "background_run", "description": "Run command in background thread. Returns task_id immediately.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status. Omit task_id to list all.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
]


def agent_loop(messages: list):
    while True:
        # Drain background notifications and inject as a synthetic user/assistant
        # transcript pair before the next model call (teaching demo behavior).
        notifs = BG.drain_notifications() # 清空通知队列，并返回已经完成的后台任务结果。
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['preview']} "
                f"(output_file={n['output_file']})"
                for n in notifs
            )
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms13 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
