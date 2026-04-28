#!/usr/bin/env python3
# Harness: autonomy -- models that find work without being told.
"""
s17_autonomous_agents.py - Autonomous Agents

Idle cycle with task board polling, auto-claiming unclaimed tasks, and
identity re-injection after context compression. Builds on task boards,
team mailboxes, and protocol support from earlier chapters.

    Teammate lifecycle:
    +-------+
    | spawn |
    +---+---+
        |
        v
    +-------+  tool_use    +-------+
    | WORK  | <----------- |  LLM  |
    +---+---+              +-------+
        |
        | stop_reason != tool_use
        v
    +--------+
    | IDLE   | poll every 5s for up to 60s
    +---+----+
        |
        +---> check inbox -> message? -> resume WORK
        |
        +---> scan .tasks/ -> unclaimed? -> claim -> resume WORK
        |
        +---> timeout (60s) -> shutdown

    Identity re-injection after compression:
    messages = [identity_block, ...remaining...]
    "You are 'coder', role: backend, team: my-team"

Key idea: an idle teammate can safely claim ready work instead of waiting
for every assignment from the lead.
A teammate here is a long-lived worker, not a one-shot subagent that only
returns a single summary.

---

s17_autonomous_agents.py - 自主智能体

实现空闲轮询任务板、自动认领未分配任务，以及上下文压缩后的身份重注入。
基于前几章的任务板、团队邮箱和协议支持构建。

    队员生命周期：
    +-------+
    | 启动   |
    +---+---+
        |
        v
    +-------+  工具调用      +-------+
    | 工作   | <----------- |  LLM  |
    +---+---+              +-------+
        |
        | 停止原因 != tool_use（非工具调用）
        v
    +--------+
    | 空闲   | 每 5 秒轮询，最长持续 60 秒
    +---+----+
        |
        +---> 检查收件箱 -> 有消息？-> 恢复工作
        |
        +---> 扫描 .tasks/ -> 有未认领任务？-> 认领 -> 恢复工作
        |
        +---> 超时（60 秒）-> 关闭

    上下文压缩后的身份重注入：
    messages = [identity_block, ...remaining...]
    "You are 'coder', role: backend, team: my-team"
    （"你是 'coder'，角色：后端，团队：my-team"）

核心思想：处于空闲状态的队员可以主动认领已就绪的任务，而无需等待
负责人逐一分配。
此处的队员是一个长期存活的工作者，而非仅返回单次摘要的一次性子智能体。
"""

import json
import os
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
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
TEAM_DIR = WORKDIR / ".team"  # 团队成员目录
INBOX_DIR = TEAM_DIR / "inbox" # 成员各自的收件箱目录
TASKS_DIR = WORKDIR / ".tasks" # 任务目录
REQUESTS_DIR = TEAM_DIR / "requests" # 成员之间通信的请求记录目录
CLAIM_EVENTS_PATH = TASKS_DIR / "claim_events.jsonl" # 任务认领事件记录文件

POLL_INTERVAL = 5 # 空闲队员每隔 5 秒轮询一次，检查是否有新消息或未认领的任务。
# 这里的关闭退出指的是关闭当前队员的程序进程，而不是退出整个团队协作环境。
IDLE_TIMEOUT = 60 # 如果空闲状态持续超过 60 秒仍无任何工作可做，队员就自动关闭退出。

SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."
# 这里的系统提示语是给团队负责人（你）的，告诉他们团队中的队员都是自主的，他们自己寻找工作。

VALID_MSG_TYPES = {
    "message",  # 普通点对点消息
    "broadcast", # 广播（发给所有队友）
    "shutdown_request", # 请求某个队友关闭  
    "shutdown_response", # 队友回复"我已关闭"
    "plan_approval", # 提交一个计划，请求审批
    "plan_approval_response", # 回复审批结果（同意/拒绝）
}

_claim_lock = threading.Lock() # 任务认领锁，确保同一时间只有一个队员能认领任务。


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir # INBOX_DIR
        self.dir.mkdir(parents=True, exist_ok=True) # 创建收件箱目录

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES: # 检查消息类型是否有效
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl" # 收件箱文件路径
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n") # 将消息写入收件箱文件
        return f"Sent {msg_type} to {to}" # 返回发送成功消息

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl" # 收件箱文件路径
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("") # 清空收件箱文件
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender: # 不发送给自己
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates" # 返回广播成功消息


BUS = MessageBus(INBOX_DIR)


class RequestStore:
    """
    Durable protocol request records.

    s17 should not regress from s16 back to in-memory trackers. These request
    files let autonomous teammates inspect or resume protocol state later.
    """

    def __init__(self, base_dir: Path):
        self.dir = base_dir # REQUESTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock() # 请求记录锁，确保同一时间只有一个队员能创建或更新请求记录。

    def _path(self, request_id: str) -> Path:
        return self.dir / f"{request_id}.json" # 请求记录文件路径

    def create(self, record: dict) -> dict:
        request_id = record["request_id"] # 请求ID
        with self._lock:
            self._path(request_id).write_text(json.dumps(record, indent=2)) # 将请求记录写入文件
        return record # 返回请求记录

    def get(self, request_id: str) -> dict | None:
        path = self._path(request_id) # 请求记录文件路径
        if not path.exists():
            return None
        return json.loads(path.read_text()) # 返回请求记录

    def update(self, request_id: str, **changes) -> dict | None:
        with self._lock:
            record = self.get(request_id) # 获取请求记录
            if not record:
                return None
            record.update(changes) # 更新请求记录
            record["updated_at"] = time.time() # 更新更新时间
            self._path(request_id).write_text(json.dumps(record, indent=2)) # 将请求记录写入文件
        return record # 返回请求记录


REQUEST_STORE = RequestStore(REQUESTS_DIR)


# -- Task board scanning --
def _append_claim_event(payload: dict):
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    with CLAIM_EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _task_allows_role(task: dict, role: str | None) -> bool:
    # 判断任务角色有没有特定需求
    required_role = task.get("claim_role") or task.get("required_role") or "" 
    if not required_role: # 任务对角色没有要求
        return True
    return bool(role) and role == required_role # 判断任务需要的角色与当前工作者角色是否相同


def is_claimable_task(task: dict, role: str | None = None) -> bool:
    # 判断一个任务是否可以被认领
    return (
        task.get("status") == "pending" # 任务状态为 pending
        and not task.get("owner") # 任务没有被认领
        and not task.get("blockedBy") # 任务没有被阻塞
        and _task_allows_role(task, role) # 任务允许被当前角色认领
    )


def scan_unclaimed_tasks(role: str | None = None) -> list:
    # 扫描没有被认领的任务
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if is_claimable_task(task, role): # 根据自己的角色来判断哪些任务还没有被认领
            unclaimed.append(task)
    return unclaimed


def claim_task(
    task_id: int,
    owner: str,
    role: str | None = None,
    source: str = "manual",
) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json" # 任务路径
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text()) # 获取任务的具体信息
        if not is_claimable_task(task, role):
            return f"Error: Task {task_id} is not claimable for role={role or '(any)'}"
        # 任务被认领成功
        task["owner"] = owner 
        task["status"] = "in_progress"
        task["claimed_at"] = time.time()
        task["claim_source"] = source # 用来说明这个任务是被自主认领auto还是领导者指定manual
        path.write_text(json.dumps(task, indent=2)) # 重新写回任务

    _append_claim_event({
        "event": "task.claimed",
        "task_id": task_id,
        "owner": owner,
        "role": role,
        "source": source,
        "ts": time.time(),
    }) # 记录任务被认领事件
    return f"Claimed task #{task_id} for {owner} via {source}"


# -- Identity re-injection after compression --
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


def ensure_identity_context(messages: list, name: str, role: str, team_name: str):
    if messages and "<identity>" in str(messages[0].get("content", "")):
        # 如果身份信息存在
        return
    # 不存在就在最开始的两个位置插入身份信息
    messages.insert(0, make_identity_block(name, role, team_name)) 
    messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})


# -- Autonomous TeammateManager --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir # TEAM_DIR
        self.dir.mkdir(exist_ok=True) # 创建团队成员目录
        self.config_path = self.dir / "config.json" # 团队成员配置文件
        self.config = self._load_config() # 加载团队成员配置
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []} # S15

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def _set_status(self, name: str, status: str):
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        # 创建一个新的队员，并开始工作
        member = self._find_member(name) # 查找队员
        if member:
            if member["status"] not in ("idle", "shutdown"): # 如果队员状态不是空闲或关闭，则返回错误
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role # 更新队员角色
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member) # 新建立一个队员，并添加到成员列表中
        self._save_config() # 保存团队成员配置
        thread = threading.Thread(
            target=self._loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _loop(self, name: str, role: str, prompt: str):
        # 队员的工作循环
        team_name = self.config["team_name"] # 团队名称
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use idle tool when you have no more work. You will auto-claim new tasks." # 当没有更多工作时，使用空闲工具。你会自动认领新任务。
        )
        messages = [{"role": "user", "content": prompt}] # 初始消息
        tools = self._teammate_tools() # 获取自己的工具列表

        while True:
            # -- WORK PHASE: standard agent loop --
            for _ in range(50):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown") # 设置队员状态为关闭
                        return # 退出当前线程
                    messages.append({"role": "user", "content": json.dumps(msg)})
                try:
                    response = client.messages.create(
                        model=MODEL,
                        system=sys_prompt,
                        messages=messages,
                        tools=tools,
                        max_tokens=8000,
                    ) # 认领成功以后回到这里继续工作
                except Exception:
                    self._set_status(name, "idle") # 设置队员状态为空闲
                    return # 退出当前线程
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break # 如果LLM没有调用工具，则退出循环
                results = []
                idle_requested = False # 是否请求空闲
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "idle":
                            idle_requested = True # 请求空闲
                            output = "Entering idle phase. Will poll for new tasks." # 空闲消息
                        else:
                            output = self._exec(name, block.name, block.input)
                        print(f"  [{name}] {block.name}: {str(output)[:120]}")
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output),
                        })
                messages.append({"role": "user", "content": results})
                if idle_requested:
                    break # 如果请求空闲，则退出循环

            # -- IDLE PHASE: poll for inbox messages and unclaimed tasks --
            self._set_status(name, "idle")
            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for _ in range(polls): # 就是自己主动认领任务的检查次数
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name) # 先看看有没有分配任务
                if inbox:
                    ensure_identity_context(messages, name, role, team_name) # 确保身份信息存在
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True # 表示"有活可干，不要关闭，要继续工作"
                    break
                unclaimed = scan_unclaimed_tasks(role) # 用自己的身份去扫描是否有任务
                if unclaimed:
                    task = unclaimed[0] # 获取第一个任务
                    claim_result = claim_task(
                        task["id"], name, role=role, source="auto"
                    )
                    if claim_result.startswith("Error:"): # 认领任务失败
                        continue
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )
                    ensure_identity_context(messages, name, role, team_name)
                    messages.append({"role": "user", "content": task_prompt})
                    messages.append({"role": "assistant", "content": f"{claim_result}. Working on it."})
                    resume = True # 继续工作
                    break

            if not resume:
                self._set_status(name, "shutdown") # 关闭，退出线程
                return
            self._set_status(name, "working")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # these base tools are unchanged from s02
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            updated = REQUEST_STORE.update(
                req_id,
                status="approved" if args["approve"] else "rejected",
                resolved_by=sender,
                resolved_at=time.time(),
                response={"approve": args["approve"], "reason": args.get("reason", "")},
            )
            if not updated:
                return f"Error: Unknown shutdown request {req_id}"
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": args["approve"]},
            )
            return f"Shutdown {'approved' if args['approve'] else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            REQUEST_STORE.create({
                "request_id": req_id,
                "kind": "plan_approval",
                "from": sender,
                "to": "lead",
                "status": "pending",
                "plan": plan_text,
                "created_at": time.time(),
                "updated_at": time.time(),
            })
            BUS.send(
                sender, "lead", plan_text, "plan_approval",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for approval."
        # 认领一个任务，通过任务看板发现一个任务
        if tool_name == "claim_task":
            return claim_task(
                args["task_id"],
                sender,
                role=self._find_member(sender).get("role") if self._find_member(sender) else None,
                source="manual",
            )
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        # these base tools are unchanged from s02
        return [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            # 与队友通信的工具
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
            # 关闭请求的工具
            {"name": "shutdown_response", "description": "Respond to a shutdown request.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            # 计划审批的工具
            {"name": "plan_approval", "description": "Submit a plan for lead approval.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
            # 空闲工具
            {"name": "idle", "description": "Signal that you have no more work. Enters idle polling phase.",
             "input_schema": {"type": "object", "properties": {}}},
            # 认领任务的工具
            {"name": "claim_task", "description": "Claim a task from the task board by ID.",
             "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
        ]

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


# -- Base tool implementations (these base tools are unchanged from s02) --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Lead-specific protocol handlers --
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    REQUEST_STORE.create({
        "request_id": req_id,
        "kind": "shutdown",
        "from": "lead",
        "to": teammate,
        "status": "pending",
        "created_at": time.time(),
        "updated_at": time.time(),
    })
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    req = REQUEST_STORE.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    REQUEST_STORE.update(
        request_id,
        status="approved" if approve else "rejected",
        reviewed_by="lead",
        resolved_at=time.time(),
        feedback=feedback,
    )
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {'approved' if approve else 'rejected'} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    return json.dumps(REQUEST_STORE.get(request_id) or {"error": "not found"})


# -- Lead tool dispatch (14 tools) --
TOOL_HANDLERS = {
    "bash":              lambda **kw: _run_bash(kw["command"]),
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":    lambda **kw: TEAM.list_all(),
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":              lambda **kw: "Lead does not idle.",
    "claim_task":        lambda **kw: claim_task(kw["task_id"], "lead"),
}

# these base tools are unchanged from s02
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "spawn_teammate", "description": "Spawn an autonomous teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "shutdown_response", "description": "Check shutdown request status.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state (for lead -- rarely used).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


def agent_loop(messages: list):
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
            messages.append({
                "role": "assistant",
                "content": "Noted inbox messages.",
            })
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
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
                print(f"> {block.name}: {str(output)[:200]}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output),
                })
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms17 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                owner = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
