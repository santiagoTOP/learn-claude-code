#!/usr/bin/env python3
# Harness: protocols -- structured handshakes between models.
"""
s16_team_protocols.py - Team Protocols

Shutdown protocol and plan approval protocol, both using the same
request_id correlation pattern. Builds on s15's mailbox-based team messaging.

    Shutdown FSM: pending -> approved | rejected

    Lead                              Teammate
    +---------------------+          +---------------------+
    | shutdown_request     |          |                     |
    | {                    | -------> | receives request    |
    |   request_id: abc    |          | decides: approve?   |
    | }                    |          |                     |
    +---------------------+          +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | shutdown_response    | <------- | shutdown_response   |
    | {                    |          | {                   |
    |   request_id: abc    |          |   request_id: abc   |
    |   approve: true      |          |   approve: true     |
    | }                    |          | }                   |
    +---------------------+          +---------------------+
            |
            v
    status -> "shutdown", thread stops

    Plan approval FSM: pending -> approved | rejected

    Teammate                          Lead
    +---------------------+          +---------------------+
    | plan_approval        |          |                     |
    | submit: {plan:"..."}| -------> | reviews plan text   |
    +---------------------+          | approve/reject?     |
                                     +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | plan_approval_response| <------ | plan_approval       |
    | {approve: true}      |          | review: {req_id,    |
    +---------------------+          |   approve: true}     |
                                     +---------------------+

    Request store: .team/requests/{request_id}.json

Key idea: one request/response shape can support multiple kinds of team workflow.
Protocol requests are structured workflow objects, not normal free-form chat.

Read this file in this order:
1. MessageBus: how protocol envelopes still travel through the same inbox surface.
2. Request files under .team/requests: how a request keeps durable status after the message is sent.
3. Protocol handlers: how shutdown and plan approval reuse the same correlation pattern.

Most common confusion:
- a protocol request is not a normal teammate chat message
- a request record is not a task record

Teaching boundary:
this file teaches durable handshakes first.
Autonomous claiming, task selection, and worktree assignment stay in later chapters.

---

团队协议

关闭协议与计划审批协议，两者共用同一套 request_id 关联模式。
基于 s15 的邮箱消息机制构建。

    关闭协议有限状态机：待处理 -> 已批准 | 已拒绝

    领队                              队友
    +---------------------+          +---------------------+
    领队发送关闭请求，                        队友收到请求，
    携带唯一 request_id                      决定是否同意关闭
    用于后续匹配响应
    +---------------------+          +---------------------+
    领队收到响应，通过                        队友回复响应，
    request_id 匹配到                        request_id 与请求一致
    原始请求，完成关联                        approve: true 表示同意关闭
            |
            v
    队友状态变为 "shutdown"，线程停止运行

    计划审批有限状态机：待处理 -> 已批准 | 已拒绝

    队友                              领队
    队友提交计划内容，                        领队审阅计划文本，
    请求领队审批                              决定批准或拒绝
    队友收到审批结果，                        领队回复审批意见，
    继续或放弃执行计划                        携带 request_id 完成关联

    请求持久化存储：.team/requests/{request_id}.json
    每条协议请求以 request_id 命名保存为独立 JSON 文件。
    即使收件箱被清空，请求状态仍可查询，不会丢失。

核心思想：同一套请求/响应结构可以支撑多种团队工作流。
协议请求是结构化的工作流对象，不是普通的自由格式聊天消息。

建议阅读顺序：
1. MessageBus：协议信封仍通过同一套收件箱机制传递
2. .team/requests 下的请求文件：消息发出后请求状态如何持久保存
3. 协议处理器：关闭协议与计划审批如何复用同一套 request_id 关联模式

常见误解：
- 协议请求不是普通的队友聊天消息——它是有状态、有生命周期的结构化对象
- 请求记录不是任务记录——前者描述一次握手流程，后者描述一项待完成的工作

教学边界说明：
本文件优先教授"持久握手"机制。
自主认领任务、任务选择、worktree 分配将在后续章节中补充。
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
MODEL = os.environ["MODEL_ID"] # 模型id
TEAM_DIR = WORKDIR / ".team"  # 团队成员目录
INBOX_DIR = TEAM_DIR / "inbox" # 成员各自的收件箱目录，读取以后就没有了
REQUESTS_DIR = TEAM_DIR / "requests" # 成员之间通信的请求记录目录，读取以后还在

SYSTEM = f"You are a team lead at {WORKDIR}. Manage teammates with shutdown and plan approval protocols."

VALID_MSG_TYPES = {
    "message", # 普通点对点消息
    "broadcast", # 广播（发给所有队友）
    "shutdown_request", # 请求某个队友关闭
    "shutdown_response", # 队友回复"我已关闭"
    "plan_approval", # 队友提交计划，请求领队审批
    "plan_approval_response", # 领队回复计划审批结果
}

# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir # 收件箱目录 TEAM_DIR / "inbox"
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("") # 读完以后清空收件箱
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender: # 不发送给自己
                self.send(sender, name, content, "broadcast") # 发送广播消息
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


class RequestStore:
    """
    Durable request records for protocol workflows.

    Protocol state should survive long enough to inspect, resume, or reconcile.
    This store keeps one JSON file per request_id under .team/requests/.

    协议工作流的持久化请求记录。

    协议状态需要足够持久，以便事后查看、恢复或对账。
    每条请求以 request_id 为文件名，独立保存为 .team/requests/ 下的 JSON 文件。
    """

    def __init__(self, base_dir: Path):
        self.dir = base_dir # 请求记录目录 REQUESTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock() # 线程锁

    def _path(self, request_id: str) -> Path:
        return self.dir / f"{request_id}.json" # 请求记录文件路径 REQUESTS_DIR / {request_id}.json

    def create(self, record: dict) -> dict:
        request_id = record["request_id"]
        with self._lock: # 加锁，防止多个线程同时写入
            self._path(request_id).write_text(json.dumps(record, indent=2))
        return record

    def get(self, request_id: str) -> dict | None:
        path = self._path(request_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def update(self, request_id: str, **changes) -> dict | None:
        with self._lock: # 加锁，防止多个线程同时更新
            record = self.get(request_id)
            if not record:
                return None
            record.update(changes)
            record["updated_at"] = time.time()
            self._path(request_id).write_text(json.dumps(record, indent=2)) # 覆盖之前的记录  
        return record


REQUEST_STORE = RequestStore(REQUESTS_DIR)


# -- TeammateManager with shutdown + plan approval --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir # 团队成员目录 TEAM_DIR
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json" # 团队成员配置文件 TEAM_DIR / "config.json"
        self.config = self._load_config() # 加载团队成员配置
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists(): # 如果团队成员配置文件存在，则加载团队成员配置
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []} # 如果团队成员配置文件不存在，则创建一个默认的团队成员配置

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2)) # 保存团队成员配置

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread # 将新的队友添加到线程列表中
        thread.start() # 启动队友的工作循环
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_response."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()
        should_exit = False # 是否需要关闭
        for _ in range(50):
            inbox = BUS.read_inbox(name) # 读取自己的收件箱
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})
            if should_exit:
                break
            try:
                response = client.messages.create(
                    model=MODEL,
                    system=sys_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=8000,
                )
            except Exception:
                break
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = self._exec(name, block.name, block.input)
                    print(f"  [{name}] {block.name}: {str(output)[:120]}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })
                    # 回复关闭请求
                    if block.name == "shutdown_response" and block.input.get("approve"):
                        should_exit = True
            messages.append({"role": "user", "content": results})
        member = self._find_member(name)
        if member:
            # 更新自己的状态
            member["status"] = "shutdown" if should_exit else "idle"
            self._save_config()

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
            # 具体将消息发给谁由自己inbox中谁给自己发的决定
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))  
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        # 处理关闭请求
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"] # 或者决定关闭，或者拒绝关闭
            updated = REQUEST_STORE.update(
                req_id,
                status="approved" if approve else "rejected",
                resolved_by=sender,
                resolved_at=time.time(),
                response={"approve": approve, "reason": args.get("reason", "")},
            ) # 修改这个请求的记录
            if not updated:
                return f"Error: Unknown shutdown request {req_id}"
            # 将消息发送给领队
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": approve},
            )
            return f"Shutdown {'approved' if approve else 'rejected'}"
        
        # 处理计划审批请求
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8] # 创建一个计划审批请求
            # 放到请求的记录中
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
            # 将消息发送给领队
            BUS.send(
                sender, "lead", plan_text, "plan_approval",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval." 
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
            # subagent的工具
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "shutdown_response", "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            {"name": "plan_approval", "description": "Submit a plan for lead approval. Provide plan text.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
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
    # 创建一个关闭请求
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
    # 通知对应的subagent关闭
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)" # 返回关闭请求的id和当前状态


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    # 获取计划审批请求
    req = REQUEST_STORE.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    # 更新计划审批请求的状态
    REQUEST_STORE.update(
        request_id,
        status="approved" if approve else "rejected",
        reviewed_by="lead",
        resolved_at=time.time(),
        feedback=feedback,
    )
    # 将消息发送给对应的subagent
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    # 反馈给主agent
    return f"Plan {'approved' if approve else 'rejected'} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    return json.dumps(REQUEST_STORE.get(request_id) or {"error": "not found"})


# -- Lead tool dispatch (12 tools) --
TOOL_HANDLERS = {
    "bash":              lambda **kw: _run_bash(kw["command"]),
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 与agent成员相关
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":    lambda **kw: TEAM.list_all(),
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    # 与关闭请求相关
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]), # 创建一个关闭请求
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")), # 检查关闭请求的状态
    # 与计划审批相关
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
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
    {"name": "spawn_teammate", "description": "Spawn a persistent teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "shutdown_response", "description": "Check the status of a shutdown request by request_id.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
]


def agent_loop(messages: list):
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
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
                print(f"> {block.name}:")
                print(str(output)[:200])
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
            query = input("\033[36ms16 >> \033[0m")
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
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
