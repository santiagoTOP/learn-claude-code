#!/usr/bin/env python3
# Harness: team mailboxes -- multiple models, coordinated through files.
"""
s15_agent_teams.py - Agent Teams

Persistent named agents with file-based JSONL inboxes. Each teammate runs
its own agent loop in a separate thread. Communication happens through
append-only inbox files.

    Subagent (s04):  spawn -> execute -> return summary -> destroyed
    Teammate (s15):  spawn -> work -> idle -> work -> ... -> shutdown

    .team/config.json                   .team/inbox/
    +----------------------------+      +------------------+
    | {"team_name": "default",   |      | alice.jsonl      |
    |  "members": [              |      | bob.jsonl        |
    |    {"name":"alice",        |      | lead.jsonl       |
    |     "role":"coder",        |      +------------------+
    |     "status":"idle"}       |
    |  ]}                        |      send_message("alice", "fix bug"):
    +----------------------------+        open("alice.jsonl", "a").write(msg)

                                        read_inbox("alice"):
    spawn_teammate("alice","coder",...)   msgs = [json.loads(l) for l in ...]
         |                                open("alice.jsonl", "w").close()
         v                                return msgs  # drain
    Thread: alice             Thread: bob
    +------------------+      +------------------+
    | agent_loop       |      | agent_loop       |
    | status: working  |      | status: idle     |
    | ... runs tools   |      | ... waits ...    |
    | status -> idle   |      |                  |
    +------------------+      +------------------+

Key idea: teammates have names, inboxes, and independent loops.

Read this file in this order:
1. MessageBus: how messages are queued and drained.
2. TeammateManager: what persistent teammate state looks like.
3. _teammate_loop / TOOL_HANDLERS: how each named teammate keeps re-entering the same tool loop.

Most common confusion:
- a teammate is not a one-shot subagent
- an inbox message is not yet a full protocol request

Teaching boundary:
this file teaches persistent named workers plus mailboxes.
Approval protocols and autonomous policies are added in later chapters.

---

智能体团队

持久命名智能体 + 基于文件的 JSONL 收件箱。
每个队友在独立线程中运行自己的 agent_loop，通过追加写入的收件箱文件相互通信。

    子智能体（s04）：创建 -> 执行 -> 返回摘要 -> 销毁          （用完即弃）
    队友    （s15）：创建 -> 工作 -> 空闲 -> 工作 -> ... -> 关闭（持久存活）

    .team/config.json 是团队配置文件，记录团队名称与成员列表（名称/角色/状态）。
    .team/inbox/ 是收件箱目录，每位队友对应一个 .jsonl 文件，消息以追加模式写入。

    send_message("alice", "fix bug")：以追加模式打开 alice.jsonl 并写入消息。
    read_inbox("alice")：逐行解析 alice.jsonl，读取后清空文件（drain），返回消息列表。
    spawn_teammate 在新线程中启动队友的 agent_loop，各线程状态独立、互不干扰。

核心思想：队友拥有名称、收件箱和独立的运行循环。

建议阅读顺序：
1. MessageBus：消息如何入队与出队（收件箱的读写逻辑）
2. TeammateManager：持久队友的状态结构（名称、角色、线程、状态）
3. _teammate_loop / TOOL_HANDLERS：每位队友如何反复进入同一个工具调用循环

常见误解：
- 队友不是一次性子智能体——它持续存活并处理多条消息
- 收件箱消息只是原始文本，尚未构成完整的协议请求

教学边界说明：
本文件仅教授"持久命名工作者 + 邮箱"机制。
审批协议与自主决策策略将在后续章节中补充。
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
TEAM_DIR = WORKDIR / ".team" # 团队目录
INBOX_DIR = TEAM_DIR / "inbox" # 收件箱目录

SYSTEM = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

VALID_MSG_TYPES = {
    "message",              # 普通点对点消息
    "broadcast",            # 广播（发给所有队友）
    "shutdown_request",     # 请求某个队友关闭
    "shutdown_response",    # 队友回复"我已关闭"
    "plan_approval",        # 提交一个计划，请求审批
    "plan_approval_response", # 回复审批结果（同意/拒绝）
}


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir # 收件箱目录 .team/inbox/
        self.dir.mkdir(parents=True, exist_ok=True) # 创建收件箱目录

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
        inbox_path = self.dir / f"{to}.jsonl" # 收件箱文件路径 .team/inbox/alice.jsonl
        with open(inbox_path, "a") as f: # 以追加模式打开收件箱文件
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = [] # 消息列表
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("") # 清空收件箱文件
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0 # 广播计数
        for name in teammates: # 遍历所有队友
            if name != sender: # 不发送给自己
                self.send(sender, name, content, "broadcast") # 发送广播消息
                count += 1 # 广播计数加1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- TeammateManager: persistent named agents with config.json --
class TeammateManager:
    """Persistent teammate registry plus worker-loop launcher.
    持久队友注册表 + 工作循环启动器。
    """

    def __init__(self, team_dir: Path):
        self.dir = team_dir # 团队目录 .team/
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json" # 团队配置文件 .team/config.json
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []} # 初始的时候，团队名称是default，成员列表是空

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2)) # 保存团队配置文件

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]: # 遍历所有成员
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        创建一个队友，并启动它的工作循环。
        如果队友已经存在，则更新它的状态和工作角色。
        如果队友不存在，则创建一个新的队友，并更新它的状态和工作角色。
        然后启动它的工作循环。
        """
        member = self._find_member(name) 
        if member:
            if member["status"] not in ("idle", "shutdown"): # 如果队友的状态不是空闲或关闭，则返回错误
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working" # 更新队友的状态为工作
            member["role"] = role # 更新队友的角色
        else:
            member = {"name": name, "role": role, "status": "working"} # 创建一个新的队友，并更新它的状态和工作角色
            self.config["members"].append(member) # 将新的队友添加到成员列表中
        self._save_config() # 保存团队配置文件
        thread = threading.Thread(
            target=self._teammate_loop, # 启动队友的工作循环
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread # 将新的队友添加到线程列表中
        thread.start() # 启动队友的工作循环
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()
        for _ in range(50): # 完成一个任务最多50循环
            inbox = BUS.read_inbox(name) # 读取自己的收件箱
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})
            try:
                response = client.messages.create(
                    model=MODEL,
                    system=sys_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=8000,
                ) # 处理信息
            except Exception:
                break
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use": # 推理完成
                break
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = self._exec(name, block.name, block.input) # 工具执行
                    print(f"  [{name}] {block.name}: {str(output)[:120]}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })
            # subagent 执行结果完成以后需要调用工具send_message将结果发送给队友
            messages.append({"role": "user", "content": results}) # 将工具执行结果添加到消息列表中
        member = self._find_member(name)
        # 任务完成以后，更新队友的状态为空闲
        if member and member["status"] != "shutdown": 
            member["status"] = "idle" # 更新队友的状态为空闲
            self._save_config() # 保存团队配置文件

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
        # 与信息传递相关的工具调用
        if tool_name == "send_message":
            # 具体将消息发给谁由自己inbox中谁给自己发的决定
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
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
        ]

    def list_all(self) -> str:
        # 列出所有的队友以及其状态
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        # 列出成员名称
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


# -- Lead tool dispatch (9 tools) --
TOOL_HANDLERS = {
    "bash":            lambda **kw: _run_bash(kw["command"]),
    "read_file":       lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":      lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":       lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 与agent成员相关
    "spawn_teammate":  lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":  lambda **kw: TEAM.list_all(),
    "send_message":    lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":      lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":       lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
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
    # 工具schema
    {"name": "spawn_teammate", "description": "Spawn a persistent teammate that runs in its own thread.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates with name, role, status.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate's inbox.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
]


def agent_loop(messages: list):
    while True:
        inbox = BUS.read_inbox("lead") # 读取lead的消息盒子
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
            query = input("\033[36ms15 >> \033[0m")
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
