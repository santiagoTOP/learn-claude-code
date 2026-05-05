#!/usr/bin/env python3
# Harness: integration -- tools aren't just in your code.
"""
s19_mcp_plugin.py - MCP & Plugin System

This teaching chapter focuses on the smallest useful idea:
external processes can expose tools, and your agent can treat them like
normal tools after a small amount of normalization.

Minimal path:
  1. start an MCP server process
  2. ask it which tools it has
  3. prefix and register those tools
  4. route matching calls to that server

Plugins add one more layer: discovery. A tiny manifest tells the agent which
external server to start.

Key insight: "External tools should enter the same tool pipeline, not form a
completely separate world." In practice that means shared permission checks
and normalized tool_result payloads.

Read this file in this order:
1. CapabilityPermissionGate: external tools still go through the same control gate.
2. MCPClient: how one server connection exposes tool specs and tool calls.
3. PluginLoader: how manifests declare external servers.
4. MCPToolRouter / build_tool_pool: how native and external tools merge into one pool.

Most common confusion:
- a plugin manifest is not an MCP server
- an MCP server is not a single MCP tool
- external capability does not bypass the native permission path

Teaching boundary:
this file teaches the smallest useful stdio MCP path.
Marketplace details, auth flows, reconnect logic, and non-tool capability layers
are intentionally left to bridge docs and later extensions.

中文注解：

s19_mcp_plugin.py - MCP 与插件系统

本章聚焦于最小可用概念：
外部进程可以暴露工具，经过少量标准化处理后，
Agent 就可以像使用普通工具一样使用它们。

最简路径：
  1. 启动一个 MCP 服务进程
  2. 查询它提供哪些工具
  3. 给这些工具加前缀并注册
  4. 将匹配的调用路由到该服务

插件多了一层：发现机制。一个小型 manifest 文件告诉 Agent 要启动哪个外部服务。

关键洞察："外部工具应进入同一个工具管道，而不是形成完全独立的世界。"
实践中意味着共享权限检查和标准化的 tool_result 载荷。

建议按以下顺序阅读本文件：
1. CapabilityPermissionGate：外部工具同样经过相同的控制门。
2. MCPClient：一个服务连接如何暴露工具规格和工具调用。
3. PluginLoader：manifest 如何声明外部服务。
4. MCPToolRouter / build_tool_pool：原生工具与外部工具如何合并为一个工具池。

最常见的误解：
- plugin manifest 不是 MCP 服务
- MCP 服务不是单个 MCP 工具
- 外部能力不会绕过原生权限路径

教学边界：
本文件讲解最小可用的 stdio MCP 路径。
Marketplace 细节、鉴权流程、断线重连逻辑和非工具能力层
有意留给桥接文档和后续扩展。

理解：
启动时
  ->
PluginLoader 找到 manifest
  ->
得到 server 配置
  ->
MCP client 连接 server
  ->
list_tools 并标准化名字
  ->
和 native tools 一起合并进同一个工具池

运行时
  ->
LLM 产出 tool_use
  ->
统一权限闸门
  ->
native route 或 mcp route
  ->
结果标准化
  ->
tool_result 回到同一个主循环
"""

import json
import os
import subprocess
import threading
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
PERMISSION_MODES = ("default", "auto")


class CapabilityPermissionGate:
    """
    Shared permission gate for native tools and external capabilities.

    The teaching goal is simple: MCP does not bypass the control plane.
    Native tools and MCP tools both become normalized capability intents first,
    then pass through the same allow / ask policy.

    共享的权限闸门，用于原生工具和外部能力。

    教学目标很简单：MCP 不绕过控制平面。
    原生工具和 MCP 工具都首先成为标准化能力意图，
    然后通过相同的允许/询问策略。
    """

    READ_PREFIXES = ("read", "list", "get", "show", "search", "query", "inspect") # 读取能力前缀
    HIGH_RISK_PREFIXES = ("delete", "remove", "drop", "shutdown") # 高风险能力前缀

    def __init__(self, mode: str = "default"):
        self.mode = mode if mode in PERMISSION_MODES else "default" # 权限模式

    def normalize(self, tool_name: str, tool_input: dict) -> dict:
        # 判断当前工具调用的风险等级：read、write、high
        if tool_name.startswith("mcp__"):
            _, server_name, actual_tool = tool_name.split("__", 2)
            source = "mcp"
        else:
            server_name = None
            actual_tool = tool_name
            source = "native"

        lowered = actual_tool.lower()
        if actual_tool == "read_file" or lowered.startswith(self.READ_PREFIXES):
            risk = "read"
        elif actual_tool == "bash":
            command = tool_input.get("command", "")
            risk = "high" if any(
                token in command for token in ("rm -rf", "sudo", "shutdown", "reboot")
            ) else "write"
        elif lowered.startswith(self.HIGH_RISK_PREFIXES):
            risk = "high"
        else:
            risk = "write"

        return {
            "source": source,
            "server": server_name,
            "tool": actual_tool,
            "risk": risk,
        }

    def check(self, tool_name: str, tool_input: dict) -> dict:
        intent = self.normalize(tool_name, tool_input) # intent是标准化后的能力意图

        if intent["risk"] == "read":
            return {"behavior": "allow", "reason": "Read capability", "intent": intent}

        if self.mode == "auto" and intent["risk"] != "high":
            return {
                "behavior": "allow",
                "reason": "Auto mode for non-high-risk capability", # 自动模式，非高风险能力
                "intent": intent,
            }

        if intent["risk"] == "high":
            return {
                "behavior": "ask",
                "reason": "High-risk capability requires confirmation", # 高风险能力需要确认
                "intent": intent,
            }

        return {
            "behavior": "ask",
            "reason": "State-changing capability requires confirmation", # 状态改变能力需要确认 
            "intent": intent,
        }

    def ask_user(self, intent: dict, tool_input: dict) -> bool:
        preview = json.dumps(tool_input, ensure_ascii=False)[:200] # 预览工具输入
        source = (
            f"{intent['source']}:{intent['server']}/{intent['tool']}"
            if intent.get("server")
            else f"{intent['source']}:{intent['tool']}" # 来源：工具名称
        )
        print(f"\n  [Permission] {source} risk={intent['risk']}: {preview}") # 打印权限信息
        try:
            answer = input("  Allow? (y/n): ").strip().lower() # 允许/拒绝
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes") # 返回是否允许


permission_gate = CapabilityPermissionGate()


class MCPClient:
    """
    Minimal MCP client over stdio.

    This is enough to teach the core architecture without dragging readers
    through every transport, auth flow, or marketplace detail up front.
    
    基于 stdio 的最简 MCP 客户端。
    这已经足够讲清楚核心架构，
    而不需要一开始就让读者面对所有传输协议、鉴权流程或 Marketplace 细节。

    "postgres": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres"]
        }

    """

    def __init__(self, server_name: str, command: str, args: list = None, env: dict = None):
        self.server_name = server_name # mcp 服务名称
        self.command = command # 启动命令
        self.args = args or [] # 启动参数
        self.env = {**os.environ, **(env or {})}
        self.process = None # 进程对象
        self._request_id = 0 # 请求ID
        self._tools = []  # 缓存工具列表

    def connect(self):
        """Start the MCP server process."""
        try:
            self.process = subprocess.Popen(  # 启动一个新的子进程，不等待它结束（异步）
                [self.command] + self.args, # 要执行的命令，比如 ["python", "my_mcp_server.py"]
                stdin=subprocess.PIPE, # 父进程可以向子进程写数据（发送请求）
                stdout=subprocess.PIPE, # 父进程可以从子进程读数据（接收响应）
                stderr=subprocess.PIPE, # 捕获子进程的错误输出
                env=self.env, # 给子进程传递环境变量（比如 API Key）
                text=True, # 以文本模式读写，而不是字节
            )
            # Send initialize request
            self._send({"method": "initialize", "params": {  # 发送握手请求，MCP 协议规定的第一条消息
                "protocolVersion": "2024-11-05",              # 声明客户端使用的协议版本，双方需要对上
                "capabilities": {},                           # 客户端支持的额外能力，空表示最简实现
                "clientInfo": {"name": "teaching-agent", "version": "1.0"},  # 客户端自我介绍
            }})
            response = self._recv() # 接收握手响应
            if response and "result" in response:
                # Send initialized notification
                self._send({"method": "notifications/initialized"}) # 发送初始化完成通知，表示握手成功
                return True
        except FileNotFoundError:
            print(f"[MCP] Server command not found: {self.command}")
        except Exception as e:
            print(f"[MCP] Connection failed: {e}")
        return False

    def list_tools(self) -> list:
        """Fetch available tools from the server."""
        self._send({"method": "tools/list", "params": {}}) # 发送工具列表请求
        response = self._recv()
        if response and "result" in response:
            self._tools = response["result"].get("tools", []) # 提取工具列表
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool on the server."""
        self._send({"method": "tools/call", "params": {
            "name": tool_name,
            "arguments": arguments,
        }}) # 发送工具调用请求
        response = self._recv() # 接收工具调用响应
        if response and "result" in response:
            content = response["result"].get("content", []) # 提取工具调用结果
            return "\n".join(c.get("text", str(c)) for c in content)
        if response and "error" in response:
            return f"MCP Error: {response['error'].get('message', 'unknown')}"
        return "MCP Error: no response"

    def get_agent_tools(self) -> list:
        """
        Convert MCP tools to agent tool format. 

        Teaching version uses the same simple prefix idea:
        mcp__{server_name}__{tool_name}

        将 mcp 工具转化为 agent 工具格式。
        教学版使用相同的简单前缀方案：
        mcp__{server_name}__{tool_name}
        """
        agent_tools = []
        for tool in self._tools:
            prefixed_name = f"mcp__{self.server_name}__{tool['name']}"
            agent_tools.append({
                "name": prefixed_name, # 添加前缀后的工具名称
                "description": tool.get("description", ""), # 工具描述
                "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}), # 工具输入参数
                "_mcp_server": self.server_name, # 所属 MCP 服务名称
                "_mcp_tool": tool["name"], # 原始工具名称
            })
        return agent_tools

    def disconnect(self):
        """Shut down the server process."""
        if self.process:
            try:
                self._send({"method": "shutdown"}) # 发送关闭请求
                self.process.terminate() # 发送终止信号，让子进程自行退出
                self.process.wait(timeout=5) # 等待子进程退出
            except Exception:
                self.process.kill() # 如果子进程没有退出，则强制杀死
            self.process = None # 重置进程对象

    def _send(self, message: dict):
        if not self.process or self.process.poll() is not None:  # 子进程不存在或已退出则跳过
            return
        self._request_id += 1                                    # 每次发送递增请求 ID，用于匹配响应
        envelope = {"jsonrpc": "2.0", "id": self._request_id, **message}  # 包装成 JSON-RPC 2.0 格式
        line = json.dumps(envelope) + "\n"                       # 序列化为 JSON 字符串，末尾加换行符作为消息分隔符
        try:
            self.process.stdin.write(line)                       # 写入子进程的标准输入
            self.process.stdin.flush()                           # 立即刷新缓冲区，确保数据发出去
        except (BrokenPipeError, OSError):                       # 子进程已关闭管道时静默忽略
            pass

    def _recv(self) -> dict | None:
        if not self.process or self.process.poll() is not None:  # 子进程不存在或已退出则返回 None
            return None
        try:
            line = self.process.stdout.readline()                # 从子进程标准输出读取一行（阻塞直到有数据）
            if line:
                return json.loads(line)                          # 解析 JSON 并返回
        except (json.JSONDecodeError, OSError):                  # JSON 格式错误或管道异常时静默忽略
            pass
        return None                                              # 读到空行（子进程关闭）时返回 None


class PluginLoader:
    """
    Load plugins from .claude-plugin/ directories.

    Teaching version implements the smallest useful plugin flow:
    read a manifest, discover MCP server configs, and register them.
    
    从 .claude-plugin/ 目录加载插件。
    教学版实现了最小可用的插件流程：
    读取 manifest 文件，发现 MCP 服务配置，并注册它们。
    """

    def __init__(self, search_dirs: list = None):
        self.search_dirs = search_dirs or [WORKDIR] # 当前.agents目录
        self.plugins = {}  # name -> manifest

    def scan(self) -> list:
        """Scan directories for .claude-plugin/plugin.json manifests."""
        found = []
        for search_dir in self.search_dirs:
            plugin_dir = Path(search_dir) / ".claude-plugin"
            manifest_path = plugin_dir / "plugin.json" # 插件manifest文件
            """
            {
                "name": "my-db-tools",
                "version": "1.0.0",
                "mcpServers": {
                    "postgres": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-postgres"]
                    }
                }
            }
            # 整个 json 文件就是一个 manifest，里面定义了插件名、版本、提供哪些 MCP server、每个 server 的启动命令是什么
            """
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    name = manifest.get("name", plugin_dir.parent.name)
                    self.plugins[name] = manifest
                    found.append(name)
                except (json.JSONDecodeError, OSError) as e:
                    print(f"[Plugin] Failed to load {manifest_path}: {e}")
        return found

    def get_mcp_servers(self) -> dict:
        """
        Extract MCP server configs from loaded plugins.
        Returns {server_name: {command, args, env}}.
        """
        servers = {}
        for plugin_name, manifest in self.plugins.items():
            for server_name, config in manifest.get("mcpServers", {}).items():
                servers[f"{plugin_name}__{server_name}"] = config
        return servers # 提取里面的mcp工具名称和配置


class MCPToolRouter:
    """
    Routes tool calls to the correct MCP server.

    MCP tools are prefixed mcp__{server}__{tool} and live alongside
    native tools in the same tool pool. The router strips the prefix
    and dispatches to the right MCPClient.

    将工具调用路由到正确的 MCP 服务。
    MCP 工具的名称格式为 mcp__{server}__{tool}，
    和 native 工具一样，都放在同一个工具池中。
    比如 mcp__postgres__query 表示调用 postgres 服务的 query 工具。
    路由器去掉前缀，分发到正确的 MCPClient。比如 mcp__postgres__query 会被路由到 postgres 服务的 MCPClient。
    """

    def __init__(self):
        self.clients = {}  # server_name -> MCPClient

    def register_client(self, client: MCPClient):
        self.clients[client.server_name] = client # 注册 MCPClient 实例，key 是 server_name，value 是 MCPClient 实例

    def is_mcp_tool(self, tool_name: str) -> bool:
        return tool_name.startswith("mcp__")

    def call(self, tool_name: str, arguments: dict) -> str:
        """Route an MCP tool call to the correct server."""
        parts = tool_name.split("__", 2) # 2表示最多分割成3段，比如 mcp__postgres__query 会被分割成 ["mcp", "postgres", "query"]
        if len(parts) != 3: # 如果不是3段，则返回错误
            return f"Error: Invalid MCP tool name: {tool_name}"
        _, server_name, actual_tool = parts # 分割后的三段分别是：前缀、服务名称、工具名称
        client = self.clients.get(server_name) # 根据服务名称获取 MCPClient 实例
        if not client: # 如果获取不到，则返回错误
            return f"Error: MCP server not found: {server_name}" # 返回错误信息
        return client.call_tool(actual_tool, arguments) # 调用工具

    def get_all_tools(self) -> list:
        """Collect tools from all connected MCP servers."""
        tools = []
        for client in self.clients.values():
            tools.extend(client.get_agent_tools()) # 获取所有转化后的 MCP 工具
        return tools


# -- Native tool implementations (same as s02) --
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

def run_read(path: str) -> str:
    try:
        return safe_path(path).read_text()[:50000]
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


NATIVE_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"]),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

NATIVE_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


# -- MCP Tool Router (global) --
mcp_router = MCPToolRouter() # 将工具路由到指定的服务
plugin_loader = PluginLoader() # 发现mcp服务


def build_tool_pool() -> list:
    """
    Assemble the complete tool pool: native + MCP tools.

    Native tools take precedence on name conflicts so the local core remains
    predictable even after external tools are added.
    """
    all_tools = list(NATIVE_TOOLS)
    mcp_tools = mcp_router.get_all_tools() # 获取所有 MCP 工具

    native_names = {t["name"] for t in all_tools}
    for tool in mcp_tools:
        if tool["name"] not in native_names:
            all_tools.append(tool)

    return all_tools


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Dispatch to native handler or MCP router."""
    if mcp_router.is_mcp_tool(tool_name): # 判断是否为mcp工具
        return mcp_router.call(tool_name, tool_input) # 路由到指定的服务
    handler = NATIVE_HANDLERS.get(tool_name)
    if handler:
        return handler(**tool_input) # 调用本地工具
    return f"Unknown tool: {tool_name}" # 返回错误信息


def normalize_tool_result(tool_name: str, output: str, intent: dict | None = None) -> str:
    intent = intent or permission_gate.normalize(tool_name, {}) # 标准化后的能力意图
    status = "error" if "Error:" in output or "MCP Error:" in output else "ok" # 状态：error或ok
    payload = {
        "source": intent["source"], # 来源：工具名称
        "server": intent.get("server"), # 所属 MCP 服务名称
        "tool": intent["tool"], # 原始工具名称
        "risk": intent["risk"], # 风险等级
        "status": status, # 状态
        "preview": output[:500], # 预览结果
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) # 返回标准化后的结果


def agent_loop(messages: list):
    """Agent loop with unified native + MCP tool pool."""
    tools = build_tool_pool() # 构建完整的工具池：原生工具 + MCP 工具

    while True:
        system = (
            f"You are a coding agent at {WORKDIR}. Use tools to solve tasks.\n"
            "You have both native tools and MCP tools available.\n"
            "MCP tools are prefixed with mcp__{server}__{tool}.\n"
            "All capabilities pass through the same permission gate before execution."
        )
        response = client.messages.create(
            model=MODEL, system=system, messages=messages,
            tools=tools, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            decision = permission_gate.check(block.name, block.input or {}) # 权限网关
            try:
                if decision["behavior"] == "deny":
                    output = f"Permission denied: {decision['reason']}"
                elif decision["behavior"] == "ask" and not permission_gate.ask_user(
                    decision["intent"], block.input or {}
                ):
                    output = f"Permission denied by user: {decision['reason']}"
                else:
                    output = handle_tool_call(block.name, block.input or {}) # 执行工具
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {block.name}: {str(output)[:200]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": normalize_tool_result( # 标准化后的结果
                    block.name,
                    str(output),
                    decision.get("intent"),
                ),
            })

        messages.append({"role": "user", "content": results})


# Further upgrades you can add later:
# - more transports
# - auth / approval flows
# - server reconnect and lifecycle management
# - filtering external tools before they reach the model
# - richer plugin installation and update handling


if __name__ == "__main__":
    # Scan for plugins
    found = plugin_loader.scan() # 扫描当前.agents目录下的.claude-plugin/plugin.json文件
    if found:
        print(f"[Plugins loaded: {', '.join(found)}]")
        for server_name, config in plugin_loader.get_mcp_servers().items():
            # 遍历插件中的mcp server，创建 MCPClient 实例
            """
            "postgres": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres"]
                }
            """
            mcp_client = MCPClient(server_name, config.get("command", ""), config.get("args", []))
            if mcp_client.connect():
                mcp_client.list_tools()
                mcp_router.register_client(mcp_client)
                print(f"[MCP] Connected to {server_name}")

    tool_count = len(build_tool_pool()) # 工具数量
    mcp_count = len(mcp_router.get_all_tools()) # MCP 工具数量
    print(f"[Tool pool: {tool_count} tools ({mcp_count} from MCP)]")

    history = []
    while True:
        try:
            query = input("\033[36ms19 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        if query.strip() == "/tools":
            for tool in build_tool_pool():
                prefix = "[MCP] " if tool["name"].startswith("mcp__") else "       "
                print(f"  {prefix}{tool['name']}: {tool.get('description', '')[:60]}")
            continue

        if query.strip() == "/mcp":
            if mcp_router.clients:
                for name, c in mcp_router.clients.items():
                    tools = c.get_agent_tools()
                    print(f"  {name}: {len(tools)} tools")
            else:
                print("  (no MCP servers connected)")
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()

    # Cleanup MCP connections
    for c in mcp_router.clients.values(): # 遍历所有 MCP 客户端
        c.disconnect() # 关闭 MCP 连接
