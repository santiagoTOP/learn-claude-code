#!/usr/bin/env python3
# Harness: safety -- the pipeline between intent and execution.
"""
s07_permission_system.py - Permission System
【中文】权限系统：在「意图」与「执行」之间插入可编排的安全流水线。

Every tool call passes through a permission pipeline before execution.
【中文】每次工具调用在执行前都会先走一遍权限流水线，而不是直接执行。

Teaching pipeline:
  1. deny rules
  2. mode check
  3. allow rules
  4. ask user
【中文】教学用流水线顺序：① 拒绝规则 → ② 模式检查 → ③ 允许规则 → ④ 询问用户。

This version intentionally teaches three modes first:
  - default
  - plan
  - auto
【中文】本版有意先只讲三种模式，便于循序渐进：default / plan / auto。

That is enough to build a real, understandable permission system without
burying readers under every advanced policy branch on day one.
【中文】这样足以搭出真实、可读懂的权限模型，又避免第一天就堆满高级策略分支。

Key insight: "Safety is a pipeline, not a boolean."
【中文】要点：安全是流水线式的多层判断，而不是单一的「允许/禁止」布尔开关。

理解：
- 任何工具调用，都不应该直接执行；中间必须先过一条权限管道
- “意图”不能直接变成“执行”，中间必须经过权限检查
"""

import json
import os
import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# -- Permission modes --
# Teaching version starts with three clear modes first.
MODES = ("default", "plan", "auto")

READ_ONLY_TOOLS = {"read_file", "bash_readonly"}

# Tools that modify state
WRITE_TOOLS = {"write_file", "edit_file", "bash"}


# -- Bash security validation --
class BashSecurityValidator:
    """
    Validate bash commands for obviously dangerous patterns.

    The teaching version deliberately keeps this small and easy to read.
    First catch a few high-risk patterns, then let the permission pipeline
    decide whether to deny or ask the user.
    """

    VALIDATORS = [
        ("shell_metachar", r"[;&|`$]"),       # shell metacharacters，shell元字符
        ("sudo", r"\bsudo\b"),                 # privilege escalation，权限提升
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"),  # recursive delete ，递归删除
        ("cmd_substitution", r"\$\("),          # command substitution，命令替换
        ("ifs_injection", r"\bIFS\s*="),        # IFS manipulation，IFS注入
    ]

    def validate(self, command: str) -> list:
        """
        Check a bash command against all validators.

        Returns list of (validator_name, matched_pattern) tuples for failures.
        An empty list means the command passed all validators.
        """
        failures = []
        for name, pattern in self.VALIDATORS:
            if re.search(pattern, command):
                failures.append((name, pattern))
        return failures

    def is_safe(self, command: str) -> bool:
        """Convenience: returns True only if no validators triggered."""
        return len(self.validate(command)) == 0

    def describe_failures(self, command: str) -> str:
        """Human-readable summary of validation failures."""
        failures = self.validate(command)
        if not failures:
            return "No issues detected"
        parts = [f"{name} (pattern: {pattern})" for name, pattern in failures]
        return "Security flags: " + ", ".join(parts)


# -- Workspace trust --
def is_workspace_trusted(workspace: Path = None) -> bool:
    """
    Check if a workspace has been explicitly marked as trusted.
    【中文】判断当前工作区是否已被用户显式标记为「受信任」。

    The teaching version uses a simple marker file. A more complete system
    can layer richer trust flows on top of the same idea.
    【中文】教学版用「是否存在标记文件」这一种简单方式；完整系统可在此思路上叠加更丰富的信任流程。
    """
    ws = workspace or WORKDIR
    trust_marker = ws / ".claude" / ".claude_trusted" # 标记的受信任工作区
    return trust_marker.exists()


# Singleton validator instance used by the permission pipeline
bash_validator = BashSecurityValidator()


# -- Permission rules --
# Rules are checked in order: first match wins.
# Format: {"tool": "<tool_name_or_*>", "path": "<glob_or_*>", "behavior": "allow|deny|ask"}
DEFAULT_RULES = [
    # Always deny dangerous patterns
    {"tool": "bash", "content": "rm -rf /", "behavior": "deny"},
    {"tool": "bash", "content": "sudo *", "behavior": "deny"},
    # Allow reading anything
    {"tool": "read_file", "path": "*", "behavior": "allow"},
]


class PermissionManager:
    """
    Manages permission decisions for tool calls.

    Pipeline: deny_rules -> mode_check -> allow_rules -> ask_user

    The teaching version keeps the decision path short on purpose so readers
    can implement it themselves before adding more advanced policy layers.
    """

    def __init__(self, mode: str = "default", rules: list = None):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from {MODES}")
        self.mode = mode
        self.rules = rules or list(DEFAULT_RULES)
        # Simple denial tracking helps surface when the agent is repeatedly
        # asking for actions the system will not allow.
        self.consecutive_denials = 0 # 连续拒绝的次数
        self.max_consecutive_denials = 3 # 最大连续拒绝的次数

    def check(self, tool_name: str, tool_input: dict) -> dict:
        """
        Returns: {"behavior": "allow"|"deny"|"ask", "reason": str}
        """
        # Step 0: Bash security validation (before deny rules)
        # Teaching version checks early for clarity.
        if tool_name == "bash":
            command = tool_input.get("command", "")
            failures = bash_validator.validate(command)
            if failures:
                # Severe patterns (sudo, rm_rf) get immediate deny
                # 严重模式（sudo, rm_rf）立即拒绝
                severe = {"sudo", "rm_rf"}
                severe_hits = [f for f in failures if f[0] in severe]
                if severe_hits:
                    desc = bash_validator.describe_failures(command)
                    return {"behavior": "deny",
                            "reason": f"Bash validator: {desc}"}
                # Other patterns escalate to ask (user can still approve)
                # 其他模式（cmd_substitution, ifs_injection）升级为询问（用户可以仍然批准）
                desc = bash_validator.describe_failures(command)
                return {"behavior": "ask",
                        "reason": f"Bash validator flagged: {desc}"}

        # Step 1: Deny rules (bypass-immune, checked first always)
        for rule in self.rules:
            # 如果规则不是拒绝规则，则跳过
            if rule["behavior"] != "deny":
                continue
            if self._matches(rule, tool_name, tool_input):
                return {"behavior": "deny",
                        "reason": f"Blocked by deny rule: {rule}"}

        # Step 2: Mode-based decisions
        if self.mode == "plan":
            # Plan mode: deny all write operations, allow reads
            # 计划模式：拒绝所有写操作，允许读操作
            if tool_name in WRITE_TOOLS:
                return {"behavior": "deny",
                        "reason": "Plan mode: write operations are blocked"}
            return {"behavior": "allow", "reason": "Plan mode: read-only allowed"}

        if self.mode == "auto":
            # Auto mode: auto-allow read-only tools, ask for writes
            if tool_name in READ_ONLY_TOOLS or tool_name == "read_file":
                return {"behavior": "allow",
                        "reason": "Auto mode: read-only tool auto-approved"}
            # Teaching: fall through to allow rules, then ask
            pass

        # Step 3: Allow rules
        # 使用default模式的时候，会走这个步骤
        for rule in self.rules:
            if rule["behavior"] != "allow":
                continue
            if self._matches(rule, tool_name, tool_input):
                self.consecutive_denials = 0
                return {"behavior": "allow",
                        "reason": f"Matched allow rule: {rule}"}

        # Step 4: Ask user (default behavior for unmatched tools)
        return {"behavior": "ask",
                "reason": f"No rule matched for {tool_name}, asking user"}

    def ask_user(self, tool_name: str, tool_input: dict) -> bool:
        """Interactive approval prompt. Returns True if approved."""
        # 交互式批准提示。返回True如果批准。
        preview = json.dumps(tool_input, ensure_ascii=False)[:200]
        print(f"\n  [Permission] {tool_name}: {preview}")
        try:
            answer = input("  Allow? (y/n/always): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if answer == "always":
            # Add permanent allow rule for this tool
            # 动态添加允许规则
            self.rules.append({"tool": tool_name, "path": "*", "behavior": "allow"})
            self.consecutive_denials = 0
            return True
        if answer in ("y", "yes"):
            self.consecutive_denials = 0
            return True

        # Track denials for circuit breaker
        # 跟踪拒绝次数，用于熔断器
        self.consecutive_denials += 1
        if self.consecutive_denials >= self.max_consecutive_denials:
            print(f"  [{self.consecutive_denials} consecutive denials -- "
                  "consider switching to plan mode]")
        return False

    def _matches(self, rule: dict, tool_name: str, tool_input: dict) -> bool:
        """Check if a rule matches the tool call."""
        # 根据工具名称、工具执行路径以及工具输入内容，判断是否匹配规则
        # Tool name match
        if rule.get("tool") and rule["tool"] != "*":
            if rule["tool"] != tool_name:
                return False
        # Path pattern match
        if "path" in rule and rule["path"] != "*":
            path = tool_input.get("path", "")
            if not fnmatch(path, rule["path"]):
                return False
        # Content pattern match (for bash commands)
        if "content" in rule:
            command = tool_input.get("command", "")
            if not fnmatch(command, rule["content"]):
                return False
        return True


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
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
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
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
]

SYSTEM = f"""You are a coding agent at {WORKDIR}. Use tools to solve tasks.
The user controls permissions. Some tool calls may be denied."""


def agent_loop(messages: list, perms: PermissionManager):
    """
    The permission-aware agent loop.

    For each tool call:
      1. LLM requests tool use
      2. Permission pipeline checks: deny_rules -> mode -> allow_rules -> ask
      3. If allowed: execute tool, return result
      4. If denied: return rejection message to LLM
    """
    while True:
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

            # -- Permission check --
            decision = perms.check(block.name, block.input or {})

            if decision["behavior"] == "deny":
                output = f"Permission denied: {decision['reason']}"
                print(f"  [DENIED] {block.name}: {decision['reason']}")

            elif decision["behavior"] == "ask":
                if perms.ask_user(block.name, block.input or {}):
                    handler = TOOL_HANDLERS.get(block.name)
                    output = handler(**(block.input or {})) if handler else f"Unknown: {block.name}"
                    print(f"> {block.name}: {str(output)[:200]}")
                else:
                    output = f"Permission denied by user for {block.name}" # 告诉模型，用户拒绝了
                    print(f"  [USER DENIED] {block.name}")

            else:  # allow
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**(block.input or {})) if handler else f"Unknown: {block.name}"
                print(f"> {block.name}: {str(output)[:200]}")

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(output),
            })

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    # Choose permission mode at startup
    print("Permission modes: default, plan, auto")
    mode_input = input("Mode (default): ").strip().lower() or "default"
    if mode_input not in MODES:
        mode_input = "default"

    perms = PermissionManager(mode=mode_input)
    print(f"[Permission mode: {mode_input}]")

    history = []
    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # /mode command to switch modes at runtime
        if query.startswith("/mode"):
            parts = query.split() # 按照空格切分命令
            if len(parts) == 2 and parts[1] in MODES:
                perms.mode = parts[1]
                print(f"[Switched to {parts[1]} mode]")
            else:
                print(f"Usage: /mode <{'|'.join(MODES)}>") # 告诉用户可用的模式
            continue

        # /rules command to show current rules
        if query.strip() == "/rules":
            for i, rule in enumerate(perms.rules):
                print(f"  {i}: {rule}")
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history, perms)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
