#!/usr/bin/env python3
# Harness: assembly -- the system prompt is a pipeline, not a string.
"""
s10_system_prompt.py - System Prompt Construction
# 中文：s10_system_prompt.py —— 系统提示词（System Prompt）的组装

This chapter teaches one core idea:
the system prompt should be assembled from clear sections, not written as one
giant hardcoded blob.
# 中文：本章核心观点：系统提示词应由若干清晰区块拼装而成，
# 中文：而不是写成一整块巨型硬编码字符串。

Teaching pipeline:
  1. core instructions
  2. tool listing
  3. skill metadata
  4. memory section
  5. CLAUDE.md chain
  6. dynamic context
# 中文：教学用流水线（组装顺序）：
# 中文：  1. 核心指令
# 中文：  2. 工具列表
# 中文：  3. Skill 元数据
# 中文：  4. 记忆区块
# 中文：  5. CLAUDE.md 链式读取
# 中文：  6. 动态上下文

The builder keeps stable information separate from information that changes
often. A simple DYNAMIC_BOUNDARY marker makes that split visible.
# 中文：组装器把相对稳定的信息与经常变化的信息分开；
# 中文：用简单的 DYNAMIC_BOUNDARY 标记让这种分界在文本中可见。

Per-turn reminders are even more dynamic. They are better injected as a
separate user-role system reminder than mixed blindly into the stable prompt.
# 中文：每轮提醒更加动态，更适合以单独的 user 角色“系统提醒”注入，
# 中文：而不是盲目混进稳定的系统提示词里。

Key insight: "Prompt construction is a pipeline with boundaries, not one
big string."
# 中文：关键洞察：“提示词构建是带边界的流水线，而不是一大段字符串。”

理解：系统提示词应该由若干清晰区块拼装而成，而不是写成一整块巨型硬编码字符串，分块写有利于维护和测试
系统提醒：只在当前轮或当前阶段临时追加的一小段系统信息，不应该永久性地塞进系统提示词中，而是应该在每次会话时重新构建系统提示词。

"""

import datetime
import json
import os
import re
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

DYNAMIC_BOUNDARY = "=== DYNAMIC_BOUNDARY ==="


class SystemPromptBuilder:
    """
    Assemble the system prompt from independent sections.

    The teaching goal here is clarity:
    each section has one source and one responsibility.

    That makes the prompt easier to reason about, easier to test, and easier
    to evolve as the agent grows new capabilities.
    """

    def __init__(self, workdir: Path = None, tools: list = None):
        self.workdir = workdir or WORKDIR # 工作目录，默认是当前目录
        self.tools = tools or [] # 工具列表，默认是空列表
        self.skills_dir = self.workdir / "skills" # 技能目录，默认是工作目录下的 skills 目录
        self.memory_dir = self.workdir / ".memory" # 记忆目录，默认是工作目录下的 .memory 目录

    # -- Section 1: Core instructions --
    def _build_core(self) -> str:
        # 核心指令、agent 的身份信息
        return (
            f"You are a coding agent operating in {self.workdir}.\n"
            "Use the provided tools to explore, read, write, and edit files.\n"
            "Always verify before assuming. Prefer reading files over guessing."
        )

    # -- Section 2: Tool listings --
    def _build_tool_listing(self) -> str:
        # 工具列表
        if not self.tools:
            return ""
        lines = ["# Available tools"]
        for tool in self.tools: # 遍历工具列表
            props = tool.get("input_schema", {}).get("properties", {}) # 获取工具的输入参数
            params = ", ".join(props.keys()) # 将输入参数拼接成字符串
            lines.append(f"- {tool['name']}({params}): {tool['description']}") # 将工具名称和输入参数拼接成字符串
        return "\n".join(lines)

    # -- Section 3: Skill metadata (layer 1 from s05 concept) --
    def _build_skill_listing(self) -> str:
        if not self.skills_dir.exists():
            return ""
        skills = []
        for skill_dir in sorted(self.skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md" # 技能文件，默认是技能目录下的 SKILL.md 文件
            if not skill_md.exists():
                continue
            text = skill_md.read_text() # 读取技能文件内容
            # Parse frontmatter for name + description
            match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
            if not match:
                continue
            meta = {}
            for line in match.group(1).splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            name = meta.get("name", skill_dir.name)
            desc = meta.get("description", "")
            skills.append(f"- {name}: {desc}")
        if not skills:
            return ""
        return "# Available skills\n" + "\n".join(skills)

    # -- Section 4: Memory content --
    def _build_memory_section(self) -> str:
        if not self.memory_dir.exists():
            return ""
        memories = []
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            text = md_file.read_text()
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
            if not match:
                continue
            header, body = match.group(1), match.group(2).strip()
            meta = {}
            for line in header.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            name = meta.get("name", md_file.stem)
            mem_type = meta.get("type", "project")
            desc = meta.get("description", "")
            memories.append(f"[{mem_type}] {name}: {desc}\n{body}")
        if not memories:
            return ""
        return "# Memories (persistent)\n\n" + "\n\n".join(memories)

    # -- Section 5: CLAUDE.md chain --
    def _build_claude_md(self) -> str:
        """
        Load CLAUDE.md files in priority order (all are included):
        1. ~/.claude/CLAUDE.md (user-global instructions)
        2. <project-root>/CLAUDE.md (project instructions)
        3. <current-subdir>/CLAUDE.md (directory-specific instructions)
        第 1 条：~/.claude/CLAUDE.md —— 你机器上的用户全局说明（对所有项目都可能生效）。
        第 2 条：项目根目录下的 CLAUDE.md —— 本项目的说明。
        第 3 条：当前工作子目录下的 CLAUDE.md —— 更细粒度、针对某个目录的说明。
        """
        sources = []

        # User-global
        user_claude = Path.home() / ".claude" / "CLAUDE.md"
        if user_claude.exists():
            # 全局 CLAUDE.md 文件
            sources.append(("user global (~/.claude/CLAUDE.md)", user_claude.read_text()))

        # Project root
        project_claude = self.workdir / "CLAUDE.md"
        if project_claude.exists():
            # 项目级 CLAUDE.md 文件
            sources.append(("project root (CLAUDE.md)", project_claude.read_text()))

        # Subdirectory -- in real CC, this walks from cwd up to project root
        # Teaching: check cwd if different from workdir
        cwd = Path.cwd()
        if cwd != self.workdir:
            subdir_claude = cwd / "CLAUDE.md" # 当前工作子目录下的 CLAUDE.md 文件
            if subdir_claude.exists():
                sources.append((f"subdir ({cwd.name}/CLAUDE.md)", subdir_claude.read_text()))

        if not sources:
            return ""
        parts = ["# CLAUDE.md instructions"] # CLAUDE.md 指令
        for label, content in sources:
            parts.append(f"## From {label}")
            parts.append(content.strip())
        return "\n\n".join(parts) # 将 CLAUDE.md 指令拼接成字符串

    # -- Section 6: Dynamic context --
    def _build_dynamic_context(self) -> str:
        # 动态上下文
        lines = [
            f"Current date: {datetime.date.today().isoformat()}",
            f"Working directory: {self.workdir}",
            f"Model: {MODEL}", # 当前使用的模型
            f"Platform: {os.uname().sysname}", # 当前使用的平台
        ]
        return "# Dynamic context\n" + "\n".join(lines) # 将动态上下文拼接成字符串

    # -- Assemble all sections --
    def build(self) -> str:
        """
        Assemble the full system prompt from all sections.

        Static sections (1-5) are separated from dynamic (6) by
        the DYNAMIC_BOUNDARY marker. In real CC, the static prefix
        is cached across turns to save prompt tokens.
        """
        sections = []

        core = self._build_core()
        if core:
            sections.append(core)

        tools = self._build_tool_listing()
        if tools:
            sections.append(tools)

        skills = self._build_skill_listing()
        if skills:
            sections.append(skills)

        memory = self._build_memory_section()
        if memory:
            sections.append(memory)

        claude_md = self._build_claude_md()
        if claude_md:
            sections.append(claude_md)

        # Static/dynamic boundary
        sections.append(DYNAMIC_BOUNDARY)

        dynamic = self._build_dynamic_context()
        if dynamic:
            sections.append(dynamic)

        return "\n\n".join(sections) # 将所有部分拼接成字符串


def build_system_reminder(extra: str = None) -> dict:
    """
    Build a system-reminder user message for per-turn dynamic content.

    The teaching version keeps reminders outside the stable system prompt so
    short-lived context does not get mixed into the long-lived instructions.
    """
    parts = []
    if extra:
        parts.append(extra)
    if not parts:
        return None
    content = "<system-reminder>\n" + "\n".join(parts) + "\n</system-reminder>" # 系统提醒用户消息
    return {"role": "user", "content": content} # 返回系统提醒用户消息


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

# Global prompt builder
prompt_builder = SystemPromptBuilder(workdir=WORKDIR, tools=TOOLS)


def agent_loop(messages: list):
    """
    Agent loop with assembled system prompt.

    The system prompt is rebuilt each iteration. In real CC, the static
    prefix is cached and only the dynamic suffix changes per turn.
    """
    while True:
        system = prompt_builder.build() # 构建系统提示词
        response = client.messages.create(
            model=MODEL, system=system, messages=messages,
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
    # Show the assembled prompt at startup for educational purposes
    full_prompt = prompt_builder.build()
    section_count = full_prompt.count("\n# ")
    print(f"[System prompt assembled: {len(full_prompt)} chars, ~{section_count} sections]")

    # /prompt command shows the full assembled prompt
    history = []
    while True:
        try:
            query = input("\033[36ms10 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        if query.strip() == "/prompt":
            print("--- System Prompt ---")
            print(prompt_builder.build())
            print("--- End ---")
            continue

        if query.strip() == "/sections":
            prompt = prompt_builder.build()
            for line in prompt.splitlines():
                if line.startswith("# ") or line == DYNAMIC_BOUNDARY:
                    print(f"  {line}")
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
