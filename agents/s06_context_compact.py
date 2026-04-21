#!/usr/bin/env python3
# Harness: compression -- keep the active context small enough to keep working.
"""
s06_context_compact.py - Context Compact

This teaching version keeps the compact model intentionally small:

1. Large tool output is persisted to disk and replaced with a preview marker.
2. Older tool results are micro-compacted into short placeholders.
3. When the whole conversation gets too large, the agent summarizes it and
   continues from that summary.

The goal is not to model every production branch. The goal is to make the
active-context idea explicit and teachable.

---
中文注解：

本章介绍如何压缩上下文，使活跃上下文始终保持在可用范围内：

1. 较大的工具输出持久化到磁盘，并用预览标记替换原内容。
   （避免大块输出占满上下文窗口）
2. 较旧的工具结果被"微压缩"为简短占位符。
   （保留关键信息，丢弃冗余细节）
3. 当整个对话过长时，由 agent 对其进行摘要，并从摘要处继续执行。
   （相当于给 agent 一个"记忆重置"的机制）

本章的目标不是模拟所有生产环境的分支情况，
而是将"活跃上下文"这一核心概念讲清楚、讲明白。

理解：上下文压缩的核心，不是尽量少字，而是让模型在更短的活跃上下文里，仍然保住继续工作的连续性
受到模型本身位置编码的限制，因此输入的上下文窗口是有限制的，
为了保证模型能够在任务的推进过程中能力获得最清晰的上下文信息，因此需要对上下文窗口做管理。

除了对于工具调用结果和之前的工具调用结果的管理，在对完整的历史对话信息做摘要的是一定需要保留这些信息：
- 当前目标是什么
- 已经做了什么
- 改过哪些文件
- 还有什么没完成
- 哪些决定不能丢
因为这些信息是模型继续工作的重要线索，如果丢失了这些信息，模型将无法继续工作。

"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Keep working step by step, and use compact if the conversation gets too long."
)

CONTEXT_LIMIT = 50000  # 上下文窗口的大小限制
KEEP_RECENT_TOOL_RESULTS = 3  # 保留最近3个工具调用的结果
PERSIST_THRESHOLD = 30000  # 持久化工具调用结果的阈值，对于工具结果大于这个阈值的，需要持久化到磁盘
PREVIEW_CHARS = 2000  # 预览字符的数量
TRANSCRIPT_DIR = WORKDIR / ".transcripts" # 保存完整的历史对话信息的目录
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results" # 保存工具调用结果的目录


@dataclass
class CompactState:
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: list[str] = field(default_factory=list)


def estimate_context_size(messages: list) -> int:
    return len(str(messages))  # 采用字符长度来估算上下文窗口的大小


def track_recent_file(state: CompactState, path: str) -> None:
    # 保留最近访问的5个文件，列表末尾的表示最新访问的文件
    if path in state.recent_files:
        state.recent_files.remove(path) # 先删除后加入表示刚刚又被访问了
    state.recent_files.append(path)
    if len(state.recent_files) > 5:
        state.recent_files[:] = state.recent_files[-5:]


def safe_path(path_str: str) -> Path:
    # 确保路径在工作目录内
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path


def persist_large_output(tool_use_id: str, output: str) -> str:
    # 对工具调用结果进行持久化，如果结果小于阈值，则直接返回
    if len(output) <= PERSIST_THRESHOLD:
        return output

    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    # 如果这个路径不存在表示第一次调用这个工具，那么就应该保存
    # 如果这个路径存在表示已经保存过，那么就应该直接返回预览
    if not stored_path.exists(): 
        stored_path.write_text(output)

    preview = output[:PREVIEW_CHARS]
    # 把绝对路径转换为相对路径，避免暴露工作目录
    rel_path = stored_path.relative_to(WORKDIR)
    return (
        "<persisted-output>\n"
        f"Full output saved to: {rel_path}\n"
        "Preview:\n"
        f"{preview}\n"
        "</persisted-output>"
    )


def collect_tool_result_blocks(messages: list) -> list[tuple[int, int, dict]]:
    blocks = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((message_index, block_index, block))
    return blocks


def micro_compact(messages: list) -> list:
    tool_results = collect_tool_result_blocks(messages) # 收集历史上下文中所有的工具调用结果
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS: # 如果工具调用结果的数量小于保留最近工具调用结果的数量，则直接返回
        return messages

    for _, _, block in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        content = block.get("content", "")
        if not isinstance(content, str) or len(content) <= 120: # 如果内容不是字符串或者长度小于120，则直接跳过
            continue
        block["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
    return messages


def write_transcript(messages: list) -> Path:
    # 在摘要之前，先把完整的历史对话信息保存到磁盘
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as handle:
        for message in messages:
            handle.write(json.dumps(message, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve:\n"
        "1. The current goal\n"
        "2. Important findings and decisions\n"
        "3. Files read or changed\n"
        "4. Remaining work\n"
        "5. User constraints and preferences\n"
        "Be compact but concrete.\n\n"
        f"{conversation}"
    )
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return response.content[0].text.strip()


def compact_history(messages: list, state: CompactState, focus: str | None = None) -> list:
    transcript_path = write_transcript(messages) # 在摘要之前，先把完整的历史对话信息保存到磁盘
    print(f"[transcript saved: {transcript_path}]")

    summary = summarize_history(messages) # 对历史对话信息进行摘要
    if focus: # 模型主动要求压缩时，将目标焦点添加到摘要中
        summary += f"\n\nFocus to preserve next: {focus}"
    if state.recent_files: # 将最近修改的文件列表添加到摘要中
        recent_lines = "\n".join(f"- {path}" for path in state.recent_files)
        summary += f"\n\nRecent files to reopen if needed:\n{recent_lines}"

    state.has_compacted = True
    state.last_summary = summary

    return [{
        "role": "user",
        "content": (
            "This conversation was compacted so the agent can continue working.\n\n"
            f"{summary}"
        ),
    }]


def run_bash(command: str, tool_use_id: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

    output = (result.stdout + result.stderr).strip() or "(no output)"
    return persist_large_output(tool_use_id, output) # 对工具的返回进行检测是否需要持久化到磁盘，同时告诉模型持久化后的结果保存在哪里


def run_read(path: str, tool_use_id: str, state: CompactState, limit: int | None = None) -> str:
    try:
        track_recent_file(state, path)
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]  # 有限制的读取文件内容
        output = "\n".join(lines)
        return persist_large_output(tool_use_id, output) # 对工具的返回进行检测是否需要持久化到磁盘，同时告诉模型持久化后的结果保存在哪里
    except Exception as exc:
        return f"Error: {exc}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        content = file_path.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        file_path.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"


TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "compact",
        "description": "Summarize earlier conversation so work can continue in a smaller context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {"type": "string"},
            },
        },
    },
]


def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def execute_tool(block, state: CompactState) -> str:
    if block.name == "bash":
        return run_bash(block.input["command"], block.id)
    if block.name == "read_file":
        return run_read(block.input["path"], block.id, state, block.input.get("limit"))
    if block.name == "write_file":
        return run_write(block.input["path"], block.input["content"])
    if block.name == "edit_file":
        return run_edit(block.input["path"], block.input["old_text"], block.input["new_text"])
    if block.name == "compact":
        return "Compacting conversation..."
    return f"Unknown tool: {block.name}"


def agent_loop(messages: list, state: CompactState) -> None:
    while True:
        messages[:] = micro_compact(messages)

        if estimate_context_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages, state)

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
        manual_compact = False
        compact_focus = None
        for block in response.content:
            if block.type != "tool_use":
                continue

            output = execute_tool(block, state)
            if block.name == "compact":
                manual_compact = True
                compact_focus = (block.input or {}).get("focus")

            print(f"> {block.name}: {str(output)[:200]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(output),
            })

        messages.append({"role": "user", "content": results})

        if manual_compact:
            print("[manual compact]")
            messages[:] = compact_history(messages, state, focus=compact_focus)


if __name__ == "__main__":
    history = []
    compact_state = CompactState()

    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history, compact_state)

        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()
