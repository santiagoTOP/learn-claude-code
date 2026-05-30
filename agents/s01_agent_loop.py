#!/usr/bin/env python3
# Harness: the loop -- keep feeding real tool results back into the model.
"""
s01_agent_loop.py - The Agent Loop / 智能体循环

EN: This file teaches the smallest useful coding-agent pattern:

    user message
      -> model reply
      -> if tool_use: execute tools
      -> write tool_result back to messages
      -> continue

It intentionally keeps the loop small, but still makes the loop state explicit
so later chapters can grow from the same structure.

中文：本文件演示最小但仍有实用价值的编程智能体模式：

    用户消息
      -> 模型回复
      -> 若使用工具：执行工具
      -> 将 tool_result 写回消息列表
      -> 继续循环

有意保持循环精简，同时把循环状态写清楚，以便后续章节在同一套结构上扩展。

model response:
Message(
    id='msg_20260402232218805fa24dc98d4fdd', 
    container=None, 
    content=[
        ToolUseBlock(
            id='call_3e9af27bf53342af9c51b0f2', 
            caller=None, 
            input={'command': 'ls -la /Users/tngpng/Documents/code/learn-claude-code/agents'}, 
            name='bash', 
            type='tool_use'
        )
    ], 
    model='glm-4.7', 
    role='assistant', 
    stop_details=None, 
    stop_reason='tool_use', 
    stop_sequence=None, 
    type='message', 
    usage=Usage(
        cache_creation=None, 
        cache_creation_input_tokens=None, 
        cache_read_input_tokens=128, 
        inference_geo=None, 
        input_tokens=60, 
        output_tokens=27, 
        server_tool_use=ServerToolUsage(web_fetch_requests=None, web_search_requests=0), 
        service_tier='standard'
    )
)

理解：基本的 agent 循环，主要是通过循环来让模型持续推理和执行工具。

https://platform.claude.com/docs/zh-CN/intro
https://platform.claude.com/docs/zh-CN/agents-and-tools/tool-use/build-a-tool-using-agent
"""

import os
import subprocess
from dataclasses import dataclass

try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly."
)

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command in the current workspace.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


@dataclass
class LoopState:
    # The minimal loop state: history, loop count, and why we continue.
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous): # 确保命令的安全性
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "(no output)"


def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def execute_tool_calls(response_content) -> list[dict]:
    results = []
    for block in response_content:
        if block.type != "tool_use":
            continue
        command = block.input["command"]
        print(f"\033[33m$ {command}\033[0m")
        output = run_bash(command)
        print(output[:200])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
    return results


def run_one_turn(state: LoopState) -> bool:
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000,
    )
    state.messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use": 
        state.transition_reason = None # 记录模型状态转移的原因，由推理变成行动
        return False

    results = execute_tool_calls(response.content)
    if not results:
        state.transition_reason = None
        return False

    state.messages.append({"role": "user", "content": results})
    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True


def agent_loop(state: LoopState) -> None:
    while run_one_turn(state):
        pass


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        state = LoopState(messages=history) # messages和history是同一个列表，拥有相同的引用
        agent_loop(state)
        # print("state: ", state)
        # print("history: ", history)
        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()
