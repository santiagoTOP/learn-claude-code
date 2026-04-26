#!/usr/bin/env python3
# Harness: persistent tasks -- goals that outlive any single conversation.
"""
s12_task_system.py - Tasks

本章讲解持久化任务系统：把任务写入磁盘，让任务状态不依赖当前对话上下文。

Tasks persist as JSON files in .tasks/ so they survive context compression.
Each task carries a small dependency graph:

任务会以 JSON 文件形式保存在 .tasks/ 目录中，因此即使对话被压缩，
任务状态也不会丢失。每个任务都带有一个小型依赖图：

- blockedBy: what must finish first
- blocks: what this task unlocks later

- blockedBy：当前任务开始前必须先完成的任务
- blocks：当前任务完成后会解锁的后续任务

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], "blocks":[], ...}

    Dependency resolution:
    依赖解析：
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- completing task 1 removes it from task 2's blockedBy
         +--- 完成 task 1 后，会把它从 task 2 的 blockedBy 中移除

Key idea: task state survives compression because it lives on disk, not only
inside the conversation.
These are durable work-graph tasks, not transient runtime execution slots.

核心思想：任务状态能跨越上下文压缩，是因为它保存在磁盘上，
而不只是保存在当前对话里。
这些任务是持久化的工作图节点，不是临时线程、后台槽位或 worker 进程。

Read this file in this order:
1. TaskManager: what a TaskRecord looks like on disk.
2. TOOL_HANDLERS / TOOLS: how task operations enter the same loop as normal tools.
3. agent_loop: how persistent work state is exposed back to the model.

建议按这个顺序阅读：
1. TaskManager：理解 TaskRecord 在磁盘上的结构。
2. TOOL_HANDLERS / TOOLS：理解任务操作如何进入普通工具调用的同一循环。
3. agent_loop：理解持久化工作状态如何暴露回模型。

Most common confusion:
- a task record is a durable work item
- it is not a thread, background slot, or worker process

最常见的混淆：
- task record 是一个持久化工作项
- 它不是线程、后台槽位，也不是 worker 进程

Teaching boundary:
this chapter teaches the durable work graph first.
Runtime execution slots and schedulers arrive later.

教学边界：
本章先讲持久化工作图。
运行时执行槽位和调度器会在后续章节出现。

理解：通过工具接口暴露任务管理功能，让模型可以创建、更新、列出和获取任务。
"""

import json
import os
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
TASKS_DIR = WORKDIR / ".tasks" # 任务目录，默认是当前工作目录下的 .tasks 目录。

SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."


# -- TaskManager: CRUD for a persistent task graph --
class TaskManager:
    """Persistent TaskRecord store.

    Think "work graph on disk", not "currently running worker".
    """

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir # 任务目录，默认是当前工作目录下的 .tasks 目录。
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1 # 下一个任务 ID, 初始化的时候为 1

    def _max_id(self) -> int:
        """
        .tasks/
            task_1.json
            task_2.json
            task_3.json
        """
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")] # [1, 2, 3]
        return max(ids) if ids else 0 # 返回最大的任务 ID。

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text()) # 读取任务

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2)) # 保存任务

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id, "subject": subject, "description": description,
            "status": "pending", "blockedBy": [], "blocks": [], "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2) # 创建任务

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2) # 获取任务

    def update(self, task_id: int, status: str = None, owner: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        task = self._load(task_id) # 加载任务
        if owner is not None:
            task["owner"] = owner # 设置任务所有者，指定是哪个 agent 执行
        if status: # 更新任务状态
            if status not in ("pending", "in_progress", "completed", "deleted"):
                raise ValueError(f"Invalid status: {status}") # 无效状态
            task["status"] = status # 更新任务状态
            # When a task is completed, remove it from all other tasks' blockedBy # 当任务完成时，从所有其他任务的 blockedBy 列表中移除
            if status == "completed":
                self._clear_dependency(task_id) # 清除依赖
        if add_blocked_by: # 给当前任务添加依赖
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by)) # 添加依赖
        if add_blocks: # 当前任务是哪些其他任务的依赖
            task["blocks"] = list(set(task["blocks"] + add_blocks)) # 添加依赖
            # Bidirectional: also update the blocked tasks' blockedBy lists
            for blocked_id in add_blocks: # 遍历添加的依赖
                try:
                    blocked = self._load(blocked_id) # 加载依赖任务
                    if task_id not in blocked["blockedBy"]: # 如果当前任务 ID 不在依赖任务的 blockedBy 列表中
                        blocked["blockedBy"].append(task_id) # 添加当前任务 ID
                        self._save(blocked) # 保存依赖任务
                except ValueError:
                    pass
        self._save(task) # 保存任务 
        return json.dumps(task, indent=2) # 返回任务

    def _clear_dependency(self, completed_id: int):
        """Remove completed_id from all other tasks' blockedBy lists.
        解除受当前任务影响的其他任务的依赖。
        """
        for f in self.dir.glob("task_*.json"): # 遍历当前任务目录下的所有任务文件
            task = json.loads(f.read_text()) # 加载任务
            if completed_id in task.get("blockedBy", []): # 如果完成任务 ID 在其他任务的 blockedBy 列表中
                task["blockedBy"].remove(completed_id) # 移除完成任务 ID 
                self._save(task) # 保存任务

    def list_all(self) -> str:
        # 列出所有任务，并返回任务列表。
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "deleted": "[-]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{blocked}")
        return "\n".join(lines)


TASKS = TaskManager(TASKS_DIR)


# -- Base tool implementations --
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
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 新增任务相关的工具
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner"), kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
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
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status, owner, or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"]}, "owner": {"type": "string", "description": "Set when a teammate claims the task"}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


def agent_loop(messages: list):
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
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms12 >> \033[0m")
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
