#!/usr/bin/env python3
# Harness: persistence -- remembering across the session boundary.
"""
s09_memory_system.py - Memory System
# 中文：s09_memory_system.py - 记忆系统

This teaching version focuses on one core idea:
some information should survive the current conversation, but not everything
belongs in memory.
# 中文：这个教学版本聚焦一个核心概念：
# 中文：有些信息应当跨越当前会话被保留，但并非所有信息
# 中文：都应该进入记忆。

Use memory for:
  - user preferences
  - repeated user feedback
  - project facts that are NOT obvious from the current code
  - pointers to external resources
# 中文：适合写入记忆的内容：
# 中文：  - 用户偏好
# 中文：  - 用户反复提到的反馈
# 中文：  - 无法从当前代码直接看出的项目事实
# 中文：  - 外部资源指针（链接、文档位置等）

Do NOT use memory for:
  - code structure that can be re-read from the repo
  - temporary task state
  - secrets
# 中文：不应写入记忆的内容：
# 中文：  - 可以从仓库重新读取到的代码结构信息
# 中文：  - 临时任务状态
# 中文：  - 任何密钥或敏感凭证

Storage layout:
  .memory/
    MEMORY.md
    prefer_tabs.md
    review_style.md
    incident_board.md
# 中文：存储目录结构如下（示例）：
# 中文：  .memory/
# 中文：    MEMORY.md
# 中文：    prefer_tabs.md
# 中文：    review_style.md
# 中文：    incident_board.md

Each memory is a small Markdown file with frontmatter.
The agent can save a memory through save_memory(), and the memory index
is rebuilt after each write.
# 中文：每条记忆都是一个带 frontmatter 的小型 Markdown 文件。
# 中文：代理可通过 save_memory() 保存记忆，并在每次写入后
# 中文：重建记忆索引。

An optional "Dream" pass can later consolidate, deduplicate, and prune
stored memories. It is useful, but it is not the first thing readers need
to understand.
# 中文：后续可选的 "Dream" 阶段会对已存记忆做合并、去重和裁剪。
# 中文：这个能力很有用，但不是读者最先需要理解的部分。

Key insight: "Memory only stores cross-session information that is still
worth recalling later and is not easy to re-derive from the current repo."
# 中文：关键洞察：“记忆只存储那些跨会话仍值得回忆，
# 中文：且难以从当前仓库重新推导出的信息。”

理解：记忆系统的主要作用是获取用户的偏好，用户多次纠正过的错误，某些不容易从代码直接看出来的项目约定，
某些外部资源在哪里找等，这些信息应该被存储起来，以便在未来的会话中被使用。

记忆的写入是通过模型调用工具实现的，当模型认为需要保存记忆时，会调用工具来保存记忆。
"""

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

MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md" # 记忆的索引文件
MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_INDEX_LINES = 200  # 这个限定了记忆索引文件中的记忆行数


class MemoryManager:
    """
    Load, build, and save persistent memories across sessions.

    The teaching version keeps memory explicit:
    one Markdown file per memory, plus one compact index file.
    """

    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or MEMORY_DIR # 定义记忆存储的的地址，显示的存储在外部
        self.memories = {}  # name -> {description, type, content}

    def load_all(self):
        """Load MEMORY.md index and all individual memory files."""
        self.memories = {}  # 将所有记忆加载到一个字典中
        if not self.memory_dir.exists():
            return

        # Scan all .md files except MEMORY.md
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            parsed = self._parse_frontmatter(md_file.read_text())
            if parsed:
                name = parsed.get("name", md_file.stem)
                self.memories[name] = {
                    "description": parsed.get("description", ""),
                    "type": parsed.get("type", "project"),
                    "content": parsed.get("content", ""),
                    "file": md_file.name,
                }

        count = len(self.memories)
        if count > 0:
            print(f"[Memory loaded: {count} memories from {self.memory_dir}]")

    def load_memory_prompt(self) -> str:
        """Build a memory section for injection into the system prompt."""
        if not self.memories:
            return ""

        sections = []
        sections.append("# Memories (persistent across sessions)")
        sections.append("")

        # Group by type for readability
        for mem_type in MEMORY_TYPES: 
            # 按照记忆类型进行分组，将相同类型的记忆放在一起
            typed = {k: v for k, v in self.memories.items() if v["type"] == mem_type}
            if not typed:
                continue
            sections.append(f"## [{mem_type}]")
            for name, mem in typed.items():
                sections.append(f"### {name}: {mem['description']}")
                if mem["content"].strip():
                    sections.append(mem["content"].strip())
                sections.append("")

        return "\n".join(sections)

    def save_memory(self, name: str, description: str, mem_type: str, content: str) -> str:
        """
        Save a memory to disk and update the index.

        Returns a status message.
        """
        if mem_type not in MEMORY_TYPES: # 确保记忆类型是有效的
            return f"Error: type must be one of {MEMORY_TYPES}"

        # Sanitize name for filename
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower()) # 归一化当前记忆的名称
        if not safe_name:
            return "Error: invalid memory name"

        self.memory_dir.mkdir(parents=True, exist_ok=True) # 确保当前记忆路径存在

        # Write individual memory file with frontmatter
        frontmatter = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"type: {mem_type}\n"
            f"---\n"
            f"{content}\n"
        )
        file_name = f"{safe_name}.md"
        file_path = self.memory_dir / file_name
        file_path.write_text(frontmatter)

        # Update in-memory store
        self.memories[name] = {
            "description": description,
            "type": mem_type,
            "content": content,
            "file": file_name,
        }

        # Rebuild MEMORY.md index
        self._rebuild_index()

        return f"Saved memory '{name}' [{mem_type}] to {file_path.relative_to(WORKDIR)}"

    def _rebuild_index(self):
        """Rebuild MEMORY.md from current in-memory state, capped at 200 lines.
        Memory Index的存储格式如下：

        # Memory Index
        - prefer_tabs: User prefers tabs for indentation [user]
        - avoid_mock_heavy_tests: User dislikes mock-heavy tests [feedback]

        """
        lines = ["# Memory Index", ""]
        for name, mem in self.memories.items():
            lines.append(f"- {name}: {mem['description']} [{mem['type']}]")
            if len(lines) >= MAX_INDEX_LINES:
                lines.append(f"... (truncated at {MAX_INDEX_LINES} lines)")
                break
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        MEMORY_INDEX.write_text("\n".join(lines) + "\n") # 按行写入到记忆缩影文件中

    def _parse_frontmatter(self, text: str) -> dict | None:
        """Parse --- delimited frontmatter + body content.
        记忆文件的存储格式如下：
        ---
        name: prefer_tabs
        description: User prefers tabs for indentation
        type: user
        ---
        The user explicitly prefers tabs over spaces when editing source files.
        """
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            return None
        header, body = match.group(1), match.group(2)
        result = {"content": body.strip()}
        for line in header.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result


class DreamConsolidator:
    """
    Auto-consolidation of memories between sessions ("Dream").

    This is an optional later-stage feature. Its job is to prevent the memory
    store from growing into a noisy pile by merging, deduplicating, and
    pruning entries over time.
    # 中文：在会话之间自动整合记忆（“Dream”阶段）。
    #
    # 中文：这是一个可选的后期功能。它的作用是通过合并、去重与裁剪，
    # 中文：防止记忆仓库随着时间推移膨胀成杂乱无序的信息堆。
    """
    
    COOLDOWN_SECONDS = 86400       # 24 hours between consolidations，每次整个的间隔24 小时后在进行整合
    SCAN_THROTTLE_SECONDS = 600    # 10 minutes between scan attempts，检查是否允许整合
    MIN_SESSION_COUNT = 5          # need enough data to consolidate，最少需要 5 个会话后再继续整合
    LOCK_STALE_SECONDS = 3600      # PID lock considered stale after 1 hour，一个进程锁文件超过 1 个小时就表示失效了

    # 整理跨会话的记忆的 4 个阶段
    PHASES = [
        "Orient: scan MEMORY.md index for structure and categories",  # 中文：定向阶段，扫描 MEMORY.md 索引以了解结构与分类
        "Gather: read individual memory files for full content",  # 中文：收集阶段，读取各条记忆文件以获取完整内容
        "Consolidate: merge related memories, remove stale entries",  # 中文：整合阶段，合并相关记忆并移除过时条目
        "Prune: enforce 200-line limit on MEMORY.md index",  # 中文：裁剪阶段，确保 MEMORY.md 索引不超过 200 行上限
    ]

    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or MEMORY_DIR
        self.lock_file = self.memory_dir / ".dream_lock"
        self.enabled = True
        self.mode = "default"
        self.last_consolidation_time = 0.0
        self.last_scan_time = 0.0
        self.session_count = 0

    def should_consolidate(self) -> tuple[bool, str]:
        """
        Check 7 gates in sequence. All must pass.
        Returns (can_run, reason) where reason explains the first failed gate.
        # 中文：按顺序检查 7 个闸门条件，必须全部通过。
        # 中文：返回 (can_run, reason)，其中 reason 用于说明第一个未通过的闸门。
        """
        import time

        now = time.time()

        # Gate 1: enabled flag
        if not self.enabled:  # 如果整合功能被禁用，则返回 False
            return False, "Gate 1: consolidation is disabled"

        # Gate 2: memory directory exists and has memory files
        if not self.memory_dir.exists(): # 如果记忆目录不存在，则返回 False
            return False, "Gate 2: memory directory does not exist"
        memory_files = list(self.memory_dir.glob("*.md")) # 获取记忆目录中的所有记忆文件
        # Exclude MEMORY.md itself from the count
        memory_files = [f for f in memory_files if f.name != "MEMORY.md"] # 排除 MEMORY.md 本身
        if not memory_files:
            return False, "Gate 2: no memory files found"  

        # Gate 3: not in plan mode (only consolidate in active modes)
        if self.mode == "plan":  # 如果当前模式是 plan，则返回 False
            return False, "Gate 3: plan mode does not allow consolidation"

        # Gate 4: 24-hour cooldown since last consolidation
        time_since_last = now - self.last_consolidation_time 
        if time_since_last < self.COOLDOWN_SECONDS: # 如果上次整合时间距离现在小于 24 小时，则返回 False
            remaining = int(self.COOLDOWN_SECONDS - time_since_last)
            return False, f"Gate 4: cooldown active, {remaining}s remaining"

        # Gate 5: 10-minute throttle since last scan attempt
        time_since_scan = now - self.last_scan_time
        if time_since_scan < self.SCAN_THROTTLE_SECONDS: # 如果上次扫描时间距离现在小于 10 分钟，则返回 False
            remaining = int(self.SCAN_THROTTLE_SECONDS - time_since_scan)
            return False, f"Gate 5: scan throttle active, {remaining}s remaining"

        # Gate 6: need at least 5 sessions worth of data
        if self.session_count < self.MIN_SESSION_COUNT: # 如果会话数量小于 5，则返回 False 
            return False, f"Gate 6: only {self.session_count} sessions, need {self.MIN_SESSION_COUNT}"

        # Gate 7: no active lock file (check PID staleness)
        if not self._acquire_lock(): # 如果无法获取锁，则返回 False
            return False, "Gate 7: lock held by another process"

        return True, "All 7 gates passed" # 如果所有闸门都通过，则返回 True

    def consolidate(self) -> list[str]:
        """
        Run the 4-phase consolidation process.

        The teaching version returns phase descriptions to make the flow
        visible without requiring an extra LLM pass here.
        """
        import time

        can_run, reason = self.should_consolidate() # 是否可以进行整合
        if not can_run:
            print(f"[Dream] Cannot consolidate: {reason}")
            return []

        print("[Dream] Starting consolidation...")
        self.last_scan_time = time.time()

        completed_phases = []
        # 模拟整合过程，实际的整合过程会复杂得多，需要考虑很多因素
        for i, phase in enumerate(self.PHASES, 1):
            print(f"[Dream] Phase {i}/4: {phase}") 
            completed_phases.append(phase)

        self.last_consolidation_time = time.time()
        self._release_lock() # 释放锁文件
        print(f"[Dream] Consolidation complete: {len(completed_phases)} phases executed")
        return completed_phases

    def _acquire_lock(self) -> bool:
        """
        Acquire a PID-based lock file. Returns False if locked by another
        live process. Stale locks (older than LOCK_STALE_SECONDS) are removed.
        # 中文：获取一个基于 PID 的锁文件。如果被另一个活跃进程锁定，则返回 False。
        # 中文：过时的锁（超过 LOCK_STALE_SECONDS 的锁）会被删除。
        """
        import time

        if self.lock_file.exists():
            try: # 如果锁文件存在，则尝试读取锁文件内容
                lock_data = self.lock_file.read_text().strip()
                pid_str, timestamp_str = lock_data.split(":", 1) # 分割锁文件内容，获取 PID 和时间戳
                pid = int(pid_str)
                lock_time = float(timestamp_str) # 将时间戳转换为浮点数 用于后续的过时检查

                # Check if lock is stale
                if (time.time() - lock_time) > self.LOCK_STALE_SECONDS: # 如果锁文件的时间戳距离现在超过 1 小时，则删除锁文件
                    print(f"[Dream] Removing stale lock from PID {pid}")
                    self.lock_file.unlink() # 删除锁文件
                else:
                    # Check if owning process is still alive
                    try: # 如果进程仍然存活，则返回 False
                        os.kill(pid, 0) # os.kill(pid, 0) 用于检查进程是否存活
                        return False  # process alive, lock is valid
                    except OSError:
                        print(f"[Dream] Removing lock from dead PID {pid}")
                        self.lock_file.unlink() # 删除锁文件, 因为进程已经不存在
            except (ValueError, OSError):
                # Corrupted lock file, remove it
                self.lock_file.unlink(missing_ok=True) # 删除锁文件, 因为锁文件已经损坏

        # Write new lock
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True) # 创建记忆目录
            self.lock_file.write_text(f"{os.getpid()}:{time.time()}") # 写入锁文件
            return True
        except OSError: 
            return False # 如果无法写入锁文件，则返回 False, 表示有其他进程正在使用锁文件

    def _release_lock(self):
        """Release the lock file if we own it."""
        # 中文：释放锁文件，如果当前进程拥有锁文件。
        try:
            if self.lock_file.exists():
                lock_data = self.lock_file.read_text().strip() # 读取锁文件内容
                pid_str = lock_data.split(":")[0] # 分割锁文件内容，获取 PID
                if int(pid_str) == os.getpid():
                    self.lock_file.unlink() # 删除锁文件
        except (ValueError, OSError):
            pass # 如果无法删除锁文件，则忽略错误


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


# Global memory manager
memory_mgr = MemoryManager()


# 定义一个保存记忆的工具函数
def run_save_memory(name: str, description: str, mem_type: str, content: str) -> str:
    return memory_mgr.save_memory(name, description, mem_type, content)


TOOL_HANDLERS = {
    "bash":         lambda **kw: run_bash(kw["command"]),
    "read_file":    lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":   lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":    lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "save_memory":  lambda **kw: run_save_memory(kw["name"], kw["description"], kw["type"], kw["content"]),
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
    {"name": "save_memory", "description": "Save a persistent memory that survives across sessions.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Short identifier (e.g. prefer_tabs, db_schema)"},
         "description": {"type": "string", "description": "One-line summary of what this memory captures"},
         "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"],
                  "description": "user=preferences, feedback=corrections, project=non-obvious project conventions or decision reasons, reference=external resource pointers"},
         "content": {"type": "string", "description": "Full memory content (multi-line OK)"},
     }, "required": ["name", "description", "type", "content"]}},
]

# 告诉模型什么时候应该记忆什么时候不应该记忆
MEMORY_GUIDANCE = """
When to save memories:
- User states a preference ("I like tabs", "always use pytest") -> type: user
- User corrects you ("don't do X", "that was wrong because...") -> type: feedback
- You learn a project fact that is not easy to infer from current code alone
  (for example: a rule exists because of compliance, or a legacy module must
  stay untouched for business reasons) -> type: project
- You learn where an external resource lives (ticket board, dashboard, docs URL)
  -> type: reference

When NOT to save:
- Anything easily derivable from code (function signatures, file structure, directory layout)
- Temporary task state (current branch, open PR numbers, current TODOs)
- Secrets or credentials (API keys, passwords)
# 中文：何时保存记忆：
# 中文：- 用户表达了稳定偏好（如 “我喜欢 tabs”、“始终使用 pytest”）-> type: user
# 中文：- 用户纠正了你（如 “不要这样做 X”、“刚才不对，因为...”）-> type: feedback
# 中文：- 你得知一个难以仅从当前代码推断的项目事实
# 中文：  （例如：某条规则出于合规要求而存在，或某个遗留模块因业务原因不能改动）-> type: project
# 中文：- 你得知某个外部资源的位置（工单看板、仪表盘、文档 URL）-> type: reference
#
# 中文：何时不要保存记忆：
# 中文：- 能从代码轻易推导出的信息（函数签名、文件结构、目录布局）
# 中文：- 临时任务状态（当前分支、打开的 PR 编号、当前 TODO）
# 中文：- 密钥或凭据（API Key、密码）
"""


def build_system_prompt() -> str:
    """Assemble system prompt with memory content included."""
    parts = [f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."]

    # Inject memory content if available
    memory_section = memory_mgr.load_memory_prompt()
    if memory_section:
        parts.append(memory_section)

    parts.append(MEMORY_GUIDANCE)
    return "\n\n".join(parts)


def agent_loop(messages: list):
    """
    Agent loop with memory-aware system prompt.

    The system prompt is rebuilt each call so newly saved memories
    are visible in the next LLM turn within the same session.
    """
    while True:
        system = build_system_prompt() # 系统提示词
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
    # Load existing memories at session start
    memory_mgr.load_all() # 加载所有记忆
    mem_count = len(memory_mgr.memories) # 获取记忆数量
    if mem_count:
        print(f"[{mem_count} memories loaded into context]")
    else:
        print("[No existing memories. The agent can create them with save_memory.]")

    history = []
    while True:
        try:
            query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # /memories command to list current memories
        if query.strip() == "/memories":
            if memory_mgr.memories:
                for name, mem in memory_mgr.memories.items():
                    print(f"  [{mem['type']}] {name}: {mem['description']}")
            else:
                print("  (no memories)")
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
