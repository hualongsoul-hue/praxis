"""内置记忆工具——将 S6 认知记忆的热路径接口暴露为 LLM 可调用工具。

让 Agent 能够主动写入语义/情景/程序记忆、按需检索、维护记忆，
以及使用 Scratchpad 暂存跨轮次的工作笔记。

这些工具仅在会话装配了 CognitiveMemory 时注册（见 register_memory_tools）。
"""

from typing import Any

from praxis.memory.core import CognitiveMemory
from praxis.memory.scratchpad import Scratchpad
from praxis.models.memory import MemoryType
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.registry import ToolHandler, ToolRegistry

_MEMORY_TYPE_VALUES = [t.value for t in MemoryType]
_SCRATCHPAD_KEYS = sorted(Scratchpad.KNOWN_KEYS)


SAVE_MEMORY = ToolDefinition(
    name="save_memory",
    description=(
        "将一条值得长期记住的事实/经验/操作步骤写入认知记忆。"
        "适用于用户偏好、项目约定、可复用的解决方案等。"
        "记忆会经过整合（去重/合并）后持久化。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "记忆内容"},
            "memory_type": {
                "type": "string",
                "enum": _MEMORY_TYPE_VALUES,
                "description": "记忆类型：semantic（事实）/episodic（事件）/procedural（步骤）/working（临时）",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选标签，便于后续检索",
            },
        },
        "required": ["content"],
    },
    metadata=ToolMetadata(
        category="memory",
        permission_level="auto_approve",
        readonly=False,
        tags=["memory", "write"],
    ),
)

SEARCH_MEMORY = ToolDefinition(
    name="search_memory",
    description=(
        "按语义检索认知记忆，返回最相关的若干条。"
        "当需要回忆此前学到的事实、用户偏好或既往方案时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索查询"},
            "top_k": {
                "type": "integer",
                "description": "返回条数（默认 5）",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    },
    metadata=ToolMetadata(
        category="memory",
        permission_level="auto_approve",
        readonly=True,
        tags=["memory", "search"],
    ),
)

UPDATE_MEMORY = ToolDefinition(
    name="update_memory",
    description="按 ID 更新一条已存在的记忆内容（用于纠正过时或错误的记忆）。",
    parameters={
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "目标记忆 ID"},
            "content": {"type": "string", "description": "新的记忆内容"},
        },
        "required": ["memory_id", "content"],
    },
    metadata=ToolMetadata(
        category="memory",
        permission_level="auto_approve",
        readonly=False,
        tags=["memory", "write"],
    ),
)

DELETE_MEMORY = ToolDefinition(
    name="delete_memory",
    description="按 ID 删除（标记为失效）一条记忆。",
    parameters={
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "目标记忆 ID"},
        },
        "required": ["memory_id"],
    },
    metadata=ToolMetadata(
        category="memory",
        permission_level="auto_approve",
        readonly=False,
        tags=["memory", "write"],
    ),
)

WRITE_SCRATCHPAD = ToolDefinition(
    name="write_scratchpad",
    description=(
        "写入跨轮次/跨窗口暂存的结构化工作笔记（按 key 覆盖），用于长任务的"
        "进度（progress.json）、待办（todos.json）、功能清单（features.json）。"
        "不进入长期记忆，会随检查点持久化。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "enum": _SCRATCHPAD_KEYS,
                "description": "条目键（仅限白名单）",
            },
            "content": {"type": "string", "description": "条目内容（建议 JSON 文本）"},
        },
        "required": ["key", "content"],
    },
    metadata=ToolMetadata(
        category="memory",
        permission_level="auto_approve",
        readonly=False,
        tags=["memory", "scratchpad"],
    ),
)

READ_SCRATCHPAD = ToolDefinition(
    name="read_scratchpad",
    description="读取此前写入 Scratchpad 的工作笔记（progress/todos/features）。",
    parameters={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "enum": _SCRATCHPAD_KEYS,
                "description": "条目键（仅限白名单）",
            },
        },
        "required": ["key"],
    },
    metadata=ToolMetadata(
        category="memory",
        permission_level="auto_approve",
        readonly=True,
        tags=["memory", "scratchpad"],
    ),
)


def create_save_handler(memory: CognitiveMemory) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        raw_type = args.get("memory_type", MemoryType.SEMANTIC.value)
        try:
            mem_type = MemoryType(raw_type)
        except ValueError:
            mem_type = MemoryType.SEMANTIC
        memory_id = await memory.save_memory(
            content=args["content"],
            memory_type=mem_type,
            tags=args.get("tags"),
        )
        return f"已保存记忆（{mem_type.value}），ID: {memory_id}"

    return handle


def create_search_handler(memory: CognitiveMemory) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        top_k = int(args.get("top_k", 5))
        results = await memory.search_memory(query=args["query"], top_k=top_k)
        if not results:
            return "未检索到相关记忆。"
        lines = [
            f"[{r.relevance_score:.2f}] ({r.entry.memory_type.value}) "
            f"id={r.entry.memory_id}: {r.entry.content[:300]}"
            for r in results
        ]
        return "\n".join(lines)

    return handle


def create_update_handler(memory: CognitiveMemory) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        await memory.update_memory(args["memory_id"], args["content"])
        return f"已更新记忆 {args['memory_id']}"

    return handle


def create_delete_handler(memory: CognitiveMemory) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        await memory.delete_memory(args["memory_id"])
        return f"已删除记忆 {args['memory_id']}"

    return handle


def create_write_scratchpad_handler(memory: CognitiveMemory) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        key = args["key"]
        if key not in Scratchpad.KNOWN_KEYS:
            return f"不支持的 Scratchpad 键: {key}，允许值为 {_SCRATCHPAD_KEYS}"
        await memory.write_scratchpad(key, args["content"])
        return f"已写入 Scratchpad: {key}"

    return handle


def create_read_scratchpad_handler(memory: CognitiveMemory) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        key = args["key"]
        if key not in Scratchpad.KNOWN_KEYS:
            return f"不支持的 Scratchpad 键: {key}，允许值为 {_SCRATCHPAD_KEYS}"
        value = await memory.read_scratchpad(key)
        if value is None:
            return f"Scratchpad 中无条目: {key}"
        return str(value)

    return handle


def register_memory_tools(registry: ToolRegistry, memory: CognitiveMemory) -> list[str]:
    """将 S6 记忆热路径接口注册为工具。返回已注册的工具名列表。"""
    registry.register(SAVE_MEMORY, create_save_handler(memory))
    registry.register(SEARCH_MEMORY, create_search_handler(memory))
    registry.register(UPDATE_MEMORY, create_update_handler(memory))
    registry.register(DELETE_MEMORY, create_delete_handler(memory))
    registry.register(WRITE_SCRATCHPAD, create_write_scratchpad_handler(memory))
    registry.register(READ_SCRATCHPAD, create_read_scratchpad_handler(memory))
    return [
        SAVE_MEMORY.name,
        SEARCH_MEMORY.name,
        UPDATE_MEMORY.name,
        DELETE_MEMORY.name,
        WRITE_SCRATCHPAD.name,
        READ_SCRATCHPAD.name,
    ]
