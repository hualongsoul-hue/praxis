"""内置 JIT 工具——按需加载标识符的完整内容（S7 即时检索）。

标识符索引通过 Prompt 注入呈现给 LLM；本工具让 LLM 能按需把某个标识符
的完整内容拉入上下文（懒加载），仅在 JITRetriever 配置了 ContentLoader 时注册。
"""

from typing import Any

from praxis.context.jit_retrieval import JITRetriever
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.registry import ToolHandler, ToolRegistry

JIT_LOAD_CONTENT = ToolDefinition(
    name="jit_load_content",
    description=(
        "按标识符（来自 <identifier_index>）懒加载其完整内容到上下文。"
        "当只看到轻量标识符、需要其完整正文时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "identifier": {"type": "string", "description": "标识符名称（见标识符索引）"},
        },
        "required": ["identifier"],
    },
    metadata=ToolMetadata(
        category="utility",
        permission_level="auto_approve",
        readonly=True,
        tags=["jit", "context"],
    ),
)


def create_load_content_handler(jit: JITRetriever) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        content = await jit.load_content(args["identifier"])
        if content is None:
            return f"未能加载标识符内容: {args['identifier']}"
        return content

    return handle


def register_jit_tools(registry: ToolRegistry, jit: JITRetriever) -> list[str]:
    """当 JITRetriever 配置了 ContentLoader 时注册懒加载工具。"""
    if jit.content_loader is None:
        return []
    registry.register(JIT_LOAD_CONTENT, create_load_content_handler(jit))
    return [JIT_LOAD_CONTENT.name]
