"""内置工具：ask_user。

向用户提出问题并等待回复。
实际实现依赖 S11 编排循环的用户交互机制。
"""

from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata

DEFINITION = ToolDefinition(
    name="ask_user",
    description="向用户提出问题。Agent 在需要澄清或确认时使用此工具。",
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "要向用户提出的问题",
            },
        },
        "required": ["question"],
    },
    metadata=ToolMetadata(
        category="autonomy",
        permission_level="auto_approve",
        readonly=True,
        tags=["autonomy", "user"],
    ),
)


async def handle(args: dict[str, Any]) -> str:
    """返回问题文本，由 S11 编排循环拦截并路由给用户。"""
    return f"[ASK_USER] {args['question']}"
