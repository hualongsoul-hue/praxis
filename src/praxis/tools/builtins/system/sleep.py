"""内置工具：sleep。

让 Agent 等待指定秒数，用于限速或等待外部操作完成。
"""

import asyncio
from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata

MAX_SECONDS = 300

DEFINITION = ToolDefinition(
    name="sleep",
    description="等待指定秒数。可用于限速、轮询等待或给外部操作留出时间。最大 300 秒。",
    parameters={
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "等待秒数（0.1~300），默认 1 秒",
                "default": 1,
            },
        },
    },
    metadata=ToolMetadata(
        category="system",
        permission_level="auto_approve",
        readonly=True,
        tags=["system", "utility"],
    ),
)


async def handle(args: dict[str, Any]) -> str:
    """执行异步睡眠。"""
    raw = args.get("seconds", 1)
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return "错误: seconds 必须是数字"
    seconds = max(0.1, min(seconds, MAX_SECONDS))
    await asyncio.sleep(seconds)
    return f"已等待 {seconds:.1f} 秒"
