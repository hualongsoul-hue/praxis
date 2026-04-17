"""内置工具：attempt_completion。

Agent 认为任务已完成时调用，提交最终结果。
实际完成判定由 S11 编排循环 + S10 验证引擎协调。
"""

from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata

DEFINITION = ToolDefinition(
    name="attempt_completion",
    description="Agent 认为当前任务已完成时调用。提交完成结果和摘要。",
    parameters={
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "description": "任务完成结果的详细描述",
            },
            "command": {
                "type": "string",
                "description": "可选的验证命令（如测试命令），供 S10 验证引擎使用",
            },
        },
        "required": ["result"],
    },
    metadata=ToolMetadata(
        category="autonomy",
        permission_level="auto_approve",
        readonly=True,
        tags=["autonomy", "completion"],
    ),
)


async def handle(args: dict[str, Any]) -> str:
    """返回完成信号，由 S11 编排循环处理。"""
    result = args["result"]
    command = args.get("command")
    parts = [f"[ATTEMPT_COMPLETION]\nResult: {result}"]
    if command:
        parts.append(f"Verification command: {command}")
    return "\n".join(parts)
