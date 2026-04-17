"""内置工具：update_plan。

更新当前任务执行计划，通过 S3 持久化存储。
"""

from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.persistence.store import PersistenceStore

PLAN_NAMESPACE = "autonomy"
PLAN_KEY = "current_plan"

DEFINITION = ToolDefinition(
    name="update_plan",
    description="创建或更新当前任务的执行计划。计划内容将持久化存储。",
    parameters={
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "计划内容（Markdown 格式）",
            },
        },
        "required": ["plan"],
    },
    metadata=ToolMetadata(
        category="autonomy",
        permission_level="auto_approve",
        readonly=False,
        tags=["autonomy", "plan"],
    ),
)


def create_handler(store: PersistenceStore | None):
    """创建绑定持久化存储的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        plan = args["plan"]
        if store is not None:
            await store.save(PLAN_NAMESPACE, PLAN_KEY, {"plan": plan})
        return f"计划已更新（{len(plan)} 字符）"

    return handle
