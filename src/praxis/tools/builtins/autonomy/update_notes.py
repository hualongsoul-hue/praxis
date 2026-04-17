"""内置工具：update_notes。

更新工作区笔记，通过 S3 持久化存储。
"""

from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.persistence.store import PersistenceStore

NOTES_NAMESPACE = "autonomy"
NOTES_KEY = "workspace_notes"

DEFINITION = ToolDefinition(
    name="update_notes",
    description="创建或更新工作区笔记。笔记内容将持久化存储，跨会话保留。",
    parameters={
        "type": "object",
        "properties": {
            "notes": {
                "type": "string",
                "description": "笔记内容（Markdown 格式）",
            },
        },
        "required": ["notes"],
    },
    metadata=ToolMetadata(
        category="autonomy",
        permission_level="auto_approve",
        readonly=False,
        tags=["autonomy", "notes"],
    ),
)


def create_handler(store: PersistenceStore | None):
    """创建绑定持久化存储的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        notes = args["notes"]
        if store is not None:
            await store.save(NOTES_NAMESPACE, NOTES_KEY, {"notes": notes})
        return f"笔记已更新（{len(notes)} 字符）"

    return handle
