"""内置工具：write_file。

创建或覆写文件，自动创建不存在的父目录。
"""

from pathlib import Path
from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.filesystem import atomic_write_text
from praxis.tools.policy import ToolPolicy

DEFINITION = ToolDefinition(
    name="write_file",
    description="创建或覆写指定路径的文件。父目录不存在时自动创建。",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件绝对路径",
            },
            "content": {
                "type": "string",
                "description": "文件内容",
            },
        },
        "required": ["file_path", "content"],
    },
    metadata=ToolMetadata(
        category="file_ops",
        permission_level="confirm",
        readonly=False,
        tags=["file", "write"],
    ),
)


def create_handler(sandbox: ToolPolicy):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        path = Path(str(args["file_path"]))
        content = args["content"]

        await atomic_write_text(path, content, sandbox)
        return f"已写入 {len(content)} 字符到 {path}"

    return handle
