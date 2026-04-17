"""内置工具：read_file。

读取指定文件内容，支持行范围限制。
"""

from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.sandbox import Sandbox

DEFINITION = ToolDefinition(
    name="read_file",
    description="读取指定路径的文件内容。支持通过 offset 和 limit 指定行范围。",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件绝对路径",
            },
            "offset": {
                "type": "integer",
                "description": "起始行号（1-indexed），默认从第 1 行开始",
            },
            "limit": {
                "type": "integer",
                "description": "读取行数上限",
            },
        },
        "required": ["file_path"],
    },
    metadata=ToolMetadata(
        category="file_ops",
        permission_level="auto_approve",
        readonly=True,
        tags=["file", "read"],
    ),
)


def create_handler(sandbox: Sandbox):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        path = sandbox.check_path(args["file_path"])

        if not path.exists():
            return f"文件不存在: {path}"
        if not path.is_file():
            return f"路径不是文件: {path}"

        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)

        offset = args.get("offset", 1)
        limit = args.get("limit")

        start = max(0, offset - 1)
        end = start + limit if limit else len(lines)
        selected = lines[start:end]

        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i:>6}\t{line.rstrip()}")
        return "\n".join(numbered)

    return handle
