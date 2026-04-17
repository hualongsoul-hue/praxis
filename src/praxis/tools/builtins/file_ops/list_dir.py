"""内置工具：list_dir。

列出目录内容，包含文件类型和大小信息。
"""

from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.sandbox import Sandbox

DEFINITION = ToolDefinition(
    name="list_dir",
    description="列出指定目录下的文件和子目录，显示类型和大小信息。",
    parameters={
        "type": "object",
        "properties": {
            "dir_path": {
                "type": "string",
                "description": "目录绝对路径",
            },
        },
        "required": ["dir_path"],
    },
    metadata=ToolMetadata(
        category="file_ops",
        permission_level="auto_approve",
        readonly=True,
        tags=["file", "list"],
    ),
)


def create_handler(sandbox: Sandbox):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        path = sandbox.check_path(args["dir_path"])

        if not path.exists():
            return f"目录不存在: {path}"
        if not path.is_dir():
            return f"路径不是目录: {path}"

        entries: list[str] = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                sub_count = sum(1 for _ in item.rglob("*"))
                entries.append(f"  {item.name}/  ({sub_count} items)")
            else:
                size = item.stat().st_size
                entries.append(f"  {item.name}  ({size} bytes)")

        if not entries:
            return f"目录为空: {path}"
        return f"{path}/\n" + "\n".join(entries)

    return handle
