"""内置工具：list_dir。

列出目录内容，包含文件类型和大小信息。
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.policy import ToolPolicy

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


def create_handler(sandbox: ToolPolicy):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        return await asyncio.to_thread(run_list_dir, sandbox, str(args["dir_path"]))

    return handle


def run_list_dir(sandbox: ToolPolicy, raw_path: str) -> str:
    """List and count directory entries under the configured traversal bound."""
    path = sandbox.check_path(raw_path)
    if not path.exists():
        return f"目录不存在: {path}"
    if not path.is_dir():
        return f"路径不是目录: {path}"

    entries: list[str] = []
    traversed = 0
    truncated = False
    for item in sorted(path.iterdir()):
        if traversed >= sandbox.search_max_files:
            truncated = True
            break
        checked = sandbox.check_path(item)
        traversed += 1
        if checked.is_dir():
            sub_count = 0
            for root, directories, files in os.walk(checked, followlinks=False):
                root_path = sandbox.check_path(root)
                directories[:] = [
                    name for name in directories
                    if not (root_path / name).is_symlink()
                    and not (
                        hasattr(Path, "is_junction")
                        and (root_path / name).is_junction()
                    )
                ]
                for descendant in [*directories, *files]:
                    if traversed >= sandbox.search_max_files:
                        truncated = True
                        break
                    sandbox.check_path(root_path / descendant)
                    traversed += 1
                    sub_count += 1
                if truncated:
                    break
            entries.append(f"  {checked.name}/  ({sub_count} items)")
        else:
            entries.append(f"  {checked.name}  ({checked.stat().st_size} bytes)")

    if truncated:
        entries.append("... 目录统计已截断")
    if not entries:
        return f"目录为空: {path}"
    return f"{path}/\n" + "\n".join(entries)
