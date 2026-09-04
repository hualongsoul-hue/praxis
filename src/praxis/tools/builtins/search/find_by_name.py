"""内置工具：find_by_name。

按文件名 glob 模式搜索文件和目录。
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.builtins.search.common import append_truncation, create_search_budget
from praxis.tools.policy import ToolPolicy

DEFINITION = ToolDefinition(
    name="find_by_name",
    description="在指定目录中递归搜索匹配 glob 模式的文件和目录。返回匹配路径列表。",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "文件名 glob 模式（如 '*.py'、'test_*'）",
            },
            "search_path": {
                "type": "string",
                "description": "搜索起始目录的绝对路径",
            },
            "max_depth": {
                "type": "integer",
                "description": "最大搜索深度",
            },
        },
        "required": ["pattern", "search_path"],
    },
    metadata=ToolMetadata(
        category="search",
        permission_level="auto_approve",
        readonly=True,
        tags=["search", "find"],
    ),
)

def create_handler(sandbox: ToolPolicy):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        return await asyncio.to_thread(run_find_by_name, sandbox, args)

    return handle


def run_find_by_name(sandbox: ToolPolicy, args: dict[str, Any]) -> str:
    """Run bounded name matching without following symlinks or junctions."""
    search_path = sandbox.check_path(args["search_path"])
    pattern = str(args["pattern"])
    max_depth_value = args.get("max_depth")
    max_depth = max_depth_value if isinstance(max_depth_value, int) else None
    if not search_path.exists():
        return f"目录不存在: {search_path}"
    if not search_path.is_dir():
        return f"路径不是目录: {search_path}"

    budget = create_search_budget(sandbox)
    results: list[str] = []
    for root, directories, files in os.walk(search_path, followlinks=False):
        root_path = sandbox.check_path(root)
        relative_root = root_path.relative_to(search_path)
        depth = len(relative_root.parts)
        directories[:] = sorted(
            name for name in directories
            if not (root_path / name).is_symlink()
            and not (
                hasattr(Path, "is_junction")
                and (root_path / name).is_junction()
            )
            and (max_depth is None or depth < max_depth)
        )
        candidates = [*(root_path / name for name in directories)]
        candidates.extend(root_path / name for name in sorted(files))
        for item in candidates:
            if max_depth is not None and len(item.relative_to(search_path).parts) > max_depth:
                continue
            if not budget.accept_file():
                break
            if not item.match(pattern):
                continue
            checked = sandbox.check_path(item)
            kind = "dir" if checked.is_dir() else "file"
            size = checked.stat().st_size if kind == "file" else 0
            if not budget.accept_match():
                break
            results.append(
                f"[{kind}] {checked}  ({size} bytes)"
                if kind == "file"
                else f"[{kind}] {checked}/"
            )
        if budget.truncated:
            break

    append_truncation(results, budget)
    if not results:
        return f"未找到匹配 '{pattern}' 的文件"
    return "\n".join(results)
