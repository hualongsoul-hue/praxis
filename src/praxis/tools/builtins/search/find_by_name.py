"""内置工具：find_by_name。

按文件名 glob 模式搜索文件和目录。
"""

from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
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

MAX_RESULTS = 50


def create_handler(sandbox: ToolPolicy):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        search_path = sandbox.check_path(args["search_path"])
        pattern = args["pattern"]
        max_depth = args.get("max_depth")

        if not search_path.exists():
            return f"目录不存在: {search_path}"
        if not search_path.is_dir():
            return f"路径不是目录: {search_path}"

        results: list[str] = []
        for item in sorted(search_path.rglob(pattern)):
            if max_depth is not None:
                try:
                    relative = item.relative_to(search_path)
                    if len(relative.parts) > max_depth:
                        continue
                except ValueError:
                    continue

            kind = "dir" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else 0
            results.append(f"[{kind}] {item}  ({size} bytes)" if kind == "file" else f"[{kind}] {item}/")

            if len(results) >= MAX_RESULTS:
                results.append(f"... 结果截断（超过 {MAX_RESULTS} 条）")
                break

        if not results:
            return f"未找到匹配 '{pattern}' 的文件"
        return "\n".join(results)

    return handle
