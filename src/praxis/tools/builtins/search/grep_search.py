"""内置工具：grep_search。

在目录中递归搜索匹配正则表达式的文件行。
"""

import re
from pathlib import Path
from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.sandbox import Sandbox

DEFINITION = ToolDefinition(
    name="grep_search",
    description="在指定目录中递归搜索匹配正则表达式的文件内容行。返回匹配的文件路径、行号和内容。",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "正则表达式搜索模式",
            },
            "search_path": {
                "type": "string",
                "description": "搜索目录或文件的绝对路径",
            },
            "includes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "文件名 glob 过滤（如 ['*.py', '*.js']）",
            },
        },
        "required": ["pattern", "search_path"],
    },
    metadata=ToolMetadata(
        category="search",
        permission_level="auto_approve",
        readonly=True,
        tags=["search", "grep"],
    ),
)

MAX_RESULTS = 100


def create_handler(sandbox: Sandbox):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        search_path = sandbox.check_path(args["search_path"])
        pattern_str = args["pattern"]
        includes = args.get("includes")

        try:
            pattern = re.compile(pattern_str)
        except re.error as exc:
            return f"无效的正则表达式: {exc}"

        if not search_path.exists():
            return f"路径不存在: {search_path}"

        files: list[Path]
        if search_path.is_file():
            files = [search_path]
        else:
            files = sorted(search_path.rglob("*"))

        if includes:
            files = [
                f for f in files
                if f.is_file() and any(f.match(g) for g in includes)
            ]
        else:
            files = [f for f in files if f.is_file()]

        matches: list[str] = []
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue

            for line_num, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append(f"{file_path}:{line_num}: {line.rstrip()}")
                    if len(matches) >= MAX_RESULTS:
                        matches.append(f"... 结果截断（超过 {MAX_RESULTS} 条）")
                        return "\n".join(matches)

        if not matches:
            return f"未找到匹配 '{pattern_str}' 的结果"
        return "\n".join(matches)

    return handle
