"""内置工具：code_search。

搜索代码中的类定义、函数定义和变量赋值。
"""

import re
from pathlib import Path
from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.sandbox import Sandbox

DEFINITION = ToolDefinition(
    name="code_search",
    description="在代码文件中搜索类定义、函数定义或符号。支持按语言过滤。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词（函数名、类名、符号名）",
            },
            "search_path": {
                "type": "string",
                "description": "搜索目录的绝对路径",
            },
            "extensions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "文件扩展名过滤（如 ['py', 'js']，不含点号）",
            },
        },
        "required": ["query", "search_path"],
    },
    metadata=ToolMetadata(
        category="search",
        permission_level="auto_approve",
        readonly=True,
        tags=["search", "code"],
    ),
)

CODE_PATTERN = re.compile(
    r"^\s*(?:(?:async\s+)?def|class|[A-Z_][A-Z0-9_]*\s*=)\s",
)
MAX_RESULTS = 50


def create_handler(sandbox: Sandbox):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        search_path = sandbox.check_path(args["search_path"])
        query = args["query"].lower()
        extensions = args.get("extensions")

        if not search_path.exists():
            return f"目录不存在: {search_path}"

        files: list[Path]
        if search_path.is_file():
            files = [search_path]
        else:
            files = sorted(search_path.rglob("*"))

        if extensions:
            ext_set = {f".{e.lstrip('.')}" for e in extensions}
            files = [f for f in files if f.is_file() and f.suffix in ext_set]
        else:
            files = [f for f in files if f.is_file() and f.suffix in {".py", ".js", ".ts", ".go", ".rs", ".java"}]

        results: list[str] = []
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue

            for line_num, line in enumerate(text.splitlines(), 1):
                if CODE_PATTERN.match(line) and query in line.lower():
                    results.append(f"{file_path}:{line_num}: {line.rstrip()}")
                    if len(results) >= MAX_RESULTS:
                        results.append(f"... 结果截断（超过 {MAX_RESULTS} 条）")
                        return "\n".join(results)

        if not results:
            return f"未找到与 '{args['query']}' 相关的代码定义"
        return "\n".join(results)

    return handle
