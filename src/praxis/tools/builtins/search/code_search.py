"""内置工具：code_search。

搜索代码中的类定义、函数定义和变量赋值。
"""

import asyncio
import re
from typing import Any, cast

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.builtins.search.common import (
    append_truncation,
    collect_search_files,
    create_search_budget,
    read_search_text,
)
from praxis.tools.policy import ToolPolicy

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
def create_handler(sandbox: ToolPolicy):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        return await asyncio.to_thread(run_code_search, sandbox, args)

    return handle


def run_code_search(sandbox: ToolPolicy, args: dict[str, Any]) -> str:
    """Run bounded code-definition search in a worker thread."""
    search_path = sandbox.check_path(args["search_path"])
    query = str(args["query"]).lower()
    extensions_value = args.get("extensions")
    extensions = (
        {
            f".{str(value).lstrip('.')}"
            for value in cast(list[object], extensions_value)
        }
        if isinstance(extensions_value, list)
        else {".py", ".js", ".ts", ".go", ".rs", ".java"}
    )
    if not search_path.exists():
        return f"目录不存在: {search_path}"

    budget = create_search_budget(sandbox)
    files = collect_search_files(search_path, sandbox, budget)
    results: list[str] = []
    for file_path in files:
        if file_path.suffix not in extensions:
            continue
        text = read_search_text(file_path, budget)
        if text is None:
            break
        for line_num, line in enumerate(text.splitlines(), 1):
            if CODE_PATTERN.match(line) and query in line.lower():
                if not budget.accept_match():
                    break
                results.append(f"{file_path}:{line_num}: {line.rstrip()}")
        if budget.truncated:
            break

    append_truncation(results, budget)
    if not results:
        return f"未找到与 '{args['query']}' 相关的代码定义"
    return "\n".join(results)
