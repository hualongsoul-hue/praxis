"""内置工具：grep_search。

在目录中递归搜索匹配正则表达式的文件行。
"""

import asyncio
from typing import Any, cast

from praxis.exceptions import ToolPolicyViolationError
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.builtins.search.common import (
    append_truncation,
    collect_search_files,
    create_search_budget,
    read_search_text,
    validate_regex,
)
from praxis.tools.policy import ToolPolicy

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

def create_handler(sandbox: ToolPolicy):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        return await asyncio.to_thread(run_grep_search, sandbox, args)

    return handle


def run_grep_search(sandbox: ToolPolicy, args: dict[str, Any]) -> str:
    """Run a bounded regex search in a worker thread."""
    search_path = sandbox.check_path(args["search_path"])
    pattern_text = str(args["pattern"])
    includes_value = args.get("includes")
    includes = (
        [str(value) for value in cast(list[object], includes_value)]
        if isinstance(includes_value, list)
        else []
    )
    if not search_path.exists():
        return f"路径不存在: {search_path}"
    try:
        pattern = validate_regex(pattern_text, sandbox)
    except ToolPolicyViolationError as exc:
        if str(exc).startswith("无效的正则表达式"):
            return str(exc)
        raise

    budget = create_search_budget(sandbox)
    files = collect_search_files(search_path, sandbox, budget)
    matches: list[str] = []
    for file_path in files:
        if includes and not any(file_path.match(glob) for glob in includes):
            continue
        text = read_search_text(file_path, budget)
        if text is None:
            break
        for line_num, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                if not budget.accept_match():
                    break
                matches.append(f"{file_path}:{line_num}: {line.rstrip()}")
        if budget.truncated:
            break

    append_truncation(matches, budget)
    if not matches:
        return f"未找到匹配 '{pattern_text}' 的结果"
    return "\n".join(matches)
