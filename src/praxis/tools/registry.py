"""工具注册表。

所有工具（内置 + MCP + 自定义）通过注册表统一管理。
提供注册/注销/查询/Schema 导出接口。
"""

from collections.abc import Awaitable, Callable
from typing import Any

from praxis.exceptions import ToolNotFoundError
from praxis.models.tools import ToolDefinition, ToolMetadata

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]
"""工具处理函数签名：接收验证后的参数字典，返回结果字符串。"""


class ToolEntry:
    """注册表中的工具条目，关联定义与处理函数。"""

    __slots__ = ("definition", "handler", "overridden_from")

    def __init__(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
        overridden_from: str | None = None,
    ) -> None:
        self.definition = definition
        self.handler = handler
        self.overridden_from = overridden_from


class ToolRegistry:
    """线程安全的工具注册表。"""

    def __init__(self) -> None:
        self.tools: dict[str, ToolEntry] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:
        """注册工具。

        如果同名工具已存在，后注册的覆盖先注册的（覆盖机制见 override.py）。

        Args:
            definition: 完整工具定义。
            handler: 异步处理函数。
        """
        overridden_from: str | None = None
        if definition.name in self.tools:
            overridden_from = definition.name
        self.tools[definition.name] = ToolEntry(
            definition=definition,
            handler=handler,
            overridden_from=overridden_from,
        )

    def unregister(self, name: str) -> bool:
        """注销工具。返回是否成功注销。"""
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def get_entry(self, name: str) -> ToolEntry:
        """获取工具条目。

        Raises:
            ToolNotFoundError: 工具未注册。
        """
        entry = self.tools.get(name)
        if entry is None:
            raise ToolNotFoundError(
                f"工具 '{name}' 未注册",
                details={"available": list(self.tools.keys())},
            )
        return entry

    def get_definition(self, name: str) -> ToolDefinition:
        """获取工具定义。"""
        return self.get_entry(name).definition

    def get_metadata(self, name: str) -> ToolMetadata:
        """获取工具元数据，供 S8 护栏裁决使用。"""
        return self.get_entry(name).definition.metadata

    def get_tool_schemas(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """导出工具 Schema 列表（LLM 可注入的 OpenAI function 格式）。

        Args:
            category: 按类别过滤。
            tags: 按标签过滤（任一匹配即可）。

        Returns:
            OpenAI function calling 格式的工具 Schema 列表。
        """
        results: list[dict[str, Any]] = []
        for entry in self.tools.values():
            defn = entry.definition
            meta = defn.metadata

            if category and meta.category != category:
                continue
            if tags and not set(tags) & set(meta.tags):
                continue

            results.append({
                "type": "function",
                "function": {
                    "name": defn.name,
                    "description": defn.description,
                    "parameters": defn.parameters,
                },
            })
        return results

    def list_tools(self) -> list[str]:
        """返回所有已注册工具名列表。"""
        return list(self.tools.keys())

    def has_tool(self, name: str) -> bool:
        """检查工具是否已注册。"""
        return name in self.tools
