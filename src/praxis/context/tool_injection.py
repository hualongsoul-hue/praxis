"""动态工具集注入。

根据任务阶段从 S5 过滤工具 Schema，
支持工具分组和懒加载，按 LLM 原生 function calling 格式注入。
"""

from typing import Any

from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolRegistry

log = get_logger("context.tool_injection")

CORE_CATEGORIES = frozenset({"file_ops", "search", "shell", "system", "general", "autonomy"})
EXTENSION_CATEGORIES = frozenset({"mcp", "skill_script", "utility", "subagent"})

STAGE_TOOL_MAP: dict[str, set[str]] = {
    "general": CORE_CATEGORIES | EXTENSION_CATEGORIES,
    "planning": {"system", "general", "autonomy", "mcp", "subagent"},
    "coding": CORE_CATEGORIES | EXTENSION_CATEGORIES,
    "testing": CORE_CATEGORIES | EXTENSION_CATEGORIES,
    "debugging": {"file_ops", "search", "shell", "system", "general", "autonomy", "mcp", "subagent"},
    "review": {"file_ops", "search", "system", "general", "autonomy", "mcp", "subagent"},
}


class ToolInjector:
    """动态工具集注入器。

    根据当前任务阶段过滤工具，支持分组和懒加载扩展。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.loaded_groups: set[str] = set()
        self.extra_categories: set[str] = set()

    def get_tools_for_stage(
        self,
        stage: str = "general",
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """获取指定阶段的工具 Schema 列表。

        Args:
            stage: 任务阶段名称。
            tags: 额外按标签过滤。

        Returns:
            OpenAI function calling 格式的工具 Schema 列表。
        """
        allowed_categories = STAGE_TOOL_MAP.get(stage, CORE_CATEGORIES)
        categories = allowed_categories | self.extra_categories

        schemas: list[dict[str, Any]] = []
        for category in categories:
            batch = self.registry.get_tool_schemas(category=category, tags=tags)
            schemas.extend(batch)

        # 去重（同名工具可能出现在多个 category 中）
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for schema in schemas:
            name = schema.get("function", {}).get("name", "")
            if name and name not in seen:
                seen.add(name)
                unique.append(schema)

        self.loaded_groups.update(str(c) for c in categories)

        log.info(
            "工具集注入",
            stage=stage,
            tool_count=len(unique),
            categories=sorted(categories),
        )
        return unique

    def load_extension_group(self, category: str) -> list[dict[str, Any]]:
        """按需加载扩展工具组。

        Args:
            category: 工具类别名。

        Returns:
            新加载的工具 Schema 列表。
        """
        self.extra_categories.add(category)
        schemas = self.registry.get_tool_schemas(category=category)
        self.loaded_groups.add(category)
        log.info("扩展工具组已加载", category=category, count=len(schemas))
        return schemas

    def unload_extension_group(self, category: str) -> None:
        """卸载扩展工具组。"""
        self.extra_categories.discard(category)
        self.loaded_groups.discard(category)

    def get_loaded_groups(self) -> list[str]:
        """获取当前已加载的工具组列表。"""
        return sorted(self.loaded_groups)
