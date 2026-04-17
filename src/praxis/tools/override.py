"""工具覆盖机制。

允许同名自定义工具覆盖内置工具，覆盖后必须保持兼容的 Schema 接口契约。
"""

from praxis.exceptions import ToolError
from praxis.models.tools import ToolDefinition
from praxis.tools.registry import ToolHandler, ToolRegistry


def override_tool(
    registry: ToolRegistry,
    definition: ToolDefinition,
    handler: ToolHandler,
) -> None:
    """覆盖已注册的同名工具。

    覆盖要求：
    - 目标工具必须已存在于注册表中。
    - 新 Schema 的所有必填参数必须在原 Schema 中存在（向后兼容）。

    Args:
        registry: 工具注册表。
        definition: 新的工具定义。
        handler: 新的处理函数。

    Raises:
        ToolError: 目标工具不存在或 Schema 不兼容。
    """
    if not registry.has_tool(definition.name):
        raise ToolError(
            f"无法覆盖不存在的工具: '{definition.name}'",
            details={"available": registry.list_tools()},
        )

    original = registry.get_definition(definition.name)
    check_schema_compatibility(original.parameters, definition.parameters)
    registry.register(definition, handler)


def check_schema_compatibility(
    original: dict,
    override: dict,
) -> None:
    """检查覆盖 Schema 与原始 Schema 的兼容性。

    规则：原始 Schema 中的所有 required 参数必须在覆盖 Schema 的 properties 中存在。

    Raises:
        ToolError: Schema 不兼容。
    """
    original_required = set(original.get("required", []))
    override_props = set(override.get("properties", {}).keys())

    missing = original_required - override_props
    if missing:
        raise ToolError(
            f"覆盖 Schema 缺少原始必填参数: {missing}",
            details={
                "missing_params": list(missing),
                "original_required": list(original_required),
                "override_properties": list(override_props),
            },
        )
