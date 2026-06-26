"""MCP 资源/提示访问工具。

将 MCP Server 的 resources（资源读取）与 prompts（提示模板）能力暴露为
LLM 可调用工具，使 Agent 能够发现并读取 MCP 资源、列出并取用提示模板。

仅在连接的 Server 具备相应能力时注册（见 register_mcp_access_tools）。
"""

import json
from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.telemetry.logger import get_logger
from praxis.tools.mcp.connection import MCPConnectionManager
from praxis.tools.registry import ToolHandler, ToolRegistry

log = get_logger("tools.mcp.access_tools")


def server_enum(servers: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": "MCP 服务器名称"}
    if servers:
        schema["enum"] = servers
    return schema


def mcp_metadata(readonly: bool, tag: str) -> ToolMetadata:
    return ToolMetadata(
        category="mcp",
        permission_level="auto_approve",
        readonly=readonly,
        tags=["mcp", tag],
    )


def make_list_resources(bridge: Any) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        resources = await bridge.list_resources(args["server_name"])
        return json.dumps([r.model_dump() for r in resources], ensure_ascii=False)

    return handle


def make_read_resource(bridge: Any) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        content = await bridge.read_resource(args["server_name"], args["uri"])
        return json.dumps(content.model_dump(), ensure_ascii=False, default=str)

    return handle


def make_list_prompts(bridge: Any) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        prompts = await bridge.list_prompts(args["server_name"])
        return json.dumps([p.model_dump() for p in prompts], ensure_ascii=False)

    return handle


def make_get_prompt(bridge: Any) -> ToolHandler:
    async def handle(args: dict[str, Any]) -> str:
        messages = await bridge.get_prompt(
            args["server_name"],
            args["prompt_name"],
            args.get("arguments") or None,
        )
        return json.dumps([m.model_dump() for m in messages], ensure_ascii=False)

    return handle


def register_mcp_access_tools(
    registry: ToolRegistry,
    manager: MCPConnectionManager,
) -> list[str]:
    """按连接的 Server 能力注册 MCP 资源/提示访问工具。

    Args:
        registry: 会话的统一工具注册表。
        manager: 已装配的 MCP 连接管理器。

    Returns:
        已注册的工具名列表。
    """
    connected = manager.list_connected_servers()
    resource_servers = [
        s for s in connected
        if (caps := manager.get_server_capabilities(s)) is not None and caps.resources
    ]
    prompt_servers = [
        s for s in connected
        if (caps := manager.get_server_capabilities(s)) is not None and caps.prompts
    ]

    registered: list[str] = []

    if resource_servers:
        registry.register(
            ToolDefinition(
                name="mcp_list_resources",
                description=f"列出指定 MCP 服务器提供的资源。可用服务器: {resource_servers}",
                parameters={
                    "type": "object",
                    "properties": {"server_name": server_enum(resource_servers)},
                    "required": ["server_name"],
                },
                metadata=mcp_metadata(readonly=True, tag="resources"),
            ),
            make_list_resources(manager.resources_bridge),
        )
        registry.register(
            ToolDefinition(
                name="mcp_read_resource",
                description=f"读取指定 MCP 服务器的资源内容（按 URI）。可用服务器: {resource_servers}",
                parameters={
                    "type": "object",
                    "properties": {
                        "server_name": server_enum(resource_servers),
                        "uri": {"type": "string", "description": "资源 URI"},
                    },
                    "required": ["server_name", "uri"],
                },
                metadata=mcp_metadata(readonly=True, tag="resources"),
            ),
            make_read_resource(manager.resources_bridge),
        )
        registered += ["mcp_list_resources", "mcp_read_resource"]

    if prompt_servers:
        registry.register(
            ToolDefinition(
                name="mcp_list_prompts",
                description=f"列出指定 MCP 服务器的提示模板。可用服务器: {prompt_servers}",
                parameters={
                    "type": "object",
                    "properties": {"server_name": server_enum(prompt_servers)},
                    "required": ["server_name"],
                },
                metadata=mcp_metadata(readonly=True, tag="prompts"),
            ),
            make_list_prompts(manager.prompts_bridge),
        )
        registry.register(
            ToolDefinition(
                name="mcp_get_prompt",
                description=f"获取并填充指定 MCP 服务器的提示模板，返回消息列表。可用服务器: {prompt_servers}",
                parameters={
                    "type": "object",
                    "properties": {
                        "server_name": server_enum(prompt_servers),
                        "prompt_name": {"type": "string", "description": "提示模板名称"},
                        "arguments": {
                            "type": "object",
                            "description": "模板参数（键值对）",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["server_name", "prompt_name"],
                },
                metadata=mcp_metadata(readonly=True, tag="prompts"),
            ),
            make_get_prompt(manager.prompts_bridge),
        )
        registered += ["mcp_list_prompts", "mcp_get_prompt"]

    if registered:
        log.info("MCP 资源/提示访问工具已注册", tools=registered)
    return registered
