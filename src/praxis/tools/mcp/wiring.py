"""Wire supervised MCP transports into one Session tool registry."""

import asyncio
from contextlib import AsyncExitStack
from functools import partial
from typing import Any, cast

from praxis.lifecycle import TaskSupervisor
from praxis.models.mcp import MCPElicitationRequest, MCPSamplingRequest, MCPServerConfig
from praxis.telemetry.logger import get_logger
from praxis.tools.mcp.access_tools import register_mcp_access_tools
from praxis.tools.mcp.auth import MCPAuthManager
from praxis.tools.mcp.connection import MCPConnectionManager, MCPTransportFactory
from praxis.tools.mcp.elicitation import ElicitationManager
from praxis.tools.mcp.prompts import MCPPromptsBridge
from praxis.tools.mcp.resources import MCPResourcesBridge
from praxis.tools.mcp.roots import RootsManager
from praxis.tools.mcp.sampling import SamplingManager
from praxis.tools.mcp.tools import MCPToolsBridge
from praxis.tools.mcp.transport import create_transport
from praxis.tools.registry import ToolRegistry

log = get_logger("tools.mcp.wiring")


def make_sampling_callback(server_name: str, manager: SamplingManager) -> Any:
    """Adapt the MCP SDK sampling callback to ``SamplingManager``."""
    from mcp.types import CreateMessageResult, ErrorData, TextContent

    async def callback(context: Any, params: Any) -> Any:
        messages: list[dict[str, Any]] = []
        for message in params.messages:
            content = getattr(message, "content", None)
            text = getattr(content, "text", "") if content is not None else ""
            messages.append({"role": getattr(message, "role", "user"), "content": text})
        preferences = cast(
            dict[str, Any],
            params.model_preferences.model_dump() if params.model_preferences else {},
        )
        request = MCPSamplingRequest(
            server_name=server_name,
            messages=messages,
            model_preferences=preferences,
            max_tokens=params.max_tokens or 1024,
        )
        try:
            result = await manager.handle_sampling(request)
        except Exception as exc:
            return ErrorData(code=-32603, message=str(exc))
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=result.get("content") or ""),
            model=result.get("model", "default"),
        )

    return callback


def make_elicitation_callback(server_name: str, manager: ElicitationManager) -> Any:
    """Adapt the MCP SDK elicitation callback to ``ElicitationManager``."""
    from mcp.types import ElicitResult, ErrorData

    async def callback(context: Any, params: Any) -> Any:
        schema = getattr(params, "requested_schema", None)
        if schema is not None and hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        request = MCPElicitationRequest(
            server_name=server_name,
            message=getattr(params, "message", "") or "",
            request_schema=cast(dict[str, Any], schema) if isinstance(schema, dict) else {},
            url=getattr(params, "url", None),
        )
        try:
            response = await manager.handle_elicitation(request)
        except Exception as exc:
            return ErrorData(code=-32603, message=str(exc))
        return ElicitResult(
            action="accept" if response.accepted else "decline",
            content=response.data or None,
        )

    return callback


async def authenticate_server_config(
    config: MCPServerConfig,
    auth_manager: MCPAuthManager | None,
) -> MCPServerConfig:
    """Return a new config containing ephemeral authentication headers."""
    if auth_manager is None:
        return config
    await auth_manager.initiate_auth_flow(config.name)
    headers = auth_manager.get_auth_headers(config.name)
    if not headers:
        return config
    return MCPServerConfig.model_validate({
        **config.model_dump(mode="python"),
        "headers": {**config.headers, **headers},
    })


async def connect_mcp_servers(
    registry: ToolRegistry,
    configs: list[MCPServerConfig],
    exit_stack: AsyncExitStack,
    sampling_manager: SamplingManager | None = None,
    elicitation_manager: ElicitationManager | None = None,
    auth_manager: MCPAuthManager | None = None,
    task_supervisor: TaskSupervisor | None = None,
) -> MCPConnectionManager:
    """Start independent reconnecting supervisors for all configured servers."""
    manager = MCPConnectionManager(
        tools_bridge=MCPToolsBridge(registry),
        resources_bridge=MCPResourcesBridge(),
        prompts_bridge=MCPPromptsBridge(),
        roots_mgr=RootsManager(),
        task_supervisor=task_supervisor,
    )
    exit_stack.push_async_callback(manager.close)
    initial_results: list[asyncio.Future[None]] = []

    for original_config in configs:
        try:
            config = await authenticate_server_config(original_config, auth_manager)
        except Exception as exc:
            log.warning(
                "MCP Server 认证失败，已跳过",
                server=original_config.name,
                error_type=type(exc).__name__,
            )
            continue

        sampling_callback = (
            make_sampling_callback(config.name, sampling_manager)
            if sampling_manager is not None
            else None
        )
        elicitation_callback = (
            make_elicitation_callback(config.name, elicitation_manager)
            if elicitation_manager is not None
            else None
        )
        transport_factory = cast(
            MCPTransportFactory,
            partial(
                create_transport,
                config,
                sampling_callback,
                elicitation_callback,
            ),
        )

        def refresh_access_tools() -> None:
            register_mcp_access_tools(registry, manager)

        initial_results.append(
            manager.start_server(config, transport_factory, refresh_access_tools)
        )

    try:
        if initial_results:
            await asyncio.gather(*initial_results)
    except BaseException:
        await manager.close()
        raise

    register_mcp_access_tools(registry, manager)
    return manager
