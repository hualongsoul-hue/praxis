"""MCP 装配入口。

将 MCP Server 连接到会话：建立传输 → 能力协商 → 工具/资源/提示桥接，
把 MCP 工具注册进会话的统一工具注册表，供 LLM 透明调用。

传输（ClientSession）的生命周期由传入的 AsyncExitStack 持有，
随会话 terminate 时统一关闭。
"""

from contextlib import AsyncExitStack
from typing import Any

from praxis.models.mcp import (
    MCPElicitationRequest,
    MCPSamplingRequest,
    MCPServerConfig,
)
from praxis.telemetry.logger import get_logger
from praxis.tools.mcp.access_tools import register_mcp_access_tools
from praxis.tools.mcp.connection import MCPConnectionManager
from praxis.tools.mcp.elicitation import ElicitationManager
from praxis.tools.mcp.prompts import MCPPromptsBridge
from praxis.tools.mcp.resources import MCPResourcesBridge
from praxis.tools.mcp.roots import RootsManager
from praxis.tools.mcp.sampling import SamplingManager
from praxis.tools.mcp.tools import MCPToolsBridge
from praxis.tools.mcp.transport import create_transport
from praxis.tools.registry import ToolRegistry

log = get_logger("tools.mcp.wiring")


def _make_sampling_callback(server_name: str, manager: SamplingManager) -> Any:
    """适配 MCP SDK sampling_callback → Praxis SamplingManager。"""
    from mcp.types import CreateMessageResult, ErrorData, TextContent

    async def callback(context: Any, params: Any) -> Any:
        messages: list[dict[str, Any]] = []
        for msg in params.messages:
            content = getattr(msg, "content", None)
            text = getattr(content, "text", "") if content is not None else ""
            messages.append({"role": getattr(msg, "role", "user"), "content": text})
        prefs = params.modelPreferences.model_dump() if params.modelPreferences else {}
        request = MCPSamplingRequest(
            server_name=server_name,
            messages=messages,
            model_preferences=prefs,
            max_tokens=params.maxTokens or 1024,
        )
        try:
            result = await manager.handle_sampling(request)
        except Exception as exc:  # noqa: BLE001 - 转为 MCP 协议错误
            return ErrorData(code=-32603, message=str(exc))
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=result.get("content") or ""),
            model=result.get("model", "default"),
        )

    return callback


def _make_elicitation_callback(server_name: str, manager: ElicitationManager) -> Any:
    """适配 MCP SDK elicitation_callback → Praxis ElicitationManager。"""
    from mcp.types import ElicitResult, ErrorData

    async def callback(context: Any, params: Any) -> Any:
        schema = getattr(params, "requestedSchema", None)
        if schema is not None and hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        request = MCPElicitationRequest(
            server_name=server_name,
            message=getattr(params, "message", "") or "",
            request_schema=schema if isinstance(schema, dict) else {},
            url=getattr(params, "url", None),
        )
        try:
            response = await manager.handle_elicitation(request)
        except Exception as exc:  # noqa: BLE001 - 转为 MCP 协议错误
            return ErrorData(code=-32603, message=str(exc))
        return ElicitResult(
            action="accept" if response.accepted else "decline",
            content=response.data or None,
        )

    return callback


async def connect_mcp_servers(
    registry: ToolRegistry,
    configs: list[MCPServerConfig],
    exit_stack: AsyncExitStack,
    sampling_manager: SamplingManager | None = None,
    elicitation_manager: ElicitationManager | None = None,
) -> MCPConnectionManager:
    """连接一组 MCP Server，并将其工具注册到 registry。

    单个服务器连接失败仅记录日志、不影响其它服务器与会话本身。
    传入 sampling/elicitation 管理器时，将其作为 MCP 客户端回调注册，
    使 Server → 客户端的采样/征询请求被路由到 Praxis（S4 网关 / 用户界面）。

    Args:
        registry: 会话的统一工具注册表。
        configs: MCP 服务器配置列表。
        exit_stack: 持有各传输上下文的退出栈（由会话负责关闭）。
        sampling_manager: 可选 Sampling 管理器（经 S4 代理 LLM 采样）。
        elicitation_manager: 可选 Elicitation 管理器（转发用户征询）。

    Returns:
        已装配的 MCPConnectionManager（含连接状态）。
    """
    tools_bridge = MCPToolsBridge(registry)
    manager = MCPConnectionManager(
        tools_bridge=tools_bridge,
        resources_bridge=MCPResourcesBridge(),
        prompts_bridge=MCPPromptsBridge(),
        roots_mgr=RootsManager(),
    )

    for config in configs:
        try:
            sampling_cb = (
                _make_sampling_callback(config.name, sampling_manager)
                if sampling_manager is not None else None
            )
            elicitation_cb = (
                _make_elicitation_callback(config.name, elicitation_manager)
                if elicitation_manager is not None else None
            )
            session = await exit_stack.enter_async_context(
                create_transport(config, sampling_cb, elicitation_cb)
            )
            caps = await manager.connect_server(config, session)
            log.info(
                "MCP Server 已装配到会话",
                server=config.name,
                tools=caps.tools,
                sampling=sampling_manager is not None,
                elicitation=elicitation_manager is not None,
            )
        except Exception as exc:
            log.warning(
                "MCP Server 连接失败，已跳过",
                server=config.name,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    # 将具备能力的 Server 的资源/提示暴露为 LLM 可调用工具
    register_mcp_access_tools(registry, manager)

    return manager
