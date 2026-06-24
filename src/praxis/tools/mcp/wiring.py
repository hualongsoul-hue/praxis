"""MCP 装配入口。

将 MCP Server 连接到会话：建立传输 → 能力协商 → 工具/资源/提示桥接，
把 MCP 工具注册进会话的统一工具注册表，供 LLM 透明调用。

传输（ClientSession）的生命周期由传入的 AsyncExitStack 持有，
随会话 terminate 时统一关闭。
"""

from contextlib import AsyncExitStack

from praxis.models.mcp import MCPServerConfig
from praxis.telemetry.logger import get_logger
from praxis.tools.mcp.connection import MCPConnectionManager
from praxis.tools.mcp.prompts import MCPPromptsBridge
from praxis.tools.mcp.resources import MCPResourcesBridge
from praxis.tools.mcp.roots import RootsManager
from praxis.tools.mcp.tools import MCPToolsBridge
from praxis.tools.mcp.transport import create_transport
from praxis.tools.registry import ToolRegistry

log = get_logger("tools.mcp.wiring")


async def connect_mcp_servers(
    registry: ToolRegistry,
    configs: list[MCPServerConfig],
    exit_stack: AsyncExitStack,
) -> MCPConnectionManager:
    """连接一组 MCP Server，并将其工具注册到 registry。

    单个服务器连接失败仅记录日志、不影响其它服务器与会话本身。

    Args:
        registry: 会话的统一工具注册表。
        configs: MCP 服务器配置列表。
        exit_stack: 持有各传输上下文的退出栈（由会话负责关闭）。

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
            session = await exit_stack.enter_async_context(create_transport(config))
            caps = await manager.connect_server(config, session)
            log.info(
                "MCP Server 已装配到会话",
                server=config.name,
                tools=caps.tools,
            )
        except Exception as exc:
            log.warning(
                "MCP Server 连接失败，已跳过",
                server=config.name,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    return manager
