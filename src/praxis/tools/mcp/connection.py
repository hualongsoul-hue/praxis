"""MCP 服务器连接管理。

初始化阶段能力协商、Bridges 注册、断开清理。
"""

from mcp import ClientSession

from praxis.models.mcp import (
    MCPServerCapabilities,
    MCPServerConfig,
    MCPServerStatus,
)
from praxis.telemetry.logger import get_logger
from praxis.tools.mcp.prompts import MCPPromptsBridge
from praxis.tools.mcp.resources import MCPResourcesBridge
from praxis.tools.mcp.roots import RootsManager
from praxis.tools.mcp.tools import MCPToolsBridge

log = get_logger("tools.mcp.connection")


class MCPServerConnection:
    """单个 MCP Server 的连接状态和元信息。"""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.status: MCPServerStatus = MCPServerStatus.DISCONNECTED
        self.capabilities: MCPServerCapabilities = MCPServerCapabilities()
        self.reconnect_count: int = 0


class MCPConnectionManager:
    """MCP 服务器连接管理器。

    管理多个 MCP Server 的连接、能力协商、崩溃重连。
    """

    def __init__(
        self,
        tools_bridge: MCPToolsBridge,
        resources_bridge: MCPResourcesBridge,
        prompts_bridge: MCPPromptsBridge,
        roots_mgr: RootsManager,
    ) -> None:
        self.tools_bridge = tools_bridge
        self.resources_bridge = resources_bridge
        self.prompts_bridge = prompts_bridge
        self.roots_mgr = roots_mgr
        self.connections: dict[str, MCPServerConnection] = {}

    async def connect_server(
        self,
        config: MCPServerConfig,
        session: ClientSession,
    ) -> MCPServerCapabilities:
        """连接并协商 MCP Server 能力。

        Args:
            config: 服务器配置。
            session: 已建立的 ClientSession。

        Returns:
            服务器能力集。
        """
        conn = MCPServerConnection(config)
        conn.status = MCPServerStatus.CONNECTING

        # 获取能力
        caps = session.get_server_capabilities()
        conn.capabilities = MCPServerCapabilities(
            tools=caps.tools is not None if caps else False,
            resources=caps.resources is not None if caps else False,
            prompts=caps.prompts is not None if caps else False,
            sampling=False,
        )
        conn.status = MCPServerStatus.CONNECTED
        self.connections[config.name] = conn

        # 注册各桥接器
        if conn.capabilities.tools:
            await self.tools_bridge.discover_tools(config.name, session)
        if conn.capabilities.resources:
            self.resources_bridge.register_session(config.name, session)
        if conn.capabilities.prompts:
            self.prompts_bridge.register_session(config.name, session)
        self.roots_mgr.register_session(config.name, session)

        log.info(
            "MCP Server 已连接",
            server=config.name,
            capabilities=conn.capabilities.model_dump(),
        )
        return conn.capabilities

    def disconnect_server(self, server_name: str) -> None:
        """断开并清理 MCP Server。"""
        self.tools_bridge.disconnect_server(server_name)
        self.resources_bridge.disconnect_server(server_name)
        self.prompts_bridge.disconnect_server(server_name)
        self.roots_mgr.disconnect_server(server_name)

        conn = self.connections.pop(server_name, None)
        if conn:
            conn.status = MCPServerStatus.DISCONNECTED
        log.info("MCP Server 已断开", server=server_name)

    def get_server_status(self, server_name: str) -> MCPServerStatus:
        """获取服务器连接状态。"""
        conn = self.connections.get(server_name)
        if conn is None:
            return MCPServerStatus.DISCONNECTED
        return conn.status

    def list_connected_servers(self) -> list[str]:
        """列出所有已连接的服务器。"""
        return [
            name for name, conn in self.connections.items()
            if conn.status == MCPServerStatus.CONNECTED
        ]

    def get_server_capabilities(
        self, server_name: str,
    ) -> MCPServerCapabilities | None:
        """获取服务器能力。"""
        conn = self.connections.get(server_name)
        if conn is None:
            return None
        return conn.capabilities
