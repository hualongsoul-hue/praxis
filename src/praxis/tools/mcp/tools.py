"""MCP Tools 集成。

tools/list 发现工具 + tools/call 执行 + 动态更新，
MCP 工具与内置工具在注册表中统一管理。
"""

from typing import Any

from mcp import ClientSession
from mcp.types import TextContent

from praxis.models.mcp import MCPToolInfo
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.telemetry.logger import get_logger
from praxis.tools.mcp.calls import invoke_mcp
from praxis.tools.registry import ToolRegistry

log = get_logger("tools.mcp.tools")


class MCPToolsBridge:
    """MCP 工具桥接器。

    发现 MCP Server 工具并注册到统一注册表，
    执行时透明代理到 MCP Server。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.server_sessions: dict[str, ClientSession] = {}
        self.server_tools: dict[str, list[str]] = {}

    async def discover_tools(
        self,
        server_name: str,
        session: ClientSession,
    ) -> list[MCPToolInfo]:
        """发现并注册 MCP Server 的工具。

        Args:
            server_name: 服务器名称。
            session: MCP 客户端会话。

        Returns:
            发现的工具信息列表。
        """
        self.server_sessions[server_name] = session
        result = await session.list_tools()

        tools: list[MCPToolInfo] = []
        tool_names: list[str] = []

        for tool in result.tools:
            mcp_name = f"mcp_{server_name}_{tool.name}"
            info = MCPToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.input_schema if tool.input_schema else {},
                server_name=server_name,
            )
            tools.append(info)
            tool_names.append(mcp_name)

            # 注册到统一注册表
            definition = ToolDefinition(
                name=mcp_name,
                description=f"[MCP:{server_name}] {tool.description or tool.name}",
                parameters=tool.input_schema or {"type": "object", "properties": {}},
                metadata=ToolMetadata(
                    category="mcp",
                    readonly=False,
                    tags=[f"mcp:{server_name}"],
                ),
            )

            # 创建代理处理函数
            bound_tool_name = tool.name
            bound_server_name = server_name

            async def handler(
                arguments: dict[str, Any] | None = None,
                tool_name: str = bound_tool_name,
                server_name: str = bound_server_name,
                **kwargs: Any,
            ) -> str:
                return await self.call_tool(server_name, tool_name, arguments or {})

            self.registry.register(definition, handler)

        self.server_tools[server_name] = tool_names
        log.info(
            "MCP 工具已发现并注册",
            server=server_name,
            tool_count=len(tools),
        )
        return tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """调用 MCP 工具。

        Args:
            server_name: 服务器名称。
            tool_name: 原始工具名称。
            arguments: 调用参数。

        Returns:
            工具执行结果文本。
        """
        session = self.server_sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        result = await invoke_mcp(session.call_tool(tool_name, arguments))
        if result.is_error:
            # Error payloads may contain server credentials or private diagnostics.
            # Raise a non-transient failure; never retry an ambiguous remote action.
            raise RuntimeError("MCP server reported tool failure")

        # 提取文本内容
        texts: list[str] = []
        for content in result.content:
            if isinstance(content, TextContent):
                texts.append(content.text)
            else:
                texts.append(f"[{content.type} content]")
        return "\n".join(texts) if texts else ""

    async def refresh_tools(self, server_name: str) -> list[MCPToolInfo]:
        """刷新 MCP Server 的工具列表（响应 list_changed 通知）。

        Args:
            server_name: 服务器名称。

        Returns:
            更新后的工具列表。
        """
        # 先注销旧工具
        self.unregister_server_tools(server_name)

        session = self.server_sessions.get(server_name)
        if session is None:
            return []
        return await self.discover_tools(server_name, session)

    def unregister_server_tools(self, server_name: str) -> None:
        """注销指定服务器的所有工具。"""
        tool_names = self.server_tools.pop(server_name, [])
        for name in tool_names:
            if self.registry.has_tool(name):
                self.registry.unregister(name)
        log.info("MCP 工具已注销", server=server_name, count=len(tool_names))

    def disconnect_server(self, server_name: str) -> None:
        """断开服务器连接并清理。"""
        self.unregister_server_tools(server_name)
        self.server_sessions.pop(server_name, None)
