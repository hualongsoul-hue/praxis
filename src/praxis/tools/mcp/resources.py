"""MCP Resources 集成。

resources/list 发现 + resources/read 读取 + 资源模板 + resources/subscribe 订阅变更。
"""

from typing import Any

from mcp import ClientSession
from pydantic import AnyUrl

from praxis.models.mcp import MCPResourceContent, MCPResourceInfo
from praxis.telemetry.logger import get_logger

log = get_logger("tools.mcp.resources")


class MCPResourcesBridge:
    """MCP 资源桥接器。

    发现、读取、订阅 MCP Server 资源。
    """

    def __init__(self) -> None:
        self.server_sessions: dict[str, ClientSession] = {}
        self.subscriptions: dict[str, set[str]] = {}

    def register_session(self, server_name: str, session: ClientSession) -> None:
        """注册 MCP 服务器会话。"""
        self.server_sessions[server_name] = session

    async def list_resources(self, server_name: str) -> list[MCPResourceInfo]:
        """列出 MCP Server 提供的资源。

        Args:
            server_name: 服务器名称。

        Returns:
            资源信息列表。
        """
        session = self.server_sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        result = await session.list_resources()

        resources: list[MCPResourceInfo] = []
        for r in result.resources:
            resources.append(MCPResourceInfo(
                uri=str(r.uri),
                name=r.name or "",
                description=r.description or "",
                mime_type=r.mimeType or "text/plain",
            ))

        log.info("MCP 资源已列出", server=server_name, count=len(resources))
        return resources

    async def list_resource_templates(
        self, server_name: str,
    ) -> list[dict[str, Any]]:
        """列出 MCP Server 的资源模板。

        Args:
            server_name: 服务器名称。

        Returns:
            资源模板列表。
        """
        session = self.server_sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        result = await session.list_resource_templates()
        templates: list[dict[str, Any]] = []
        for t in result.resourceTemplates:
            templates.append({
                "uri_template": str(t.uriTemplate),
                "name": t.name or "",
                "description": t.description or "",
                "mime_type": t.mimeType or "text/plain",
            })
        return templates

    async def read_resource(
        self, server_name: str, uri: str,
    ) -> MCPResourceContent:
        """读取 MCP 资源内容。

        Args:
            server_name: 服务器名称。
            uri: 资源 URI。

        Returns:
            资源内容。
        """
        session = self.server_sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        result = await session.read_resource(AnyUrl(uri))

        # 提取第一个内容块
        if result.contents:
            content = result.contents[0]
            text = getattr(content, "text", None)
            blob = getattr(content, "blob", None)
            mime_type = getattr(content, "mimeType", "text/plain") or "text/plain"
            return MCPResourceContent(
                uri=uri,
                mime_type=mime_type,
                text=text,
                blob=blob,
            )

        return MCPResourceContent(uri=uri)

    async def subscribe_resource(self, server_name: str, uri: str) -> None:
        """订阅资源变更通知。

        Args:
            server_name: 服务器名称。
            uri: 资源 URI。
        """
        session = self.server_sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        await session.subscribe_resource(AnyUrl(uri))
        self.subscriptions.setdefault(server_name, set()).add(uri)
        log.info("已订阅资源", server=server_name, uri=uri)

    async def unsubscribe_resource(self, server_name: str, uri: str) -> None:
        """取消订阅资源。"""
        session = self.server_sessions.get(server_name)
        if session is None:
            return

        await session.unsubscribe_resource(AnyUrl(uri))
        subs = self.subscriptions.get(server_name)
        if subs:
            subs.discard(uri)

    def disconnect_server(self, server_name: str) -> None:
        """清理服务器会话和订阅。"""
        self.server_sessions.pop(server_name, None)
        self.subscriptions.pop(server_name, None)
