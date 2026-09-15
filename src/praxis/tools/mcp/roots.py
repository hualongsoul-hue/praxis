"""MCP Roots（工作目录声明）。

向 MCP Server 声明 Praxis 的工作目录范围。
"""

from mcp import ClientSession
from mcp.types import RootsListChangedNotification

from praxis.telemetry.logger import get_logger

log = get_logger("tools.mcp.roots")


class RootsManager:
    """Roots 管理器。

    管理工作目录声明，Server 变更时发送通知。
    """

    def __init__(self) -> None:
        self.roots: list[str] = []
        self.sessions: dict[str, ClientSession] = {}

    def register_session(self, server_name: str, session: ClientSession) -> None:
        """注册 MCP 服务器会话。"""
        self.sessions[server_name] = session

    def set_roots(self, roots: list[str]) -> None:
        """设置工作目录列表。

        Args:
            roots: 工作目录路径列表。
        """
        self.roots = list(roots)
        log.info("Roots 已设置", count=len(roots))

    def get_roots(self) -> list[str]:
        """获取当前工作目录列表。"""
        return list(self.roots)

    async def notify_roots_changed(self) -> None:
        """通知所有 MCP Server 工作目录已变更。"""
        for server_name, session in self.sessions.items():
            # Roots is part of the initialize-based protocol used by Praxis.
            await session.send_notification(RootsListChangedNotification())
            log.info("已通知 Roots 变更", server=server_name)

    def disconnect_server(self, server_name: str) -> None:
        """清理服务器会话。"""
        self.sessions.pop(server_name, None)
