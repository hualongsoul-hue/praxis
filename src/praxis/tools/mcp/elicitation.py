"""MCP Elicitation（信息征询）。

结构化信息征询 + URL Mode，将 MCP Server 请求转发给用户。
"""

from collections.abc import Awaitable, Callable
from typing import Any

from praxis.models.mcp import MCPElicitationRequest, MCPElicitationResponse
from praxis.telemetry.logger import get_logger

log = get_logger("tools.mcp.elicitation")

ElicitationHandler = Callable[[MCPElicitationRequest], Awaitable[MCPElicitationResponse]]


class ElicitationManager:
    """Elicitation 管理器。

    将 MCP Server 的信息征询请求转发给用户界面层。
    """

    def __init__(self) -> None:
        self.handler: ElicitationHandler | None = None

    def set_handler(self, handler: ElicitationHandler) -> None:
        """注册用户界面层的 Elicitation 处理函数。

        Args:
            handler: 异步处理函数，接收请求返回响应。
        """
        self.handler = handler

    async def handle_elicitation(
        self, request: MCPElicitationRequest,
    ) -> MCPElicitationResponse:
        """处理 Elicitation 请求。

        Args:
            request: Elicitation 请求。

        Returns:
            用户响应。
        """
        if self.handler is None:
            log.warning("无 Elicitation 处理器，自动拒绝", server=request.server_name)
            return MCPElicitationResponse(accepted=False)

        log.info(
            "转发 Elicitation 请求",
            server=request.server_name,
            has_url=request.url is not None,
        )
        response = await self.handler(request)
        log.info(
            "Elicitation 响应",
            server=request.server_name,
            accepted=response.accepted,
        )
        return response
