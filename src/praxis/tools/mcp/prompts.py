"""MCP Prompts 集成。

prompts/list 发现 + prompts/get 获取 + 参数补全（Completion）。
"""

from typing import Any

from mcp import ClientSession
from mcp.types import PromptReference, TextContent

from praxis.models.mcp import MCPPromptInfo, MCPPromptMessage
from praxis.telemetry.logger import get_logger
from praxis.tools.mcp.calls import invoke_mcp

log = get_logger("tools.mcp.prompts")


class MCPPromptsBridge:
    """MCP 提示模板桥接器。

    发现、获取和补全 MCP Server 提示模板。
    """

    def __init__(self) -> None:
        self.server_sessions: dict[str, ClientSession] = {}

    def register_session(self, server_name: str, session: ClientSession) -> None:
        """注册 MCP 服务器会话。"""
        self.server_sessions[server_name] = session

    async def list_prompts(self, server_name: str) -> list[MCPPromptInfo]:
        """列出 MCP Server 的提示模板。

        Args:
            server_name: 服务器名称。

        Returns:
            提示模板信息列表。
        """
        session = self.server_sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        result = await invoke_mcp(session.list_prompts())

        prompts: list[MCPPromptInfo] = []
        for p in result.prompts:
            args: list[dict[str, Any]] = []
            if p.arguments:
                for arg in p.arguments:
                    args.append({
                        "name": arg.name,
                        "description": arg.description or "",
                        "required": arg.required if hasattr(arg, "required") else False,
                    })
            prompts.append(MCPPromptInfo(
                name=p.name,
                description=p.description or "",
                arguments=args,
            ))

        log.info("MCP 提示已列出", server=server_name, count=len(prompts))
        return prompts

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: dict[str, str] | None = None,
    ) -> list[MCPPromptMessage]:
        """获取填充参数后的提示模板消息列表。

        Args:
            server_name: 服务器名称。
            prompt_name: 提示名称。
            arguments: 模板参数。

        Returns:
            提示消息列表。
        """
        session = self.server_sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        result = await invoke_mcp(session.get_prompt(prompt_name, arguments))

        messages: list[MCPPromptMessage] = []
        for msg in result.messages:
            # 提取文本内容
            content_text = (
                msg.content.text
                if isinstance(msg.content, TextContent)
                else ""
            )
            messages.append(MCPPromptMessage(
                role=msg.role,
                content=content_text,
            ))

        log.info(
            "MCP 提示已获取",
            server=server_name,
            prompt=prompt_name,
            message_count=len(messages),
        )
        return messages

    async def complete_argument(
        self,
        server_name: str,
        prompt_name: str,
        argument_name: str,
        partial_value: str,
    ) -> list[str]:
        """参数补全。

        Args:
            server_name: 服务器名称。
            prompt_name: 提示名称。
            argument_name: 参数名。
            partial_value: 部分输入值。

        Returns:
            补全建议列表。
        """
        session = self.server_sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        result = await session.complete(
            ref=PromptReference(type="ref/prompt", name=prompt_name),
            argument={"name": argument_name, "value": partial_value},
        )
        return list(result.completion.values) if result.completion else []

    def disconnect_server(self, server_name: str) -> None:
        """清理服务器会话。"""
        self.server_sessions.pop(server_name, None)
