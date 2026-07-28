"""MCP Sampling（LLM 采样）。

代理 LLM 调用 + Human-in-the-loop + 工具调用循环，
MCP Server 请求由 Praxis 通过 S4 模型网关代理完成。
"""

from collections.abc import Awaitable, Callable
from typing import Any, cast

from praxis.models.mcp import MCPSamplingRequest
from praxis.models.messages import Message, Role
from praxis.protocols import ModelGateway
from praxis.telemetry.logger import get_logger

log = get_logger("tools.mcp.sampling")

HumanReviewHandler = Callable[[list[dict[str, Any]], str], Awaitable[bool]]


class SamplingManager:
    """Sampling 管理器。

    将 MCP Server 的 LLM 调用请求通过 S4 模型网关代理完成。
    """

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway
        self.review_handler: HumanReviewHandler | None = None

    def set_review_handler(self, handler: HumanReviewHandler) -> None:
        """注册 Human-in-the-loop 审核函数。

        Args:
            handler: 异步审核函数，返回 True 表示批准。
        """
        self.review_handler = handler

    async def handle_sampling(
        self, request: MCPSamplingRequest,
    ) -> dict[str, Any]:
        """处理 Sampling 请求。

        Args:
            request: Sampling 请求。

        Returns:
            模型响应结果。
        """
        # 构造消息列表
        messages: list[Message] = []
        for msg in request.messages:
            role_value = cast(object, msg.get("role", "user"))
            role_str = role_value if isinstance(role_value, str) else "user"
            content_value = cast(object, msg.get("content", ""))
            if isinstance(content_value, list):
                texts: list[str] = []
                for raw_part in cast(list[object], content_value):
                    if not isinstance(raw_part, dict):
                        continue
                    part = cast(dict[str, Any], raw_part)
                    if part.get("type") == "text":
                        texts.append(str(part.get("text", "")))
                content = "\n".join(texts)
            else:
                content = str(content_value)
            messages.append(Message(role=Role(role_str), content=content))

        # Human-in-the-loop 审核
        if self.review_handler is not None:
            approved = await self.review_handler(request.messages, request.server_name)
            if not approved:
                log.info("Sampling 请求被用户拒绝", server=request.server_name)
                return {"role": "assistant", "content": "用户拒绝了此请求。"}

        # MCP Server 的偏好不能绕过 Runtime 的统一默认模型部署。
        model = self.gateway.config.default_model

        # 只依赖公共 ModelGateway 边界，支持 Runtime 注入自定义网关。
        msg_dicts = [{"role": m.role.value, "content": m.content} for m in messages]
        response = await self.gateway.complete(
            msg_dicts,
            model=model,
            max_tokens=request.max_tokens,
        )

        log.info(
            "Sampling 完成",
            server=request.server_name,
            model=model,
            tokens=response.usage.total_tokens if response.usage else 0,
        )

        return {
            "role": "assistant",
            "content": response.content,
            "model": model,
        }
