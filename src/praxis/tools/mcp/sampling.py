"""MCP Sampling（LLM 采样）。

代理 LLM 调用 + Human-in-the-loop + 工具调用循环，
MCP Server 请求由 Praxis 通过 S4 模型网关代理完成。
"""

from collections.abc import Awaitable, Callable
from typing import Any

from praxis.gateway.chat import chat
from praxis.gateway.router import GatewayRouter
from praxis.models.mcp import MCPSamplingRequest
from praxis.models.messages import Message, Role
from praxis.telemetry.logger import get_logger

log = get_logger("tools.mcp.sampling")

HumanReviewHandler = Callable[[list[dict[str, Any]], str], Awaitable[bool]]


class SamplingManager:
    """Sampling 管理器。

    将 MCP Server 的 LLM 调用请求通过 S4 模型网关代理完成。
    """

    def __init__(self, router: GatewayRouter) -> None:
        self.router = router
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
            role_str = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = "\n".join(texts)
            messages.append(Message(role=Role(role_str), content=content))

        # Human-in-the-loop 审核
        if self.review_handler is not None:
            approved = await self.review_handler(request.messages, request.server_name)
            if not approved:
                log.info("Sampling 请求被用户拒绝", server=request.server_name)
                return {"role": "assistant", "content": "用户拒绝了此请求。"}

        # 解析模型偏好
        model = self.resolve_model_preference(request.model_preferences)

        # 通过 S4 调用 LLM
        msg_dicts = [{"role": m.role.value, "content": m.content} for m in messages]
        response = await chat(
            gateway=self.router,
            messages=msg_dicts,
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

    @staticmethod
    def resolve_model_preference(preferences: dict[str, Any]) -> str:
        """根据偏好选择模型。

        Args:
            preferences: 模型偏好（intelligence/speed/cost）。

        Returns:
            模型名称。
        """
        if not preferences:
            return "default"

        hints = preferences.get("hints", [])
        if hints:
            for hint in hints:
                if isinstance(hint, dict) and hint.get("name"):
                    return hint["name"]

        # 按优先级排序
        priority = preferences.get("intelligencePriority", 0)
        if priority > 0.7:
            return "default"

        speed = preferences.get("speedPriority", 0)
        if speed > 0.7:
            return "default"

        return "default"
