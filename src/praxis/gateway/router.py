"""类型化 LiteLLM 模型网关适配器。"""

import asyncio
import os
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from litellm.router import Router

from praxis.config.schemas import GatewayConfig, ModelCapabilities, ModelDeployment
from praxis.exceptions import AuthenticationError, BudgetExceededError, GatewayError
from praxis.models.responses import ModelResponse, ModelResponseChunk


@dataclass(frozen=True, slots=True)
class UsageReservation:
    identifier: int
    estimated_tokens: int
    estimated_cost: float


class GatewayRouter:
    """Runtime 实例拥有的 LiteLLM Router、预算计数与并发闸门。"""

    def __init__(
        self,
        config: GatewayConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.gateway_config = config
        self.environ = os.environ if environ is None else environ
        self.deployments = list(config.deployments)
        self.litellm_router = self.build_litellm_router(config, self.deployments, self.environ)
        self.total_spend_usd = 0.0
        self.total_token_count = 0
        self.reserved_spend_usd = 0.0
        self.reserved_tokens = 0
        self.reservations: dict[int, UsageReservation] = {}
        self.next_reservation_id = 1
        self.budget_lock = threading.Lock()
        self.request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.health_probe_lock = asyncio.Lock()
        self.health_probe_value: bool | None = None
        self.health_probe_expires_at = 0.0
        self.last_health_error = ""

    @staticmethod
    def to_litellm_deployment(
        deployment: ModelDeployment,
        environ: Mapping[str, str],
    ) -> dict[str, Any]:
        api_key = environ.get(deployment.api_key_env)
        if not api_key:
            raise AuthenticationError(
                f"缺少模型凭据环境变量: {deployment.api_key_env}",
                details={"environment_variable": deployment.api_key_env},
            )
        return {
            "model_name": deployment.model_name,
            "litellm_params": {
                "model": deployment.model,
                "api_base": deployment.api_base,
                "api_key": api_key,
            },
        }

    @classmethod
    def build_litellm_router(
        cls,
        config: GatewayConfig,
        deployments: list[ModelDeployment],
        environ: Mapping[str, str],
    ) -> Router:
        model_list = [cls.to_litellm_deployment(item, environ) for item in deployments]
        return Router(
            model_list=model_list,
            routing_strategy=config.routing_strategy,
            num_retries=config.num_retries,
            timeout=config.timeout,
        )

    @property
    def router(self) -> Router:
        return self.litellm_router

    @property
    def config(self) -> GatewayConfig:
        return self.gateway_config

    @property
    def total_spend(self) -> float:
        with self.budget_lock:
            return self.total_spend_usd

    @property
    def total_tokens(self) -> int:
        with self.budget_lock:
            return self.total_token_count

    def add_usage(self, *, cost: float, tokens: int) -> None:
        """线程安全地累加模型用量。"""
        with self.budget_lock:
            if cost > 0:
                self.total_spend_usd += cost
            if tokens > 0:
                self.total_token_count += tokens

    def add_spend(self, cost: float) -> None:
        """兼容内部计量调用；新代码应使用 add_usage。"""
        self.add_usage(cost=cost, tokens=0)

    def resolve_deployment(self, model_name: str) -> ModelDeployment:
        for deployment in self.deployments:
            if deployment.model_name == model_name:
                return deployment
        raise GatewayError("模型别名未配置", details={"model_name": model_name})

    def capabilities(self, model_name: str | None = None) -> ModelCapabilities:
        """Return the typed capabilities declared for a configured model alias."""
        resolved_name = model_name or self.gateway_config.default_model
        return self.resolve_deployment(resolved_name).capabilities

    def reserve_usage(
        self,
        *,
        estimated_tokens: int,
        estimated_cost: float | None,
    ) -> UsageReservation:
        """在线程锁内原子校验并预留本次调用的 Token 与金额预算。"""
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens 不能为负数")
        with self.budget_lock:
            if self.gateway_config.max_total_tokens is not None:
                projected_tokens = (
                    self.total_token_count + self.reserved_tokens + estimated_tokens
                )
                if projected_tokens > self.gateway_config.max_total_tokens:
                    raise BudgetExceededError(
                        "预计累计 Token 超出上限",
                        details={
                            "projected_tokens": projected_tokens,
                            "max_total_tokens": self.gateway_config.max_total_tokens,
                        },
                    )
            if self.gateway_config.max_budget is not None:
                if estimated_cost is None:
                    raise BudgetExceededError("模型价格未知，启用金额预算时拒绝调用")
                projected_cost = (
                    self.total_spend_usd + self.reserved_spend_usd + estimated_cost
                )
                if projected_cost > self.gateway_config.max_budget:
                    raise BudgetExceededError(
                        "预计累计成本超出金额预算",
                        details={
                            "projected_cost": projected_cost,
                            "max_budget": self.gateway_config.max_budget,
                        },
                    )

            reservation = UsageReservation(
                identifier=self.next_reservation_id,
                estimated_tokens=estimated_tokens,
                estimated_cost=estimated_cost or 0.0,
            )
            self.next_reservation_id += 1
            self.reservations[reservation.identifier] = reservation
            self.reserved_tokens += reservation.estimated_tokens
            self.reserved_spend_usd += reservation.estimated_cost
            return reservation

    def settle_usage(
        self,
        reservation: UsageReservation,
        *,
        actual_tokens: int | None,
        actual_cost: float | None,
    ) -> tuple[int, float]:
        """结算预留；缺少最终用量时保守采用预估值。"""
        with self.budget_lock:
            active = self.reservations.pop(reservation.identifier, None)
            if active is None:
                return (0, 0.0)
            self.reserved_tokens -= active.estimated_tokens
            self.reserved_spend_usd -= active.estimated_cost
            settled_tokens = active.estimated_tokens if actual_tokens is None else actual_tokens
            settled_cost = active.estimated_cost if actual_cost is None else actual_cost
            self.total_token_count += max(settled_tokens, 0)
            self.total_spend_usd += max(settled_cost, 0.0)
            return settled_tokens, settled_cost

    def release_usage(self, reservation: UsageReservation) -> None:
        """模型请求未开始或明确失败时释放预留。"""
        with self.budget_lock:
            active = self.reservations.pop(reservation.identifier, None)
            if active is None:
                return
            self.reserved_tokens -= active.estimated_tokens
            self.reserved_spend_usd -= active.estimated_cost

    @asynccontextmanager
    async def request_slot(self) -> AsyncGenerator[None]:
        """限制一个 Runtime 内所有会话共享的并发模型请求数。"""
        async with self.request_semaphore:
            yield

    def get_model_names(self) -> list[str]:
        return list(dict.fromkeys(item.model_name for item in self.deployments))

    def get_model_list(self) -> list[ModelDeployment]:
        """返回不含密钥的类型化部署快照。"""
        return list(self.deployments)

    def register_model(self, deployment: ModelDeployment) -> None:
        self.deployments.append(deployment)
        model_list = [
            self.to_litellm_deployment(item, self.environ)
            for item in self.deployments
        ]
        self.litellm_router.set_model_list(model_list)  # pyright: ignore[reportUnknownMemberType]

    def unregister_model(self, model_name: str) -> int:
        original_count = len(self.deployments)
        remaining_deployments = [
            item for item in self.deployments if item.model_name != model_name
        ]
        if not remaining_deployments:
            raise GatewayError("不能移除最后一个模型部署")
        model_list = [
            self.to_litellm_deployment(item, self.environ)
            for item in remaining_deployments
        ]
        self.litellm_router.set_model_list(model_list)  # pyright: ignore[reportUnknownMemberType]
        self.deployments = remaining_deployments
        return original_count - len(self.deployments)

    async def close(self) -> None:
        """释放网关资源；LiteLLM Router 当前没有异步持有资源需要关闭。"""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        from praxis.gateway.chat import chat

        return await chat(self, messages, model=model, tools=tools, **kwargs)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelResponseChunk]:
        from praxis.gateway.chat import chat_stream

        async with aclosing(chat_stream(self, messages, model=model, tools=tools, **kwargs)) as stream:
            async for chunk in stream:
                yield chunk

    async def health(self) -> bool:
        """Execute a bounded live completion and cache the result for the configured TTL."""
        now = time.monotonic()
        if self.health_probe_value is not None and now < self.health_probe_expires_at:
            return self.health_probe_value

        async with self.health_probe_lock:
            now = time.monotonic()
            if self.health_probe_value is not None and now < self.health_probe_expires_at:
                return self.health_probe_value
            try:
                async with asyncio.timeout(self.gateway_config.health_probe_timeout):
                    await self.complete(
                        [{"role": "user", "content": "Reply with OK."}],
                        model=self.gateway_config.default_model,
                        max_tokens=1,
                        temperature=0,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                value = False
                self.last_health_error = type(exc).__name__
            else:
                value = True
                self.last_health_error = ""
            self.health_probe_value = value
            self.health_probe_expires_at = (
                time.monotonic() + self.gateway_config.health_probe_ttl
            )
            return value
