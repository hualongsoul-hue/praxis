"""Token 计量、成本估算与预算管控。"""

import json
from typing import Any

import litellm

from praxis.gateway.router import GatewayRouter
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("gateway.metering")
DEFAULT_MAX_TOKENS = 128_000


def get_token_count(messages: list[dict[str, Any]], model: str = "default") -> int:
    """使用 LiteLLM 的模型 tokenizer 估算消息 Token 数。"""
    try:
        return litellm.token_counter(  # pyright: ignore[reportUnknownMemberType]
            model=model,
            messages=messages,
        )
    except ValueError as exc:
        encoded = json.dumps(
            messages,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        log.debug(
            "LiteLLM tokenizer does not support a content block; using a conservative byte bound",
            model=model,
            error=str(exc),
        )
        return len(encoded)


def get_max_tokens(model: str = "default") -> int:
    """查询上下文窗口；未知模型返回保守的默认值。"""
    try:
        return int(litellm.get_max_tokens(model))
    except Exception as exc:
        log.debug("上下文窗口查询失败，使用默认值", model=model, error=str(exc))
        return DEFAULT_MAX_TOKENS


def completion_cost(response: Any) -> float:
    """按 LiteLLM 价格表计算完整响应成本。"""
    return float(litellm.completion_cost(completion_response=response))


def estimate_input_cost(
    estimated_tokens: int,
    model: str = "default",
) -> float | None:
    """估算输入成本；价格未知时返回 ``None``，不得视作免费。"""
    try:
        prompt_cost, completion_cost_ = litellm.cost_per_token(
            model=model,
            prompt_tokens=estimated_tokens,
            completion_tokens=0,
        )
        total = float(prompt_cost) + float(completion_cost_)
        return total if estimated_tokens == 0 or total > 0 else None
    except Exception as exc:
        log.debug("成本估算失败", model=model, error=str(exc))
        return None


def estimate_output_cost(
    estimated_tokens: int,
    model: str = "default",
) -> float | None:
    """估算输出成本；价格未知时返回 ``None``。"""
    try:
        prompt_cost, completion_cost_ = litellm.cost_per_token(
            model=model,
            prompt_tokens=0,
            completion_tokens=estimated_tokens,
        )
        total = float(prompt_cost) + float(completion_cost_)
        return total if estimated_tokens == 0 or total > 0 else None
    except Exception as exc:
        log.debug("输出成本估算失败", model=model, error=str(exc))
        return None


def check_budget(
    gateway: GatewayRouter,
    estimated_tokens: int,
    model: str = "default",
) -> None:
    """原子检查 Token/金额预算，不保留调用预留。"""
    estimated_cost = (
        estimate_input_cost(estimated_tokens, model)
        if gateway.config.max_budget is not None
        else None
    )
    reservation = gateway.reserve_usage(
        estimated_tokens=estimated_tokens,
        estimated_cost=estimated_cost,
    )
    gateway.release_usage(reservation)


def record_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
) -> None:
    """记录模型 Token 与成本指标。"""
    tags = {"model": model}
    emit_metric("llm_tokens_input", float(prompt_tokens), tags, "counter")
    emit_metric("llm_tokens_output", float(completion_tokens), tags, "counter")
    if cost > 0:
        emit_metric("llm_cost", cost, tags, "counter")
