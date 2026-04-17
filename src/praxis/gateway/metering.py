"""Token 计量、成本追踪与预算管控。

提供 Token 预估、上下文窗口查询、成本计算和预算检查。
通过 S2 遥测发射 llm_tokens_* 和 llm_cost 指标。
"""

from typing import Any

import litellm

from praxis.exceptions import BudgetExceededError
from praxis.gateway.router import GatewayRouter
from praxis.telemetry.metrics import emit_metric


def get_token_count(messages: list[dict[str, Any]], model: str = "default") -> int:
    """预估消息列表的 Token 数量。

    使用 LiteLLM 内置的各 Provider 特定 tokenizer，
    未知模型自动回退到 tiktoken。

    Args:
        messages: OpenAI 格式消息列表。
        model: 模型名称，用于选择正确的 tokenizer。

    Returns:
        预估 Token 数量。
    """
    return litellm.token_counter(model=model, messages=messages)


DEFAULT_MAX_TOKENS = 128_000


def get_max_tokens(model: str = "default") -> int:
    """查询模型最大上下文窗口大小。

    对于 LiteLLM 模型数据库中未收录的模型（如 Router 别名、自部署模型），
    回退到 DEFAULT_MAX_TOKENS。

    Args:
        model: 模型名称。

    Returns:
        模型最大 Token 数。
    """
    try:
        return litellm.get_max_tokens(model)  # type: ignore[return-value]
    except Exception:
        return DEFAULT_MAX_TOKENS


def completion_cost(response: Any) -> float:
    """计算单次调用成本（USD）。

    使用 LiteLLM 社区维护的模型价格表。

    Args:
        response: LiteLLM 原始响应对象。

    Returns:
        成本（USD）。
    """
    cost = litellm.completion_cost(completion_response=response)
    return float(cost)


def check_budget(gateway: GatewayRouter, estimated_tokens: int, model: str = "default") -> None:
    """调用前预算检查。

    如果配置了 max_budget 且预估成本将超出预算，抛出 BudgetExceededError。

    Args:
        gateway: 网关路由器实例。
        estimated_tokens: 预估输入 Token 数。
        model: 模型名称。
    """
    max_budget = gateway.config.max_budget
    if max_budget is None:
        return

    try:
        estimated_cost = litellm.completion_cost(
            model=model,
            prompt=str(estimated_tokens),
            completion="",
        )
    except Exception:
        return

    if estimated_cost > max_budget:
        raise BudgetExceededError(
            f"预估成本 ${estimated_cost:.4f} 超出预算上限 ${max_budget:.4f}",
            details={
                "estimated_cost": estimated_cost,
                "max_budget": max_budget,
                "model": model,
            },
        )


def record_usage(model: str, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
    """记录用量指标到 S2 遥测。

    Args:
        model: 模型名称。
        prompt_tokens: 输入 Token 数。
        completion_tokens: 输出 Token 数。
        cost: 调用成本（USD）。
    """
    tags = {"model": model}
    emit_metric("llm_tokens_input", float(prompt_tokens), tags, "counter")
    emit_metric("llm_tokens_output", float(completion_tokens), tags, "counter")
    if cost > 0:
        emit_metric("llm_cost", cost, tags, "counter")
