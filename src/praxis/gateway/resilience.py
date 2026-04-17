"""异常标准化。

将 LiteLLM 异常映射到 Praxis 内部异常体系，供 S9 错误恢复进行分类和策略决策。
"""

import litellm

from praxis.exceptions import (
    AuthenticationError,
    BudgetExceededError,
    ContextWindowExceededError,
    GatewayError,
    GatewayTimeoutError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)

EXCEPTION_MAP: dict[type[Exception], type[GatewayError]] = {
    litellm.AuthenticationError: AuthenticationError,
    litellm.RateLimitError: RateLimitError,
    litellm.NotFoundError: ModelNotFoundError,
    litellm.ContextWindowExceededError: ContextWindowExceededError,
    litellm.BudgetExceededError: BudgetExceededError,
    litellm.ServiceUnavailableError: ProviderUnavailableError,
    litellm.Timeout: GatewayTimeoutError,
}


def map_litellm_exception(exc: Exception) -> GatewayError:
    """将 LiteLLM 异常转换为 Praxis GatewayError 子类。

    遍历 EXCEPTION_MAP 查找匹配类型（含子类），
    未匹配时回退到通用 GatewayError。
    """
    for litellm_type, praxis_type in EXCEPTION_MAP.items():
        if isinstance(exc, litellm_type):
            return praxis_type(
                str(exc),
                details={"original_type": type(exc).__name__},
            )
    return GatewayError(
        str(exc),
        details={"original_type": type(exc).__name__},
    )
