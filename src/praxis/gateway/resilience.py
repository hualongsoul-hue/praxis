"""异常标准化。

将 LiteLLM 异常映射到 Praxis 内部异常体系，供 S9 错误恢复进行分类和策略决策。
"""

from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
)
from litellm.exceptions import (
    BudgetExceededError as LiteLLMBudgetExceededError,
)
from litellm.exceptions import (
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from litellm.exceptions import (
    NotFoundError as LiteLLMNotFoundError,
)
from litellm.exceptions import (
    RateLimitError as LiteLLMRateLimitError,
)
from litellm.exceptions import (
    ServiceUnavailableError as LiteLLMServiceUnavailableError,
)
from litellm.exceptions import (
    Timeout as LiteLLMTimeout,
)

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
    LiteLLMAuthenticationError: AuthenticationError,
    LiteLLMRateLimitError: RateLimitError,
    LiteLLMNotFoundError: ModelNotFoundError,
    LiteLLMContextWindowExceededError: ContextWindowExceededError,
    LiteLLMBudgetExceededError: BudgetExceededError,
    LiteLLMServiceUnavailableError: ProviderUnavailableError,
    LiteLLMTimeout: GatewayTimeoutError,
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
