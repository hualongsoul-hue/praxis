"""四类错误分类与恢复策略。

classify_error 返回 ErrorClassification（Transient/Model-Recoverable/User-Fixable/Unexpected），
每类附带建议策略。
"""

from praxis.exceptions import (
    AuthenticationError,
    BudgetExceededError,
    ContextWindowExceededError,
    GatewayTimeoutError,
    GuardrailError,
    ModelNotFoundError,
    PraxisError,
    ProviderUnavailableError,
    RateLimitError,
    SandboxViolationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from praxis.models.recovery import (
    ErrorCategory,
    ErrorClassification,
    RecoveryStrategy,
)

TRANSIENT_TYPES: set[type] = {
    GatewayTimeoutError,
    RateLimitError,
    ProviderUnavailableError,
    ToolTimeoutError,
}

MODEL_RECOVERABLE_TYPES: set[type] = {
    ToolNotFoundError,
    ToolExecutionError,
    ContextWindowExceededError,
    ModelNotFoundError,
}

USER_FIXABLE_TYPES: set[type] = {
    AuthenticationError,
    BudgetExceededError,
    SandboxViolationError,
    GuardrailError,
}


def classify_error(error: Exception) -> ErrorClassification:
    """对异常进行分类，返回分类结果和建议恢复策略。

    分类优先级：瞬态 > LLM 可恢复 > 用户可修复 > 未预期。

    Args:
        error: 异常对象。

    Returns:
        ErrorClassification 包含分类、策略和消息。
    """
    error_type = type(error)
    original_type = f"{error_type.__module__}.{error_type.__qualname__}"
    message = str(error)

    if isinstance(error, PraxisError):
        message = error.message

    if error_type in TRANSIENT_TYPES:
        return ErrorClassification(
            category=ErrorCategory.TRANSIENT,
            strategy=RecoveryStrategy.RETRY,
            message=message,
            original_type=original_type,
        )

    if error_type in MODEL_RECOVERABLE_TYPES:
        return ErrorClassification(
            category=ErrorCategory.MODEL_RECOVERABLE,
            strategy=RecoveryStrategy.RETURN_TO_MODEL,
            message=message,
            original_type=original_type,
        )

    if error_type in USER_FIXABLE_TYPES:
        return ErrorClassification(
            category=ErrorCategory.USER_FIXABLE,
            strategy=RecoveryStrategy.ASK_USER,
            message=message,
            original_type=original_type,
        )

    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return ErrorClassification(
            category=ErrorCategory.TRANSIENT,
            strategy=RecoveryStrategy.RETRY,
            message=message,
            original_type=original_type,
        )

    return ErrorClassification(
        category=ErrorCategory.UNEXPECTED,
        strategy=RecoveryStrategy.ABORT,
        message=message,
        original_type=original_type,
    )
