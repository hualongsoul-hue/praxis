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
    ToolExecutionError,
    ToolNotFoundError,
    ToolPolicyViolationError,
    ToolTimeoutError,
)
from praxis.models.recovery import (
    ErrorCategory,
    ErrorClassification,
    RecoveryStrategy,
)

TRANSIENT_TYPES: frozenset[type[Exception]] = frozenset({
    GatewayTimeoutError,
    RateLimitError,
    ProviderUnavailableError,
    ToolTimeoutError,
})

MODEL_RECOVERABLE_TYPES: frozenset[type[Exception]] = frozenset({
    ToolNotFoundError,
    ToolExecutionError,
    ContextWindowExceededError,
    ModelNotFoundError,
})

USER_FIXABLE_TYPES: frozenset[type[Exception]] = frozenset({
    AuthenticationError,
    BudgetExceededError,
    ToolPolicyViolationError,
    GuardrailError,
})


def classify_by_type_name(
    qualified_name: str,
    message: str = "",
) -> ErrorClassification | None:
    """根据完全限定类型名分类（用于仅有字符串类型信息的场景）。

    Args:
        qualified_name: 异常完全限定名，如 ``praxis.exceptions.ToolTimeoutError``。
        message: 附带消息。

    Returns:
        分类结果；若类型名未知返回 None。
    """
    by_name = {
        f"{cls.__module__}.{cls.__qualname__}": cls
        for cls in (
            *TRANSIENT_TYPES,
            *MODEL_RECOVERABLE_TYPES,
            *USER_FIXABLE_TYPES,
        )
    }
    cls = by_name.get(qualified_name)
    if cls is None:
        return None
    if cls in TRANSIENT_TYPES:
        category = ErrorCategory.TRANSIENT
        strategy = RecoveryStrategy.RETRY
    elif cls in MODEL_RECOVERABLE_TYPES:
        category = ErrorCategory.MODEL_RECOVERABLE
        strategy = RecoveryStrategy.RETURN_TO_MODEL
    else:
        category = ErrorCategory.USER_FIXABLE
        strategy = RecoveryStrategy.ASK_USER
    return ErrorClassification(
        category=category,
        strategy=strategy,
        message=message,
        original_type=qualified_name,
    )


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
