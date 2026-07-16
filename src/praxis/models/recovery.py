"""错误恢复数据模型——S9 内部及跨组件共享。"""

from enum import StrEnum

from pydantic import BaseModel


class ErrorCategory(StrEnum):
    """错误分类类别。"""

    TRANSIENT = "transient"
    MODEL_RECOVERABLE = "model_recoverable"
    USER_FIXABLE = "user_fixable"
    UNEXPECTED = "unexpected"


class RecoveryStrategy(StrEnum):
    """恢复策略建议。"""

    RETRY = "retry"
    RETURN_TO_MODEL = "return_to_model"
    ASK_USER = "ask_user"
    ABORT = "abort"


class ErrorClassification(BaseModel):
    """错误分类结果，附带建议恢复策略。"""

    category: ErrorCategory
    strategy: RecoveryStrategy
    message: str
    original_type: str


class RetryDecision(BaseModel):
    """重试决策。"""

    should_retry: bool
    wait_seconds: float = 0.0
    reason: str = ""


class CircuitState(StrEnum):
    """熔断器状态。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
