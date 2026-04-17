"""praxis.recovery — 错误恢复（S9）：错误分类、重试策略、熔断器。"""

from praxis.recovery.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from praxis.recovery.classifier import classify_error
from praxis.recovery.fallback import FallbackRegistry
from praxis.recovery.retry import RetryPolicy

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "FallbackRegistry",
    "RetryPolicy",
    "classify_error",
]
