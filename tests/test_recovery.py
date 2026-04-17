"""S9 错误恢复验证测试。"""

import time

import pytest

from praxis.exceptions import (
    AuthenticationError,
    GatewayTimeoutError,
    RateLimitError,
    SandboxViolationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from praxis.models.recovery import CircuitState, ErrorCategory, RecoveryStrategy
from praxis.recovery.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from praxis.recovery.classifier import classify_error
from praxis.recovery.fallback import FallbackRegistry
from praxis.recovery.retry import RetryPolicy


# ── Task 6.4: 错误分类 ─────────────────────────────────────────────────────


class TestErrorClassifier:
    """Task 6.4: 错误分类验证。"""

    def test_transient_timeout(self) -> None:
        exc = GatewayTimeoutError("timeout")
        result = classify_error(exc)
        assert result.category == ErrorCategory.TRANSIENT
        assert result.strategy == RecoveryStrategy.RETRY

    def test_transient_rate_limit(self) -> None:
        exc = RateLimitError("rate limited")
        result = classify_error(exc)
        assert result.category == ErrorCategory.TRANSIENT

    def test_transient_tool_timeout(self) -> None:
        exc = ToolTimeoutError("tool timed out")
        result = classify_error(exc)
        assert result.category == ErrorCategory.TRANSIENT

    def test_model_recoverable_tool_not_found(self) -> None:
        exc = ToolNotFoundError("tool 'x' not found")
        result = classify_error(exc)
        assert result.category == ErrorCategory.MODEL_RECOVERABLE
        assert result.strategy == RecoveryStrategy.RETURN_TO_MODEL

    def test_model_recoverable_tool_execution(self) -> None:
        exc = ToolExecutionError("bad args")
        result = classify_error(exc)
        assert result.category == ErrorCategory.MODEL_RECOVERABLE

    def test_user_fixable_auth(self) -> None:
        exc = AuthenticationError("invalid api key")
        result = classify_error(exc)
        assert result.category == ErrorCategory.USER_FIXABLE
        assert result.strategy == RecoveryStrategy.ASK_USER

    def test_user_fixable_sandbox(self) -> None:
        exc = SandboxViolationError("path blocked")
        result = classify_error(exc)
        assert result.category == ErrorCategory.USER_FIXABLE

    def test_unexpected_runtime_error(self) -> None:
        exc = RuntimeError("something broke")
        result = classify_error(exc)
        assert result.category == ErrorCategory.UNEXPECTED
        assert result.strategy == RecoveryStrategy.ABORT

    def test_connection_error_is_transient(self) -> None:
        exc = ConnectionError("connection refused")
        result = classify_error(exc)
        assert result.category == ErrorCategory.TRANSIENT

    def test_original_type_recorded(self) -> None:
        exc = ValueError("test")
        result = classify_error(exc)
        assert "ValueError" in result.original_type


# ── Task 6.5: 重试策略 ─────────────────────────────────────────────────────


class TestRetryPolicy:
    """Task 6.5: 重试策略验证。"""

    def test_first_attempt_retries(self) -> None:
        policy = RetryPolicy(max_retries=3)
        decision = policy.get_retry_decision("tool_a", 0)
        assert decision.should_retry is True
        assert decision.wait_seconds > 0

    def test_exceeds_max_retries(self) -> None:
        policy = RetryPolicy(max_retries=2)
        decision = policy.get_retry_decision("tool_a", 2)
        assert decision.should_retry is False

    def test_exponential_backoff(self) -> None:
        policy = RetryPolicy(max_retries=5, initial_delay=1.0, jitter_factor=0.0)
        d0 = policy.get_retry_decision("t", 0)
        d1 = policy.get_retry_decision("t", 1)
        d2 = policy.get_retry_decision("t", 2)
        assert d1.wait_seconds >= d0.wait_seconds
        assert d2.wait_seconds >= d1.wait_seconds

    def test_max_delay_cap(self) -> None:
        policy = RetryPolicy(max_retries=10, initial_delay=1.0, max_delay=5.0, jitter_factor=0.0)
        decision = policy.get_retry_decision("t", 8)
        assert decision.wait_seconds <= 5.0

    def test_record_and_reset(self) -> None:
        policy = RetryPolicy()
        assert policy.record_attempt("t") == 1
        assert policy.record_attempt("t") == 2
        policy.reset("t")
        assert policy.record_attempt("t") == 1

    def test_reset_all(self) -> None:
        policy = RetryPolicy()
        policy.record_attempt("a")
        policy.record_attempt("b")
        policy.reset_all()
        assert len(policy.attempt_counts) == 0


# ── Task 6.5: 熔断器 ───────────────────────────────────────────────────────


class TestCircuitBreaker:
    """Task 6.5: 熔断器验证。"""

    def test_initial_state_closed(self) -> None:
        cb = CircuitBreaker("tool_a")
        assert cb.check() == CircuitState.CLOSED

    def test_failures_open_circuit(self) -> None:
        cb = CircuitBreaker("tool_a", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.check() == CircuitState.CLOSED
        cb.record_failure()
        assert cb.check() == CircuitState.OPEN

    def test_cooldown_half_open(self) -> None:
        cb = CircuitBreaker("tool_a", failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        assert cb.check() == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.check() == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker("tool_a", failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        time.sleep(0.15)
        cb.check()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker("tool_a", failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        time.sleep(0.15)
        cb.check()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerRegistry:
    """Task 6.5: 熔断器注册表验证。"""

    def test_auto_create(self) -> None:
        registry = CircuitBreakerRegistry()
        state = registry.check_circuit("new_tool")
        assert state == CircuitState.CLOSED
        assert "new_tool" in registry.breakers

    def test_record_outcome(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=2)
        registry.record_outcome("t", success=False)
        registry.record_outcome("t", success=False)
        assert registry.check_circuit("t") == CircuitState.OPEN

    def test_success_resets(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=2, cooldown_seconds=0.1)
        registry.record_outcome("t", success=False)
        registry.record_outcome("t", success=False)
        assert registry.check_circuit("t") == CircuitState.OPEN
        time.sleep(0.15)
        registry.check_circuit("t")
        registry.record_outcome("t", success=True)
        assert registry.check_circuit("t") == CircuitState.CLOSED


# ── Task 6.6: 优雅降级 ─────────────────────────────────────────────────────


class TestFallback:
    """Task 6.6: 优雅降级验证。"""

    def test_register_and_get(self) -> None:
        registry = FallbackRegistry()
        registry.register("web_fetch", "web_search")
        assert registry.get_fallback("web_fetch") == "web_search"

    def test_no_fallback_returns_none(self) -> None:
        registry = FallbackRegistry()
        assert registry.get_fallback("no_such_tool") is None

    def test_unregister(self) -> None:
        registry = FallbackRegistry()
        registry.register("a", "b")
        assert registry.unregister("a") is True
        assert registry.unregister("a") is False
        assert registry.get_fallback("a") is None

    def test_load_mappings(self) -> None:
        registry = FallbackRegistry()
        registry.load_mappings({"web_fetch": "web_search", "code_search": "grep_search"})
        assert registry.get_fallback("web_fetch") == "web_search"
        assert registry.get_fallback("code_search") == "grep_search"

    def test_list_mappings(self) -> None:
        registry = FallbackRegistry()
        registry.register("a", "b")
        registry.register("c", "d")
        mappings = registry.list_mappings()
        assert mappings == {"a": "b", "c": "d"}
