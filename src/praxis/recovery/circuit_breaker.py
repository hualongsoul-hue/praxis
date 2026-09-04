"""熔断器（Circuit Breaker）。

每个工具独立维护熔断器实例，三态模型 Closed→Open→Half-Open。
状态转换通过 S2 遥测记录。
"""

import time
from typing import Any, cast

from praxis.models.recovery import CircuitState
from praxis.telemetry.metrics import emit_metric


class CircuitBreaker:
    """单工具熔断器。"""

    def __init__(
        self,
        tool_name: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.tool_name = tool_name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self.last_state_change: float = time.monotonic()

    def check(self) -> CircuitState:
        """检查当前熔断器状态。

        如果处于 OPEN 状态且已过 cooldown 时间，自动转为 HALF_OPEN。

        Returns:
            当前熔断器状态。
        """
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.cooldown_seconds:
                self.transition_to(CircuitState.HALF_OPEN)

        return self.state

    def record_success(self) -> None:
        """记录成功执行。

        HALF_OPEN 状态下成功，回到 CLOSED。
        """
        if self.state == CircuitState.HALF_OPEN:
            self.transition_to(CircuitState.CLOSED)
        self.failure_count = 0

    def record_failure(self) -> None:
        """记录失败执行。

        CLOSED 状态下连续失败超过阈值，转为 OPEN。
        HALF_OPEN 状态下探测失败，重新转为 OPEN。
        """
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN or (self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold):
            self.transition_to(CircuitState.OPEN)

    def transition_to(self, new_state: CircuitState) -> None:
        """状态转换并记录到遥测。"""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.monotonic()

        if new_state == CircuitState.CLOSED:
            self.failure_count = 0

        emit_metric(
            "circuit_breaker_transition",
            1.0,
            {
                "tool": self.tool_name,
                "from": old_state.value,
                "to": new_state.value,
            },
            "counter",
        )


class CircuitBreakerRegistry:
    """工具级熔断器注册表。"""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.breakers: dict[str, CircuitBreaker] = {}

    def get(self, tool_name: str) -> CircuitBreaker:
        """获取指定工具的熔断器，不存在则创建。"""
        if tool_name not in self.breakers:
            self.breakers[tool_name] = CircuitBreaker(
                tool_name=tool_name,
                failure_threshold=self.failure_threshold,
                cooldown_seconds=self.cooldown_seconds,
            )
        return self.breakers[tool_name]

    def check_circuit(self, tool_name: str) -> CircuitState:
        """检查指定工具的熔断器状态。"""
        return self.get(tool_name).check()

    def record_outcome(self, tool_name: str, success: bool) -> None:
        """记录工具执行结果。"""
        breaker = self.get(tool_name)
        if success:
            breaker.record_success()
        else:
            breaker.record_failure()

    def export_state(self) -> dict[str, dict[str, Any]]:
        """Return durable circuit state without persisting monotonic timestamps."""
        return {
            name: {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
            }
            for name, breaker in self.breakers.items()
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore breaker states while resetting process-local timestamps."""
        self.breakers.clear()
        for name, raw in state.items():
            if not isinstance(raw, dict):
                continue
            values = cast(dict[str, object], raw)
            breaker = self.get(name)
            try:
                breaker.state = CircuitState(str(values.get("state", CircuitState.CLOSED.value)))
            except ValueError:
                breaker.state = CircuitState.CLOSED
            count = values.get("failure_count", 0)
            breaker.failure_count = count if isinstance(count, int) and count >= 0 else 0
            breaker.last_failure_time = time.monotonic()
