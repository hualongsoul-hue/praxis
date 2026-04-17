"""重试策略。

指数退避 + 随机抖动 + 可配置上限，重试前检查熔断器状态。
"""

import random

from praxis.models.recovery import RetryDecision


class RetryPolicy:
    """重试策略管理器。

    支持按工具配置最大重试次数、退避参数。
    """

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter_factor: float = 0.25,
    ) -> None:
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor
        self.attempt_counts: dict[str, int] = {}

    def get_retry_decision(self, tool_name: str, attempt_count: int) -> RetryDecision:
        """计算重试决策。

        Args:
            tool_name: 工具名称。
            attempt_count: 已重试次数。

        Returns:
            RetryDecision 包含是否重试和等待时间。
        """
        if attempt_count >= self.max_retries:
            return RetryDecision(
                should_retry=False,
                reason=f"已达最大重试次数 ({self.max_retries})",
            )

        base_delay = min(
            self.initial_delay * (2 ** attempt_count),
            self.max_delay,
        )
        jitter = base_delay * self.jitter_factor * (2 * random.random() - 1)
        wait = max(0.0, base_delay + jitter)

        return RetryDecision(
            should_retry=True,
            wait_seconds=round(wait, 3),
            reason=f"第 {attempt_count + 1} 次重试，等待 {wait:.3f}s",
        )

    def record_attempt(self, tool_name: str) -> int:
        """记录一次重试尝试，返回当前累计次数。"""
        self.attempt_counts[tool_name] = self.attempt_counts.get(tool_name, 0) + 1
        return self.attempt_counts[tool_name]

    def reset(self, tool_name: str) -> None:
        """重置指定工具的重试计数。"""
        self.attempt_counts.pop(tool_name, None)

    def reset_all(self) -> None:
        """重置所有重试计数。"""
        self.attempt_counts.clear()
