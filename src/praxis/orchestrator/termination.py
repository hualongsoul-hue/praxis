"""终止条件管理。

6 层优先级评估：
1. 护栏绊线触发  2. 用户中断  3. 安全拒绝
4. 自然终止      5. 最大轮次  6. Token 耗尽
所有阈值通过 S1 配置。
"""

from praxis.config.subsystems import OrchestratorConfig
from praxis.models.context import TokenUsage
from praxis.models.orchestrator import LoopState, TerminationReason
from praxis.telemetry.logger import get_logger

log = get_logger("orchestrator.termination")


class TerminationManager:
    """终止条件管理器。

    按 6 层优先级评估终止条件，返回首个命中的终止原因。
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.max_turns = config.max_turns

    def evaluate(
        self,
        state: LoopState,
        tripwire: bool = False,
        safety_refusal: bool = False,
        is_final_response: bool = False,
        token_usage: TokenUsage | None = None,
    ) -> TerminationReason | None:
        """评估终止条件。

        按优先级依次检查，返回首个命中的终止原因，全部通过则返回 None。

        Args:
            state: 当前循环状态。
            tripwire: 护栏绊线是否触发。
            safety_refusal: 是否安全拒绝。
            is_final_response: 模型是否返回最终响应（无工具调用）。
            token_usage: 当前 Token 用量。

        Returns:
            终止原因，None 表示继续循环。
        """
        # Priority 1: 护栏绊线
        if tripwire:
            log.warning("终止: 护栏绊线触发")
            return TerminationReason.TRIPWIRE

        # Priority 2: 用户中断
        if state.aborted:
            log.info("终止: 用户中断")
            return TerminationReason.USER_ABORT

        # Priority 3: 安全拒绝
        if safety_refusal:
            log.warning("终止: 安全拒绝")
            return TerminationReason.SAFETY_REFUSAL

        # Priority 4: 自然终止
        if is_final_response:
            log.info("终止: 自然终止（模型返回最终响应）")
            return TerminationReason.NATURAL

        # Priority 5: 最大轮次
        if state.current_turn >= self.max_turns:
            log.warning("终止: 达到最大轮次", max_turns=self.max_turns)
            return TerminationReason.MAX_TURNS

        # Priority 6: Token 耗尽
        if token_usage is not None and token_usage.usage_ratio >= 0.95:
            log.warning(
                "终止: Token 耗尽",
                usage_ratio=token_usage.usage_ratio,
            )
            return TerminationReason.TOKEN_EXHAUSTED

        return None
