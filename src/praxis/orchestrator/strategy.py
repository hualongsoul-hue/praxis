"""循环策略选择。

ReAct 模式（默认，交叉推理与行动）和
Plan-and-Execute 模式（先规划后执行），运行时可切换。
"""

from typing import Any, cast
from uuid import uuid4

from praxis.models.orchestrator import StrategyMode
from praxis.telemetry.logger import get_logger

log = get_logger("orchestrator.strategy")


class PlanStep:
    """Plan-and-Execute 模式下的计划步骤。"""

    __slots__ = ("completed", "description", "result", "tool_hint")

    def __init__(
        self,
        description: str,
        tool_hint: str = "",
    ) -> None:
        self.description = description
        self.tool_hint = tool_hint
        self.completed: bool = False
        self.result: str = ""


class LoopStrategy:
    """循环策略管理器。

    管理 ReAct 和 Plan-and-Execute 两种模式的切换和状态。
    """

    def __init__(self, mode: StrategyMode = StrategyMode.REACT) -> None:
        self.mode = mode
        self.plan: list[PlanStep] = []
        self.current_step_index: int = 0
        self.request_id: str = ""

    def begin_request(self, request_id: str | None = None) -> str:
        """Create isolated plan state for a new user request."""
        self.request_id = request_id or uuid4().hex
        self.plan.clear()
        self.current_step_index = 0
        return self.request_id

    def switch_mode(self, mode: StrategyMode) -> None:
        """运行时切换策略模式。"""
        if mode == self.mode:
            return
        log.info("策略模式切换", from_mode=self.mode.value, to_mode=mode.value)
        self.mode = mode
        if mode == StrategyMode.REACT:
            self.plan.clear()
            self.current_step_index = 0

    def set_plan(self, steps: list[PlanStep]) -> None:
        """设置执行计划（仅 Plan-and-Execute 模式）。"""
        self.plan = steps
        self.current_step_index = 0
        log.info("执行计划已设置", step_count=len(steps))

    def get_current_step(self) -> PlanStep | None:
        """获取当前待执行步骤。"""
        if self.mode != StrategyMode.PLAN_AND_EXECUTE:
            return None
        if self.current_step_index >= len(self.plan):
            return None
        return self.plan[self.current_step_index]

    def advance_step(self, result: str = "") -> bool:
        """标记当前步骤完成，前进到下一步。

        Args:
            result: 当前步骤执行结果。

        Returns:
            是否还有剩余步骤。
        """
        step = self.get_current_step()
        if step is None:
            return False
        step.completed = True
        step.result = result
        self.current_step_index += 1
        return self.current_step_index < len(self.plan)

    def is_plan_complete(self) -> bool:
        """计划是否全部完成。"""
        if self.mode != StrategyMode.PLAN_AND_EXECUTE:
            return True
        return self.current_step_index >= len(self.plan)

    def get_plan_context(self) -> str:
        """生成计划上下文（注入到 Prompt 中引导 LLM）。

        Returns:
            格式化的计划进度文本。
        """
        if self.mode != StrategyMode.PLAN_AND_EXECUTE or not self.plan:
            return ""

        lines = ["<execution_plan>"]
        for i, step in enumerate(self.plan):
            status = "✓" if step.completed else ("→" if i == self.current_step_index else " ")
            lines.append(f"  {status} {i + 1}. {step.description}")
            if step.completed and step.result:
                lines.append(f"      结果: {step.result}")
        lines.append("</execution_plan>")
        return "\n".join(lines)

    def get_step_instruction(self) -> str:
        """获取当前步骤的执行指令（注入用户消息后补充引导）。"""
        step = self.get_current_step()
        if step is None:
            return ""
        instruction = f"\n\n当前执行步骤 {self.current_step_index + 1}: {step.description}"
        if step.tool_hint:
            instruction += f"\n建议使用工具: {step.tool_hint}"
        return instruction

    def export_state(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the complete strategy state."""
        return {
            "mode": self.mode.value,
            "request_id": self.request_id,
            "current_step_index": self.current_step_index,
            "plan": [
                {
                    "description": step.description,
                    "tool_hint": step.tool_hint,
                    "completed": step.completed,
                    "result": step.result,
                }
                for step in self.plan
            ],
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`export_state`."""
        mode_value = state.get("mode", StrategyMode.REACT.value)
        self.mode = StrategyMode(str(mode_value))
        self.request_id = str(state.get("request_id", ""))
        plan_value = state.get("plan", [])
        plan_items = cast(list[object], plan_value) if isinstance(plan_value, list) else []
        restored: list[PlanStep] = []
        for item in plan_items:
            if not isinstance(item, dict):
                continue
            values = cast(dict[str, object], item)
            step = PlanStep(
                description=str(values.get("description", "")),
                tool_hint=str(values.get("tool_hint", "")),
            )
            step.completed = bool(values.get("completed", False))
            step.result = str(values.get("result", ""))
            restored.append(step)
        self.plan = restored
        index_value = state.get("current_step_index", 0)
        self.current_step_index = (
            min(index_value, len(restored)) if isinstance(index_value, int) and index_value >= 0 else 0
        )
