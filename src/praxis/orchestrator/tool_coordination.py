"""工具调用协调流程。

按序调用：
S8.check_tool_call → S9.check_circuit → S5.execute_tool →
S9.record_outcome → 失败时 S9.classify_error 决策。
"""

from typing import Any

from json_repair import repair_json

from praxis.guardrails.engine import GuardrailEngine
from praxis.models.guardrails import VerdictType
from praxis.models.orchestrator import AgentEvent
from praxis.models.recovery import CircuitState, ErrorCategory, ErrorClassification
from praxis.models.tools import ToolCall, ToolResult
from praxis.orchestrator.events import EventEmitter
from praxis.recovery.circuit_breaker import CircuitBreakerRegistry
from praxis.recovery.classifier import classify_by_type_name, classify_error
from praxis.recovery.fallback import FallbackRegistry
from praxis.recovery.retry import RetryPolicy
from praxis.telemetry.logger import get_logger
from praxis.tools.executor import ToolExecutor
from praxis.tools.registry import ToolRegistry

log = get_logger("orchestrator.tool_coordination")


class ToolCallOutcome:
    """单次工具调用结果。"""

    __slots__ = (
        "tool_call",
        "result",
        "skipped",
        "skip_reason",
        "needs_user_confirm",
    )

    def __init__(
        self,
        tool_call: ToolCall,
        result: ToolResult | None = None,
        skipped: bool = False,
        skip_reason: str = "",
        needs_user_confirm: bool = False,
    ) -> None:
        self.tool_call = tool_call
        self.result = result
        self.skipped = skipped
        self.skip_reason = skip_reason
        self.needs_user_confirm = needs_user_confirm


class ToolCoordinator:
    """工具调用协调器。

    编排 S5、S8、S9 之间的完整调用链路。
    """

    def __init__(
        self,
        executor: ToolExecutor,
        registry: ToolRegistry,
        guardrails: GuardrailEngine,
        circuit_registry: CircuitBreakerRegistry,
        retry_policy: RetryPolicy,
        emitter: EventEmitter,
        fallback_registry: FallbackRegistry | None = None,
    ) -> None:
        self.executor = executor
        self.registry = registry
        self.guardrails = guardrails
        self.circuits = circuit_registry
        self.retry_policy = retry_policy
        self.emitter = emitter
        self.fallbacks = fallback_registry

    async def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        turn: int,
    ) -> list[ToolCallOutcome]:
        """按序执行工具调用列表。

        每个调用经过完整协调流程：护栏检查 → 熔断检查 → 执行 → 记录结果。

        Args:
            tool_calls: 工具调用列表。
            turn: 当前轮次号。

        Returns:
            每个调用的结果。
        """
        outcomes: list[ToolCallOutcome] = []
        for tc in tool_calls:
            outcome = await self.execute_single(tc, turn)
            outcomes.append(outcome)
        return outcomes

    async def execute_single(
        self,
        tool_call: ToolCall,
        turn: int,
    ) -> ToolCallOutcome:
        """执行单个工具调用的完整协调流程。"""
        name = tool_call.function.name
        raw_args = tool_call.function.arguments

        # 解析参数
        arguments = self.parse_arguments(raw_args)

        self.emitter.emit(
            "tool_call_start",
            turn=turn,
            data={
                "tool_name": name,
                "tool_call_id": tool_call.id,
                "arguments": arguments,
            },
        )

        # S9 降级：工具未注册时尝试查询降级替代
        if not self.registry.has_tool(name) and self.fallbacks is not None:
            fallback = self.fallbacks.get_fallback(name)
            if fallback and self.registry.has_tool(fallback):
                log.info("降级到替代工具", primary=name, fallback=fallback)
                name = fallback
                tool_call = ToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=tool_call.function.model_copy(update={"name": fallback}),
                )

        # Step 1: S8 护栏检查
        if self.registry.has_tool(name):
            meta = self.registry.get_metadata(name)
            verdict = await self.guardrails.check_tool_call(name, arguments, meta)

            if verdict.verdict in (VerdictType.DENY, VerdictType.BLOCK):
                return self.make_skipped(
                    tool_call, turn, f"护栏拒绝: {verdict.reason}", tripwire=verdict.tripwire
                )
            if verdict.verdict == VerdictType.CONFIRM:
                self.emitter.emit(
                    "tool_call_end",
                    turn=turn,
                    data={
                        "tool_name": name,
                        "tool_call_id": tool_call.id,
                        "needs_user_confirm": True,
                        "reason": verdict.reason,
                    },
                )
                return ToolCallOutcome(
                    tool_call=tool_call,
                    needs_user_confirm=True,
                    skip_reason=f"需要用户确认: {verdict.reason}",
                )
            if verdict.tripwire:
                return self.make_skipped(tool_call, turn, "绊线触发", tripwire=True)

        # Step 2: S9 熔断检查（OPEN 时尝试降级）
        circuit_state = self.circuits.check_circuit(name)
        if circuit_state == CircuitState.OPEN:
            if self.fallbacks is not None:
                fallback = self.fallbacks.get_fallback(name)
                if fallback and self.registry.has_tool(fallback):
                    fb_state = self.circuits.check_circuit(fallback)
                    if fb_state != CircuitState.OPEN:
                        log.info("熔断降级", primary=name, fallback=fallback)
                        name = fallback
                        tool_call = ToolCall(
                            id=tool_call.id,
                            type=tool_call.type,
                            function=tool_call.function.model_copy(update={"name": fallback}),
                        )
            if self.circuits.check_circuit(name) == CircuitState.OPEN:
                return self.make_skipped(tool_call, turn, f"熔断器断开: {name}")

        # Step 3: S5 工具执行
        result = await self.try_execute(name, arguments, tool_call.id, turn)

        # Step 4: S9 记录结果
        self.circuits.record_outcome(name, result.success)

        # Step 5: 失败处理
        if not result.success and result.error:
            result = await self.handle_failure(name, result, tool_call.id, turn)

        self.emitter.emit(
            "tool_call_end",
            turn=turn,
            data={
                "tool_name": name,
                "tool_call_id": tool_call.id,
                "success": result.success,
                "content": result.content,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
            },
        )

        return ToolCallOutcome(tool_call=tool_call, result=result)

    async def try_execute(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        turn: int,
    ) -> ToolResult:
        """尝试执行工具。"""
        try:
            return await self.executor.execute(name, arguments, tool_call_id)
        except Exception as exc:
            log.error("工具执行异常", tool_name=name, error=str(exc))
            return ToolResult(
                tool_call_id=tool_call_id,
                success=False,
                content="",
                error=str(exc),
                error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            )

    async def handle_failure(
        self,
        name: str,
        result: ToolResult,
        tool_call_id: str,
        turn: int,
    ) -> ToolResult:
        """处理工具执行失败。

        根据 S9 错误分类决定恢复策略。
        """
        message = result.error or "unknown error"
        classification: ErrorClassification | None = None
        if result.error_type:
            classification = classify_by_type_name(result.error_type, message)
        if classification is None:
            classification = classify_error(Exception(message))

        if classification.category == ErrorCategory.TRANSIENT:
            decision = self.retry_policy.get_retry_decision(name, 1)
            if decision.should_retry:
                log.info("重试工具调用", tool_name=name, delay=decision.wait_seconds)
                return result

        if classification.category == ErrorCategory.MODEL_RECOVERABLE:
            log.info("错误返回 LLM 自修正", tool_name=name)

        return result

    def make_skipped(
        self,
        tool_call: ToolCall,
        turn: int,
        reason: str,
        tripwire: bool = False,
    ) -> ToolCallOutcome:
        """创建跳过的结果。"""
        self.emitter.emit(
            "tool_call_end",
            turn=turn,
            data={
                "tool_name": tool_call.function.name,
                "tool_call_id": tool_call.id,
                "skipped": True,
                "reason": reason,
                "tripwire": tripwire,
            },
        )
        return ToolCallOutcome(
            tool_call=tool_call,
            skipped=True,
            skip_reason=reason,
        )

    @staticmethod
    def parse_arguments(raw: str) -> dict[str, Any]:
        """解析参数 JSON 字符串。"""
        if not raw:
            return {}
        result = repair_json(raw, return_objects=True)
        return result if isinstance(result, dict) else {}
