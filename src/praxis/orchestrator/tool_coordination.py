"""工具调用协调流程。

按序调用：
S8.check_tool_call → S9.check_circuit → S5.execute_tool →
S9.record_outcome → 失败时 S9.classify_error 决策。
"""

import asyncio
from collections.abc import Callable
from typing import Any

from json_repair import repair_json

from praxis.guardrails.engine import GuardrailEngine
from praxis.models.guardrails import VerdictType
from praxis.models.recovery import CircuitState, ErrorCategory, ErrorClassification
from praxis.models.session import SessionStatus
from praxis.models.telemetry import AuditEvent
from praxis.models.tools import ApprovalRequest, ToolCall, ToolResult
from praxis.orchestrator.events import EventEmitter
from praxis.protocols import ApprovalHandler, AuditSink
from praxis.recovery.circuit_breaker import CircuitBreakerRegistry
from praxis.recovery.classifier import classify_by_type_name, classify_error
from praxis.recovery.fallback import FallbackRegistry
from praxis.recovery.retry import RetryPolicy
from praxis.telemetry.audit import NullAuditSink
from praxis.telemetry.logger import get_logger
from praxis.tools.executor import ToolExecutor
from praxis.tools.registry import ToolRegistry

log = get_logger("orchestrator.tool_coordination")


class ToolCallOutcome:
    """单次工具调用结果。"""

    __slots__ = (
        "needs_user_confirm",
        "result",
        "skip_reason",
        "skipped",
        "tool_call",
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
        approval_handler: ApprovalHandler | None = None,
        approval_timeout: float = 60.0,
        audit_sink: AuditSink | None = None,
        session_id: str | None = None,
        status_callback: Callable[[SessionStatus], None] | None = None,
    ) -> None:
        self.executor = executor
        self.registry = registry
        self.guardrails = guardrails
        self.circuits = circuit_registry
        self.retry_policy = retry_policy
        self.emitter = emitter
        self.fallbacks = fallback_registry
        self.approval_handler = approval_handler
        self.approval_timeout = approval_timeout
        self.audit_sink = audit_sink or NullAuditSink()
        self.session_id = session_id
        self.status_callback = status_callback

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
                approved, reason = await self.request_approval(name, arguments)
                if not approved:
                    return self.make_skipped(
                        tool_call,
                        turn,
                        f"审批拒绝: {reason}",
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

        # Step 3~5: S5 执行 + S9 记录结果 + 瞬态错误按退避策略真正重试
        result = await self.execute_with_retry(name, arguments, tool_call.id, turn)

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

    async def request_approval(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, str]:
        """调用异步审批处理器；缺失、超时和异常均失败关闭。"""
        request = ApprovalRequest(
            session_id=self.session_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        approved = False
        reason = "未配置 ApprovalHandler"
        if self.approval_handler is None:
            pass
        elif self.status_callback is not None:
            self.status_callback(SessionStatus.WAITING_APPROVAL)
        if self.approval_handler is not None:
            try:
                decision = await asyncio.wait_for(
                    self.approval_handler.request_approval(request),
                    timeout=self.approval_timeout,
                )
            except TimeoutError:
                approved, reason = False, "审批处理超时"
            except Exception as exc:
                approved, reason = False, f"审批处理异常: {type(exc).__name__}"
            else:
                approved, reason = decision.approved, decision.reason
            finally:
                if self.status_callback is not None:
                    self.status_callback(SessionStatus.ACTIVE)
        await self.audit_sink.record(AuditEvent(
            event_type="permission_decision",
            component="tools",
            action="tool_approval",
            session_id=self.session_id,
            details={
                "tool_name": tool_name,
                "approved": approved,
                "reason": reason,
            },
        ))
        return approved, reason

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

    async def execute_with_retry(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        turn: int,
    ) -> ToolResult:
        """执行工具，并对瞬态错误按 S9 退避策略真正重试。

        每次尝试都向熔断器记录结果；仅 TRANSIENT 类错误重试，
        其余分类（含 MODEL_RECOVERABLE）直接返回交由 LLM 自修正。
        """
        attempt = 0
        while True:
            result = await self.try_execute(name, arguments, tool_call_id, turn)

            # S9 记录结果（每次实际执行都计入熔断器）
            self.circuits.record_outcome(name, result.success)

            if result.success or not result.error:
                self.retry_policy.reset(name)
                return result

            classification = self.classify_result(result)
            if classification.category != ErrorCategory.TRANSIENT:
                if classification.category == ErrorCategory.MODEL_RECOVERABLE:
                    log.info("错误返回 LLM 自修正", tool_name=name)
                self.retry_policy.reset(name)
                return result

            metadata = self.registry.get_metadata(name)
            if not (metadata.readonly or metadata.idempotent):
                log.info("非幂等写工具不自动重试", tool_name=name)
                self.retry_policy.reset(name)
                return result

            decision = self.retry_policy.get_retry_decision(name, attempt)
            if not decision.should_retry:
                log.info("瞬态错误重试已达上限", tool_name=name, attempts=attempt)
                self.retry_policy.reset(name)
                return result

            log.info(
                "重试工具调用",
                tool_name=name,
                attempt=attempt + 1,
                delay=decision.wait_seconds,
            )
            self.emitter.emit(
                "tool_retry",
                turn=turn,
                data={
                    "tool_name": name,
                    "tool_call_id": tool_call_id,
                    "attempt": attempt + 1,
                    "delay_seconds": decision.wait_seconds,
                    "error": result.error,
                },
            )
            self.retry_policy.record_attempt(name)
            if decision.wait_seconds > 0:
                await asyncio.sleep(decision.wait_seconds)
            attempt += 1

    @staticmethod
    def classify_result(result: ToolResult) -> ErrorClassification:
        """根据工具结果的错误信息推断 S9 错误分类。"""
        message = result.error or "unknown error"
        classification: ErrorClassification | None = None
        if result.error_type:
            classification = classify_by_type_name(result.error_type, message)
        if classification is None:
            classification = classify_error(Exception(message))
        return classification

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
