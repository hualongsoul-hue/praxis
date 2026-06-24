"""S11 编排循环单元测试。"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import OrchestratorConfig
from praxis.models.context import TokenUsage
from praxis.models.guardrails import GuardrailVerdict, VerdictType
from praxis.models.orchestrator import (
    AgentEvent,
    LoopPhase,
    LoopState,
    StrategyMode,
    TerminationReason,
)
from praxis.models.responses import ModelResponse, Usage
from praxis.models.tools import FunctionCall, ToolCall, ToolResult
from praxis.orchestrator.events import EventEmitter, EventListener, StreamCollector
from praxis.orchestrator.parser import OutputParser, ParsedOutput
from praxis.orchestrator.strategy import LoopStrategy, PlanStep
from praxis.orchestrator.termination import TerminationManager
from praxis.orchestrator.tool_coordination import ToolCallOutcome


# ── 公共辅助 ──────────────────────────────────────────────────────────────


def make_model_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> ModelResponse:
    return ModelResponse(
        id="resp-1",
        content=content,
        tool_calls=tool_calls,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test-model",
        finish_reason=finish_reason,
        created=1700000000,
    )


def make_tool_call(
    name: str = "read_file",
    arguments: str = '{"path": "test.py"}',
    tc_id: str = "tc-1",
) -> ToolCall:
    return ToolCall(
        id=tc_id,
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


# ── Task 12.2: 输出解析 ─────────────────────────────────────────────────────


class TestOutputParser:
    """输出解析测试。"""

    def test_final_response(self) -> None:
        parser = OutputParser()
        resp = make_model_response(content="最终回复")
        parsed = parser.parse(resp)
        assert parsed.is_final is True
        assert parsed.content == "最终回复"
        assert parsed.tool_calls == []
        assert parsed.handoff_target is None

    def test_with_tool_calls(self) -> None:
        parser = OutputParser()
        tc = make_tool_call()
        resp = make_model_response(tool_calls=[tc])
        parsed = parser.parse(resp)
        assert parsed.is_final is False
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].function.name == "read_file"

    def test_multiple_tool_calls(self) -> None:
        parser = OutputParser()
        tc1 = make_tool_call("read_file", tc_id="tc-1")
        tc2 = make_tool_call("grep_search", '{"query": "test"}', tc_id="tc-2")
        resp = make_model_response(tool_calls=[tc1, tc2])
        parsed = parser.parse(resp)
        assert len(parsed.tool_calls) == 2

    def test_handoff_detection(self) -> None:
        parser = OutputParser()
        tc = make_tool_call("handoff_to_code_review", "{}", "tc-h")
        resp = make_model_response(tool_calls=[tc])
        parsed = parser.parse(resp)
        assert parsed.handoff_target == "code_review"

    def test_no_handoff(self) -> None:
        parser = OutputParser()
        tc = make_tool_call("read_file")
        resp = make_model_response(tool_calls=[tc])
        parsed = parser.parse(resp)
        assert parsed.handoff_target is None

    def test_parse_tool_arguments(self) -> None:
        args = OutputParser.parse_tool_arguments('{"path": "test.py", "line": 42}')
        assert args["path"] == "test.py"
        assert args["line"] == 42

    def test_parse_empty_arguments(self) -> None:
        assert OutputParser.parse_tool_arguments("") == {}

    def test_extract_schema_response(self) -> None:
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            value: int

        content = '{"name": "test", "value": 42}'
        result = OutputParser.extract_schema_response(content, TestSchema)
        assert result is not None
        assert result.name == "test"
        assert result.value == 42

    def test_extract_schema_empty(self) -> None:
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str

        assert OutputParser.extract_schema_response("", TestSchema) is None


# ── Task 12.3: 终止条件管理 ─────────────────────────────────────────────────


class TestTerminationManager:
    """终止条件管理测试。"""

    def test_tripwire_highest_priority(self) -> None:
        mgr = TerminationManager(OrchestratorConfig())
        state = LoopState(aborted=True, current_turn=200)
        reason = mgr.evaluate(state, tripwire=True, is_final_response=True)
        assert reason == TerminationReason.TRIPWIRE

    def test_user_abort(self) -> None:
        mgr = TerminationManager(OrchestratorConfig())
        state = LoopState(aborted=True)
        reason = mgr.evaluate(state)
        assert reason == TerminationReason.USER_ABORT

    def test_safety_refusal(self) -> None:
        mgr = TerminationManager(OrchestratorConfig())
        state = LoopState()
        reason = mgr.evaluate(state, safety_refusal=True)
        assert reason == TerminationReason.SAFETY_REFUSAL

    def test_natural_termination(self) -> None:
        mgr = TerminationManager(OrchestratorConfig())
        state = LoopState()
        reason = mgr.evaluate(state, is_final_response=True)
        assert reason == TerminationReason.NATURAL

    def test_max_turns(self) -> None:
        mgr = TerminationManager(OrchestratorConfig(max_turns=5))
        state = LoopState(current_turn=5)
        reason = mgr.evaluate(state)
        assert reason == TerminationReason.MAX_TURNS

    def test_token_exhausted(self) -> None:
        mgr = TerminationManager(OrchestratorConfig())
        state = LoopState()
        usage = TokenUsage(current_tokens=95000, max_tokens=100000, usage_ratio=0.95)
        reason = mgr.evaluate(state, token_usage=usage)
        assert reason == TerminationReason.TOKEN_EXHAUSTED

    def test_continue(self) -> None:
        mgr = TerminationManager(OrchestratorConfig())
        state = LoopState(current_turn=1)
        reason = mgr.evaluate(state)
        assert reason is None

    def test_priority_order(self) -> None:
        mgr = TerminationManager(OrchestratorConfig(max_turns=5))
        state = LoopState(aborted=True, current_turn=10)
        reason = mgr.evaluate(
            state,
            safety_refusal=True,
            is_final_response=True,
        )
        assert reason == TerminationReason.USER_ABORT


# ── Task 12.5: 循环策略选择 ─────────────────────────────────────────────────


class TestLoopStrategy:
    """循环策略选择测试。"""

    def test_default_react(self) -> None:
        s = LoopStrategy()
        assert s.mode == StrategyMode.REACT
        assert s.is_plan_complete() is True

    def test_switch_mode(self) -> None:
        s = LoopStrategy()
        s.switch_mode(StrategyMode.PLAN_AND_EXECUTE)
        assert s.mode == StrategyMode.PLAN_AND_EXECUTE

    def test_plan_and_execute(self) -> None:
        s = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)
        steps = [
            PlanStep("分析需求"),
            PlanStep("编写代码", tool_hint="write_file"),
            PlanStep("运行测试", tool_hint="run_command"),
        ]
        s.set_plan(steps)
        assert s.is_plan_complete() is False
        assert s.get_current_step() is not None
        assert s.get_current_step().description == "分析需求"

    def test_advance_step(self) -> None:
        s = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)
        s.set_plan([PlanStep("步骤1"), PlanStep("步骤2")])
        has_more = s.advance_step("完成")
        assert has_more is True
        assert s.get_current_step().description == "步骤2"
        has_more = s.advance_step("完成")
        assert has_more is False
        assert s.is_plan_complete() is True

    def test_plan_context(self) -> None:
        s = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)
        s.set_plan([PlanStep("步骤1"), PlanStep("步骤2")])
        ctx = s.get_plan_context()
        assert "步骤1" in ctx
        assert "步骤2" in ctx
        assert "<execution_plan>" in ctx

    def test_step_instruction(self) -> None:
        s = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)
        s.set_plan([PlanStep("编写代码", tool_hint="write_file")])
        inst = s.get_step_instruction()
        assert "编写代码" in inst
        assert "write_file" in inst

    def test_switch_to_react_clears_plan(self) -> None:
        s = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)
        s.set_plan([PlanStep("a")])
        s.switch_mode(StrategyMode.REACT)
        assert s.plan == []
        assert s.get_plan_context() == ""


# ── Task 12.6: 事件发射系统 ─────────────────────────────────────────────────


class TestEventEmitter:
    """事件发射系统测试。"""

    def test_emit_and_collect(self) -> None:
        emitter = EventEmitter()
        emitter.emit("turn_start", turn=1, data={"info": "test"})
        emitter.emit("llm_request", turn=1)
        events = emitter.get_events()
        assert len(events) == 2
        assert events[0].event_type == "turn_start"
        assert events[0].turn == 1
        assert events[0].data["info"] == "test"

    def test_listener(self) -> None:
        received: list[AgentEvent] = []

        class Collector(EventListener):
            def on_event(self, event: AgentEvent) -> None:
                received.append(event)

        emitter = EventEmitter()
        collector = Collector()
        emitter.add_listener(collector)
        emitter.emit("test_event")
        assert len(received) == 1
        emitter.remove_listener(collector)
        emitter.emit("test_event_2")
        assert len(received) == 1

    def test_clear(self) -> None:
        emitter = EventEmitter()
        emitter.emit("a")
        emitter.emit("b")
        emitter.clear()
        assert emitter.get_events() == []

    async def test_stream_collector(self) -> None:
        collector = StreamCollector()
        collector.on_event(AgentEvent(event_type="e1", turn=1))
        collector.on_event(AgentEvent(event_type="e2", turn=2))
        collector.close()

        events: list[AgentEvent] = []
        async for e in collector.iter_events():
            events.append(e)
        assert len(events) == 2
        assert events[0].event_type == "e1"


# ── Task 12.4: 工具调用协调 ─────────────────────────────────────────────────


class TestToolCoordination:
    """工具调用协调测试（集成级，使用 mock）。"""

    def make_coordinator(
        self,
        guardrail_verdict: GuardrailVerdict | None = None,
        execute_result: ToolResult | None = None,
    ) -> tuple:
        """创建带 mock 的协调器。"""
        executor = AsyncMock()
        if execute_result:
            executor.execute = AsyncMock(return_value=execute_result)
        else:
            executor.execute = AsyncMock(return_value=ToolResult(
                tool_call_id="tc-1", success=True, content="结果",
            ))

        registry = MagicMock()
        registry.has_tool.return_value = True
        from praxis.models.tools import ToolMetadata
        registry.get_metadata.return_value = ToolMetadata()

        guardrails = AsyncMock()
        if guardrail_verdict:
            guardrails.check_tool_call = AsyncMock(return_value=guardrail_verdict)
        else:
            guardrails.check_tool_call = AsyncMock(return_value=GuardrailVerdict(
                verdict=VerdictType.AUTO_APPROVE, reason="自动批准"
            ))

        from praxis.recovery.circuit_breaker import CircuitBreakerRegistry
        circuits = CircuitBreakerRegistry()

        from praxis.recovery.retry import RetryPolicy
        retry = RetryPolicy()

        emitter = EventEmitter()

        from praxis.orchestrator.tool_coordination import ToolCoordinator
        coordinator = ToolCoordinator(
            executor=executor,
            registry=registry,
            guardrails=guardrails,
            circuit_registry=circuits,
            retry_policy=retry,
            emitter=emitter,
        )
        return coordinator, executor, emitter

    async def test_successful_execution(self) -> None:
        coordinator, executor, emitter = self.make_coordinator()
        tc = make_tool_call()
        outcomes = await coordinator.execute_tool_calls([tc], turn=1)
        assert len(outcomes) == 1
        assert outcomes[0].result is not None
        assert outcomes[0].result.success is True
        assert not outcomes[0].skipped

    async def test_guardrail_deny(self) -> None:
        verdict = GuardrailVerdict(verdict=VerdictType.DENY, reason="危险操作")
        coordinator, executor, emitter = self.make_coordinator(guardrail_verdict=verdict)
        tc = make_tool_call()
        outcomes = await coordinator.execute_tool_calls([tc], turn=1)
        assert outcomes[0].skipped is True
        assert "护栏拒绝" in outcomes[0].skip_reason

    async def test_guardrail_confirm(self) -> None:
        verdict = GuardrailVerdict(verdict=VerdictType.CONFIRM, reason="需确认")
        coordinator, executor, emitter = self.make_coordinator(guardrail_verdict=verdict)
        tc = make_tool_call()
        outcomes = await coordinator.execute_tool_calls([tc], turn=1)
        assert outcomes[0].needs_user_confirm is True

    async def test_circuit_open(self) -> None:
        coordinator, executor, emitter = self.make_coordinator()
        # 触发熔断
        for _ in range(6):
            coordinator.circuits.record_outcome("read_file", success=False)
        tc = make_tool_call()
        outcomes = await coordinator.execute_tool_calls([tc], turn=1)
        assert outcomes[0].skipped is True
        assert "熔断器" in outcomes[0].skip_reason

    async def test_execution_failure(self) -> None:
        fail_result = ToolResult(
            tool_call_id="tc-1", success=False, content="", error="文件未找到"
        )
        coordinator, executor, emitter = self.make_coordinator(execute_result=fail_result)
        tc = make_tool_call()
        outcomes = await coordinator.execute_tool_calls([tc], turn=1)
        assert outcomes[0].result is not None
        assert outcomes[0].result.success is False

    async def test_events_emitted(self) -> None:
        coordinator, executor, emitter = self.make_coordinator()
        tc = make_tool_call()
        await coordinator.execute_tool_calls([tc], turn=1)
        events = emitter.get_events()
        types = [e.event_type for e in events]
        assert "tool_call_start" in types
        assert "tool_call_end" in types

    async def test_multiple_tool_calls(self) -> None:
        coordinator, executor, emitter = self.make_coordinator()
        tc1 = make_tool_call("read_file", tc_id="tc-1")
        tc2 = make_tool_call("grep_search", '{"q": "x"}', tc_id="tc-2")
        outcomes = await coordinator.execute_tool_calls([tc1, tc2], turn=1)
        assert len(outcomes) == 2


# ── Task 12.1: 核心循环引擎（集成测试） ─────────────────────────────────────


class TestOrchestrationLoop:
    """核心循环引擎测试（使用 mock 依赖）。"""

    def make_loop(
        self,
        responses: list[ModelResponse] | None = None,
    ) -> "OrchestrationLoop":
        """创建带 mock 依赖的循环引擎。"""
        from praxis.context.assembler import PromptAssembler
        from praxis.models.context import AssembledPrompt
        from praxis.orchestrator.loop import OrchestrationLoop

        config = OrchestratorConfig(max_turns=10)
        gateway = MagicMock()
        emitter = EventEmitter()

        # Mock assembler
        assembler = MagicMock(spec=PromptAssembler)
        assembler.assemble_prompt.return_value = AssembledPrompt(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            token_count=100,
            max_tokens=128000,
        )
        assembler.get_token_usage.return_value = TokenUsage(
            current_tokens=100, max_tokens=128000, usage_ratio=0.001
        )
        assembler.conversation_history = []

        # Mock coordinator
        coordinator = AsyncMock()
        coordinator.execute_tool_calls = AsyncMock(return_value=[
            ToolCallOutcome(
                tool_call=make_tool_call(),
                result=ToolResult(tool_call_id="tc-1", success=True, content="ok"),
            )
        ])

        # Mock guardrails
        guardrails = AsyncMock()
        guardrails.check_input = AsyncMock(return_value=GuardrailVerdict(
            verdict=VerdictType.PASS, reason="通过"
        ))
        guardrails.check_output = AsyncMock(return_value=GuardrailVerdict(
            verdict=VerdictType.PASS, reason="通过"
        ))

        termination = TerminationManager(config)
        strategy = LoopStrategy()
        parser = OutputParser()

        loop = OrchestrationLoop(
            config=config,
            gateway=gateway,
            assembler=assembler,
            tool_coordinator=coordinator,
            guardrails=guardrails,
            termination=termination,
            strategy=strategy,
            parser=parser,
            emitter=emitter,
        )
        return loop

    @patch("praxis.orchestrator.loop.chat")
    async def test_plan_and_execute_generates_plan(self, mock_chat: Any) -> None:
        """plan-and-execute 模式应在 prepare_run 生成并设置计划（否则退化为 ReAct）。"""
        from praxis.models.orchestrator import StrategyMode
        from praxis.orchestrator.strategy import LoopStrategy
        from praxis.models.context import RunContext, TurnContext

        loop = self.make_loop()
        loop.strategy = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)
        mock_chat.return_value = make_model_response(
            content='[{"description":"分析需求"},{"description":"编写代码"},{"description":"运行测试"}]'
        )
        ctx = RunContext(turn_context=TurnContext(user_message="实现功能X"))
        early = await loop.prepare_run("实现功能X", ctx)
        assert early is None
        assert len(loop.strategy.plan) == 3
        assert loop.strategy.plan[0].description == "分析需求"
        assert any(e.event_type == "plan_created" for e in loop.emitter.events)

    @patch("praxis.orchestrator.loop.chat")
    async def test_plan_generation_failure_degrades_to_react(self, mock_chat: Any) -> None:
        """规划 LLM 调用失败时应退化为 ReAct（plan 留空），不中断。"""
        from praxis.models.orchestrator import StrategyMode
        from praxis.orchestrator.strategy import LoopStrategy
        from praxis.models.context import RunContext, TurnContext

        loop = self.make_loop()
        loop.strategy = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)
        mock_chat.side_effect = RuntimeError("LLM 不可用")
        ctx = RunContext(turn_context=TurnContext(user_message="任务"))
        early = await loop.prepare_run("任务", ctx)
        assert early is None
        assert loop.strategy.plan == []

    async def test_gav_feedback_injected_on_verification_failure(self) -> None:
        """验证失败时，GAV 应把结构化反馈注入上下文并发 gav_feedback 事件。"""
        from praxis.models.verification import (
            VerificationResult, VerificationStatus, VerificationType,
        )
        from praxis.orchestrator.parser import ParsedOutput

        loop = self.make_loop()
        loop.verifier_registry = AsyncMock()
        loop.verifier_registry.run_computational = AsyncMock(return_value=[
            VerificationResult(
                status=VerificationStatus.FAIL,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name="lint_ruff",
                feedback="语法错误",
            )
        ])
        parsed = ParsedOutput(
            content="", tool_calls=[make_tool_call()], is_final=False, handoff_target=None,
        )
        await loop.process_tool_outcomes(parsed)

        # 最后一次 update_with_result 应注入验证反馈
        last_call = loop.assembler.update_with_result.call_args_list[-1]
        injected = last_call.args[0][0]["content"]
        assert "verification_feedback" in injected
        assert "语法错误" in injected
        assert any(e.event_type == "gav_feedback" for e in loop.emitter.events)

    async def test_gav_no_feedback_on_skip_or_error(self) -> None:
        """验证结果仅为 SKIP/ERROR（非 FAIL）时不应注入自我修正反馈。"""
        from praxis.models.verification import (
            VerificationResult, VerificationStatus, VerificationType,
        )
        from praxis.orchestrator.parser import ParsedOutput

        loop = self.make_loop()
        loop.verifier_registry = AsyncMock()
        loop.verifier_registry.run_computational = AsyncMock(return_value=[
            VerificationResult(
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name="broken", feedback="缺少参数",
            ),
            VerificationResult(
                status=VerificationStatus.SKIP,
                verification_type=VerificationType.COMPUTATIONAL,
                verifier_name="skipped",
            ),
        ])
        parsed = ParsedOutput(
            content="", tool_calls=[make_tool_call()], is_final=False, handoff_target=None,
        )
        before = loop.assembler.update_with_result.call_count
        await loop.process_tool_outcomes(parsed)
        # 只应有工具结果一次注入，无 verification_feedback
        assert not any(
            "verification_feedback" in str(c.args)
            for c in loop.assembler.update_with_result.call_args_list[before:]
        )
        assert not any(e.event_type == "gav_feedback" for e in loop.emitter.events)

    @patch("praxis.orchestrator.loop.chat")
    async def test_natural_termination(self, mock_chat: Any) -> None:
        mock_chat.return_value = make_model_response(content="最终回答")
        loop = self.make_loop()
        resp = await loop.run("你好")
        assert resp.content == "最终回答"
        assert resp.termination_reason == TerminationReason.NATURAL
        assert resp.total_turns == 1

    @patch("praxis.orchestrator.loop.chat")
    async def test_tool_call_then_final(self, mock_chat: Any) -> None:
        mock_chat.side_effect = [
            make_model_response(tool_calls=[make_tool_call()]),
            make_model_response(content="处理完成"),
        ]
        loop = self.make_loop()
        resp = await loop.run("读取文件")
        assert resp.content == "处理完成"
        assert resp.termination_reason == TerminationReason.NATURAL
        assert resp.total_turns == 2

    @patch("praxis.orchestrator.loop.chat")
    async def test_input_blocked(self, mock_chat: Any) -> None:
        loop = self.make_loop()
        loop.guardrails.check_input = AsyncMock(return_value=GuardrailVerdict(
            verdict=VerdictType.BLOCK, reason="恶意输入", tripwire=True
        ))
        resp = await loop.run("恶意消息")
        assert resp.termination_reason == TerminationReason.TRIPWIRE
        assert "拒绝" in resp.content

    @patch("praxis.orchestrator.loop.chat")
    async def test_max_turns(self, mock_chat: Any) -> None:
        mock_chat.return_value = make_model_response(tool_calls=[make_tool_call()])
        loop = self.make_loop()
        loop.termination.max_turns = 2
        resp = await loop.run("无限循环")
        assert resp.termination_reason == TerminationReason.MAX_TURNS

    @patch("praxis.orchestrator.loop.chat")
    async def test_abort(self, mock_chat: Any) -> None:
        call_count = 0

        async def side_effect(*args: Any, **kwargs: Any) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                loop.abort()
                return make_model_response(tool_calls=[make_tool_call()])
            return make_model_response(content="不应到达")

        mock_chat.side_effect = side_effect
        loop = self.make_loop()
        resp = await loop.run("测试中断")
        assert resp.termination_reason == TerminationReason.USER_ABORT

    @patch("praxis.orchestrator.loop.chat")
    async def test_events_emitted(self, mock_chat: Any) -> None:
        mock_chat.return_value = make_model_response(content="回复")
        loop = self.make_loop()
        resp = await loop.run("测试事件")
        event_types = [e.event_type for e in resp.events]
        assert "turn_start" in event_types
        assert "llm_request" in event_types
        assert "llm_response" in event_types
        assert "termination" in event_types

    @patch("praxis.orchestrator.loop.chat")
    async def test_get_state(self, mock_chat: Any) -> None:
        mock_chat.return_value = make_model_response(content="done")
        loop = self.make_loop()
        state_before = loop.get_state()
        assert state_before.phase == LoopPhase.IDLE
        await loop.run("test")
        state_after = loop.get_state()
        assert state_after.current_turn == 1
