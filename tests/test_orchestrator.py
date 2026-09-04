"""S11 编排循环单元测试。"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from copy import deepcopy
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import ContextConfig, OrchestratorConfig
from praxis.context.assembler import PromptAssembler
from praxis.context.compaction import ContextCompactor
from praxis.models.context import RunContext, TokenUsage
from praxis.models.guardrails import GuardrailVerdict, VerdictType
from praxis.models.messages import (
    ImageContent,
    ImageUrl,
    ResolvedUserInput,
    TextContent,
)
from praxis.models.orchestrator import (
    AgentEvent,
    LoopPhase,
    LoopState,
    StrategyMode,
    TerminationReason,
)
from praxis.models.responses import ModelResponse, ModelResponseChunk, Usage
from praxis.models.tools import (
    ApprovalDecision,
    FunctionCall,
    ToolCall,
    ToolExecutionRecord,
    ToolExecutionState,
    ToolResult,
)
from praxis.orchestrator.events import EventEmitter, EventListener, StreamCollector
from praxis.orchestrator.loop import OrchestrationLoop
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


def resolved_image_input() -> ResolvedUserInput:
    return ResolvedUserInput(
        content=[
            TextContent(text="describe"),
            ImageContent(
                image_url=ImageUrl(url="data:image/png;base64,cG5n")
            ),
        ],
        text_projection="describe\n\n[image: image.png, image/png, 3 bytes]",
    )


def resolved_text_input(text: str) -> ResolvedUserInput:
    return ResolvedUserInput(content=text, text_projection=text)


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
        approval_handler: Any = None,
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
            approval_handler=approval_handler,
            approval_timeout=0.05,
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

    async def test_non_idempotent_execution_persists_each_boundary(self) -> None:
        coordinator, executor, emitter = self.make_coordinator()
        records: list[ToolExecutionRecord] = []

        async def observe(record: ToolExecutionRecord) -> None:
            records.append(record.model_copy(deep=True))

        coordinator.configure_execution_tracking({}, observe)
        await coordinator.execute_tool_calls([make_tool_call()], turn=1)

        assert [record.state for record in records] == [
            ToolExecutionState.PREPARED,
            ToolExecutionState.STARTED,
            ToolExecutionState.SUCCEEDED,
        ]

    async def test_uncertain_non_idempotent_execution_is_not_repeated(self) -> None:
        coordinator, executor, emitter = self.make_coordinator()
        tool_call = make_tool_call()
        digest = coordinator.argument_digest({"path": "test.py"})
        ledger = {
            tool_call.id: ToolExecutionRecord(
                tool_call_id=tool_call.id,
                tool_name=tool_call.function.name,
                argument_digest=digest,
                state=ToolExecutionState.UNCERTAIN,
                idempotent=False,
                readonly=False,
            ),
        }
        coordinator.configure_execution_tracking(ledger, None)

        outcome = await coordinator.execute_single(tool_call, turn=1)

        assert outcome.skipped is True
        assert "不确定" in outcome.skip_reason
        executor.execute.assert_not_awaited()

    async def test_guardrail_deny(self) -> None:
        verdict = GuardrailVerdict(verdict=VerdictType.DENY, reason="危险操作")
        coordinator, executor, emitter = self.make_coordinator(guardrail_verdict=verdict)
        tc = make_tool_call()
        outcomes = await coordinator.execute_tool_calls([tc], turn=1)
        assert outcomes[0].skipped is True
        assert "护栏拒绝" in outcomes[0].skip_reason

    async def test_guardrail_confirm_without_handler_denies(self) -> None:
        verdict = GuardrailVerdict(verdict=VerdictType.CONFIRM, reason="需确认")
        coordinator, executor, emitter = self.make_coordinator(guardrail_verdict=verdict)
        tc = make_tool_call()
        outcomes = await coordinator.execute_tool_calls([tc], turn=1)
        assert outcomes[0].skipped is True
        assert "未配置 ApprovalHandler" in outcomes[0].skip_reason

    async def test_guardrail_confirm_uses_async_handler(self) -> None:
        class Approver:
            async def request_approval(self, request: Any) -> ApprovalDecision:
                return ApprovalDecision(approved=True, reason="operator approved")

        verdict = GuardrailVerdict(verdict=VerdictType.CONFIRM, reason="需确认")
        coordinator, executor, emitter = self.make_coordinator(
            guardrail_verdict=verdict,
            approval_handler=Approver(),
        )
        outcomes = await coordinator.execute_tool_calls([make_tool_call()], turn=1)
        assert outcomes[0].result is not None
        executor.execute.assert_awaited_once()

    async def test_guardrail_confirm_handler_error_denies(self) -> None:
        class BrokenApprover:
            async def request_approval(self, request: Any) -> ApprovalDecision:
                raise RuntimeError("approval unavailable")

        verdict = GuardrailVerdict(verdict=VerdictType.CONFIRM, reason="需确认")
        coordinator, executor, emitter = self.make_coordinator(
            guardrail_verdict=verdict,
            approval_handler=BrokenApprover(),
        )
        outcomes = await coordinator.execute_tool_calls([make_tool_call()], turn=1)
        assert outcomes[0].skipped is True
        executor.execute.assert_not_awaited()

    async def test_circuit_open(self) -> None:
        coordinator, executor, emitter = self.make_coordinator()
        # 触发熔断
        for failure_index in range(6):  # noqa: B007 - public discard name
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
    ) -> OrchestrationLoop:
        """创建带 mock 依赖的循环引擎。"""
        from praxis.context.assembler import PromptAssembler
        from praxis.models.context import AssembledPrompt
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

    def make_history_loop(self) -> OrchestrationLoop:
        loop = self.make_loop()
        loop.assembler = PromptAssembler(ContextConfig())
        return loop

    async def test_safe_projection_is_the_only_input_for_text_consumers(self) -> None:
        loop = self.make_history_loop()
        memory = MagicMock()
        memory.append_message = MagicMock()
        memory.get_memory_index = AsyncMock(return_value=[])
        memory.search_memory = AsyncMock(return_value=[])
        loop.memory = memory

        skill_entry = MagicMock()
        skill_entry.name = "safe-skill"
        skill_entry.description = "safe"
        skill_manager = MagicMock()
        skill_manager.get_skill_index.return_value = [skill_entry]
        loop.skill_manager = skill_manager

        jit_retriever = MagicMock()
        jit_retriever.get_examples_for_task.return_value = []
        jit_retriever.get_identifier_index.return_value = []
        loop.jit_retriever = jit_retriever
        loop.strategy = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)

        resolved = resolved_image_input()
        ctx = loop.init_run(resolved)
        with patch.object(loop, "generate_plan", AsyncMock()) as generate_plan:
            early = await loop.prepare_run(ctx)

        safe_text = resolved.text_projection
        assert early is None
        loop.guardrails.check_input.assert_awaited_once_with(safe_text)
        memory_message = memory.append_message.call_args.args[0]
        assert memory_message.content == safe_text
        memory.search_memory.assert_awaited_once_with(safe_text, top_k=5)
        skill_manager.auto_activate_for_task.assert_called_once_with(safe_text)
        generate_plan.assert_awaited_once_with(safe_text)
        jit_retriever.get_examples_for_task.assert_called_once_with("general")
        consumer_calls = (
            loop.guardrails.check_input.call_args_list
            + memory.search_memory.call_args_list
            + skill_manager.auto_activate_for_task.call_args_list
            + generate_plan.call_args_list
            + jit_retriever.get_examples_for_task.call_args_list
        )
        assert "data:" not in str(consumer_calls)
        assert "cG5n" not in str(consumer_calls)

    @patch("praxis.orchestrator.loop.chat")
    async def test_tool_rounds_retain_structured_input_until_sanitized(
        self,
        mock_chat: Any,
    ) -> None:
        captured_messages: list[list[dict[str, Any]]] = []
        responses = [
            make_model_response(tool_calls=[make_tool_call()]),
            make_model_response(content="ok"),
        ]

        async def capture_chat(
            gateway: Any,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> ModelResponse:
            captured_messages.append(deepcopy(messages))
            return responses.pop(0)

        mock_chat.side_effect = capture_chat
        loop = self.make_history_loop()

        response = await loop.run(resolved_image_input())

        assert response.content == "ok"
        assert len(captured_messages) == 2
        for messages in captured_messages:
            user_messages = [item for item in messages if item["role"] == "user"]
            assert len(user_messages) == 1
            content = user_messages[0]["content"]
            assert isinstance(content, list)
            assert content[1]["type"] == "image_url"
        assert loop.assembler.conversation_history[0] == {
            "role": "user",
            "content": "describe\n\n[image: image.png, image/png, 3 bytes]",
        }
        assert "data:" not in str(loop.assembler.conversation_history)
        assert "cG5n" not in str(loop.assembler.conversation_history)

    @patch("praxis.orchestrator.loop.chat")
    async def test_gateway_error_sanitizes_structured_history(
        self,
        mock_chat: Any,
    ) -> None:
        mock_chat.side_effect = RuntimeError("provider unavailable")
        loop = self.make_history_loop()

        with pytest.raises(RuntimeError, match="provider unavailable"):
            await loop.run(resolved_image_input())

        assert loop.assembler.conversation_history == [{
            "role": "user",
            "content": "describe\n\n[image: image.png, image/png, 3 bytes]",
        }]

    @patch("praxis.orchestrator.loop.chat")
    async def test_tool_error_sanitizes_structured_history(
        self,
        mock_chat: Any,
    ) -> None:
        mock_chat.return_value = make_model_response(tool_calls=[make_tool_call()])
        loop = self.make_history_loop()
        loop.coordinator.execute_tool_calls.side_effect = RuntimeError("tool failed")

        with pytest.raises(RuntimeError, match="tool failed"):
            await loop.run(resolved_image_input())

        assert loop.assembler.conversation_history[0] == {
            "role": "user",
            "content": "describe\n\n[image: image.png, image/png, 3 bytes]",
        }
        assert "data:" not in str(loop.assembler.conversation_history)

    async def test_input_tripwire_never_persists_structured_history(self) -> None:
        loop = self.make_history_loop()
        loop.guardrails.check_input = AsyncMock(return_value=GuardrailVerdict(
            verdict=VerdictType.BLOCK,
            reason="blocked",
            tripwire=True,
        ))

        response = await loop.run(resolved_image_input())

        assert response.termination_reason is TerminationReason.TRIPWIRE
        assert loop.assembler.conversation_history == []

    async def test_stream_cancellation_sanitizes_structured_history(self) -> None:
        stream_started = asyncio.Event()

        async def blocking_stream(
            gateway: Any,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> Any:
            stream_started.set()
            await asyncio.Event().wait()
            yield ModelResponseChunk(id="unreachable")

        loop = self.make_history_loop()
        with patch("praxis.orchestrator.loop.chat_stream", new=blocking_stream):
            event_stream = loop.run_stream(resolved_image_input())
            assert (await anext(event_stream)).event_type == "turn_start"
            assert (await anext(event_stream)).event_type == "llm_request"
            pending = asyncio.create_task(anext(event_stream))
            await stream_started.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

        assert loop.assembler.conversation_history == [{
            "role": "user",
            "content": "describe\n\n[image: image.png, image/png, 3 bytes]",
        }]

    async def test_stream_generator_close_sanitizes_structured_history(self) -> None:
        loop = self.make_history_loop()
        delegated_closed = asyncio.Event()
        delegated_streams: list[AsyncGenerator[AgentEvent, None]] = []
        original_stream_run = loop.stream_run

        async def observe_delegated_stream(
            ctx: RunContext,
        ) -> AsyncGenerator[AgentEvent, None]:
            inner_stream = original_stream_run(ctx)
            try:
                async with aclosing(inner_stream):
                    async for event in inner_stream:
                        yield event
            finally:
                delegated_closed.set()

        def observed_stream_run(ctx: RunContext) -> AsyncGenerator[AgentEvent, None]:
            delegated = observe_delegated_stream(ctx)
            delegated_streams.append(delegated)
            return delegated

        event_stream = loop.run_stream(resolved_image_input())

        try:
            with patch.object(loop, "stream_run", observed_stream_run):
                assert (await anext(event_stream)).event_type == "turn_start"
                await event_stream.aclose()

            assert delegated_closed.is_set()
            assert loop.assembler.conversation_history == [{
                "role": "user",
                "content": "describe\n\n[image: image.png, image/png, 3 bytes]",
            }]
        finally:
            await event_stream.aclose()
            for delegated in delegated_streams:
                await delegated.aclose()

    @patch("praxis.orchestrator.loop.chat")
    async def test_max_turn_termination_sanitizes_structured_history(
        self,
        mock_chat: Any,
    ) -> None:
        mock_chat.return_value = make_model_response(tool_calls=[make_tool_call()])
        loop = self.make_history_loop()
        loop.termination.max_turns = 1

        response = await loop.run(resolved_image_input())

        assert response.termination_reason is TerminationReason.MAX_TURNS
        assert loop.assembler.conversation_history[0] == {
            "role": "user",
            "content": "describe\n\n[image: image.png, image/png, 3 bytes]",
        }
        assert "data:" not in str(loop.assembler.conversation_history)

    @patch("praxis.context.compaction.summarize", new_callable=AsyncMock)
    @patch("praxis.orchestrator.loop.chat")
    async def test_compaction_retains_active_attachment_and_refreshes_cleanup_index(
        self,
        mock_chat: Any,
        mock_summarize: AsyncMock,
    ) -> None:
        captured_messages: list[list[dict[str, Any]]] = []
        active_history_message: dict[str, Any] | None = None
        responses = [
            make_model_response(tool_calls=[make_tool_call()]),
            make_model_response(content="ok"),
        ]

        async def capture_chat(
            gateway: Any,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> ModelResponse:
            nonlocal active_history_message
            if active_history_message is None:
                active_history_message = loop.assembler.conversation_history[0]
            else:
                assert any(
                    message is active_history_message
                    for message in loop.assembler.conversation_history
                )
            captured_messages.append(deepcopy(messages))
            return responses.pop(0)

        mock_chat.side_effect = capture_chat
        mock_summarize.return_value = "compacted"
        loop = self.make_loop()
        context_config = ContextConfig(compaction_threshold=0.000001)
        loop.assembler = PromptAssembler(context_config)
        loop.compactor = ContextCompactor(context_config, loop.gateway)
        loop.compaction_min_history = 1

        response = await loop.run(resolved_image_input())

        assert response.content == "ok"
        mock_summarize.assert_awaited_once()
        second_user = [
            message
            for message in captured_messages[1]
            if message["role"] == "user"
        ]
        assert second_user[0]["content"][1]["type"] == "image_url"
        assert active_history_message is not None
        assert any(
            message is active_history_message
            for message in loop.assembler.conversation_history
        )
        assert active_history_message == {
            "role": "user",
            "content": "describe\n\n[image: image.png, image/png, 3 bytes]",
        }
        assert "data:" not in str(loop.assembler.conversation_history)

    @patch("praxis.orchestrator.loop.chat")
    async def test_copying_compactor_failure_sanitizes_live_structured_history(
        self,
        mock_chat: Any,
    ) -> None:
        class CopyingCompactor:
            async def compact(
                self,
                messages: list[dict[str, Any]],
                file_refs: list[str],
            ) -> None:
                copied_structured = [
                    deepcopy(message)
                    for message in messages
                    if message.get("role") == "user"
                    and isinstance(message.get("content"), list)
                ]
                messages.clear()
                messages.extend([
                    {"role": "system", "content": "compacted"},
                    *copied_structured,
                ])

        mock_chat.return_value = make_model_response(tool_calls=[make_tool_call()])
        loop = self.make_loop()
        loop.assembler = PromptAssembler(
            ContextConfig(compaction_threshold=0.000001)
        )
        loop.compactor = cast(Any, CopyingCompactor())
        loop.compaction_min_history = 1

        with pytest.raises(
            RuntimeError,
            match="context compaction discarded the active user input",
        ):
            await loop.run(resolved_image_input())

        assert loop.assembler.conversation_history == [
            {"role": "system", "content": "compacted"},
            {
                "role": "user",
                "content": "describe\n\n[image: image.png, image/png, 3 bytes]",
            },
        ]
        assert "data:" not in str(loop.assembler.conversation_history)
        assert "cG5n" not in str(loop.assembler.conversation_history)

    @patch("praxis.orchestrator.loop.chat")
    async def test_plan_and_execute_generates_plan(self, mock_chat: Any) -> None:
        """plan-and-execute 模式应在 prepare_run 生成并设置计划（否则退化为 ReAct）。"""
        from praxis.models.context import RunContext, TurnContext
        from praxis.models.orchestrator import StrategyMode
        from praxis.orchestrator.strategy import LoopStrategy

        loop = self.make_loop()
        loop.strategy = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)
        mock_chat.return_value = make_model_response(
            content='[{"description":"分析需求"},{"description":"编写代码"},{"description":"运行测试"}]'
        )
        ctx = RunContext(turn_context=TurnContext(
            user_content="实现功能X",
            user_text="实现功能X",
        ))
        early = await loop.prepare_run(ctx)
        assert early is None
        assert len(loop.strategy.plan) == 3
        assert loop.strategy.plan[0].description == "分析需求"
        assert any(e.event_type == "plan_created" for e in loop.emitter.events)

    @patch("praxis.orchestrator.loop.chat")
    async def test_plan_generation_failure_degrades_to_react(self, mock_chat: Any) -> None:
        """规划 LLM 调用失败时应退化为 ReAct（plan 留空），不中断。"""
        from praxis.models.context import RunContext, TurnContext
        from praxis.models.orchestrator import StrategyMode
        from praxis.orchestrator.strategy import LoopStrategy

        loop = self.make_loop()
        loop.strategy = LoopStrategy(StrategyMode.PLAN_AND_EXECUTE)
        mock_chat.side_effect = RuntimeError("LLM 不可用")
        ctx = RunContext(turn_context=TurnContext(
            user_content="任务",
            user_text="任务",
        ))
        early = await loop.prepare_run(ctx)
        assert early is None
        assert loop.strategy.plan == []

    async def test_gav_feedback_injected_on_verification_failure(self) -> None:
        """验证失败时，GAV 应把结构化反馈注入上下文并发 gav_feedback 事件。"""
        from praxis.models.verification import (
            VerificationResult,
            VerificationStatus,
            VerificationType,
        )

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

    async def test_gav_error_fails_closed_and_skip_does_not_hide_it(self) -> None:
        """验证基础设施 ERROR 必须失败关闭，不能被同时出现的 SKIP 隐藏。"""
        from praxis.models.verification import (
            VerificationResult,
            VerificationStatus,
            VerificationType,
        )

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
        assert any(
            "verification_feedback" in str(c.args)
            for c in loop.assembler.update_with_result.call_args_list[before:]
        )
        assert any(e.event_type == "gav_feedback" for e in loop.emitter.events)

    @patch("praxis.orchestrator.loop.chat")
    async def test_session_model_reaches_llm_call(self, mock_chat: Any) -> None:
        """会话选定的 model 必须传入 chat（此前恒用网关 default_model）。"""
        mock_chat.return_value = make_model_response(content="ok")
        loop = self.make_loop()
        loop.model = "fast-model"
        await loop.run(resolved_text_input("你好"))
        assert mock_chat.await_args.kwargs.get("model") == "fast-model"

    @patch("praxis.orchestrator.loop.chat")
    async def test_natural_termination(self, mock_chat: Any) -> None:
        mock_chat.return_value = make_model_response(content="最终回答")
        loop = self.make_loop()
        resp = await loop.run(resolved_text_input("你好"))
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
        resp = await loop.run(resolved_text_input("读取文件"))
        assert resp.content == "处理完成"
        assert resp.termination_reason == TerminationReason.NATURAL
        assert resp.total_turns == 2

    @patch("praxis.orchestrator.loop.chat")
    async def test_input_blocked(self, mock_chat: Any) -> None:
        loop = self.make_loop()
        loop.guardrails.check_input = AsyncMock(return_value=GuardrailVerdict(
            verdict=VerdictType.BLOCK, reason="恶意输入", tripwire=True
        ))
        resp = await loop.run(resolved_text_input("恶意消息"))
        assert resp.termination_reason == TerminationReason.TRIPWIRE
        assert "拒绝" in resp.content

    @patch("praxis.orchestrator.loop.chat")
    async def test_max_turns(self, mock_chat: Any) -> None:
        mock_chat.return_value = make_model_response(tool_calls=[make_tool_call()])
        loop = self.make_loop()
        loop.termination.max_turns = 2
        resp = await loop.run(resolved_text_input("无限循环"))
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
        resp = await loop.run(resolved_text_input("测试中断"))
        assert resp.termination_reason == TerminationReason.USER_ABORT

    @patch("praxis.orchestrator.loop.chat")
    async def test_events_emitted(self, mock_chat: Any) -> None:
        mock_chat.return_value = make_model_response(content="回复")
        loop = self.make_loop()
        resp = await loop.run(resolved_text_input("测试事件"))
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
        await loop.run(resolved_text_input("test"))
        state_after = loop.get_state()
        assert state_after.current_turn == 1
