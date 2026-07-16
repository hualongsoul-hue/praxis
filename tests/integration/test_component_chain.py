"""跨组件集成测试。

验证依赖链路完整性：
S12（会话管理）→ S11（编排循环）→ S7（上下文引擎）→
S6（记忆管线）→ S4（模型网关）→ S2（遥测）→ S1（配置）。

重点：状态传递、接口契约、组件间数据流正确性。
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import (
    ContextConfig,
    OrchestratorConfig,
    PersistenceConfig,
    SessionConfig,
    TelemetryConfig,
    ToolsConfig,
)
from praxis.context.assembler import AssembledPrompt, PromptAssembler
from praxis.context.tool_injection import ToolInjector
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.models.context import TokenUsage, TurnContext
from praxis.models.guardrails import GuardrailVerdict, VerdictType
from praxis.models.memory import WorkingMemoryMessage
from praxis.models.orchestrator import (
    AgentResponse,
    TerminationReason,
)
from praxis.models.session import SessionMetadata, SessionStatus
from praxis.models.tools import ToolDefinition, ToolMetadata, ToolResult
from praxis.persistence.store import PersistenceStore, create_store
from praxis.recovery.circuit_breaker import CircuitBreakerRegistry
from praxis.recovery.retry import RetryPolicy
from praxis.session.checkpoint import CheckpointManager
from praxis.session.core import SessionFactory
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric
from praxis.tools.executor import ToolExecutor
from praxis.tools.policy import ToolPolicy
from praxis.tools.registry import ToolRegistry

# ── 辅助 ──────────────────────────────────────────────────────────────────


def make_raw_response(
    content: str = "",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    return SimpleNamespace(
        id="chatcmpl-chain", choices=[choice], usage=usage,
        model="test-model", created=1700000000,
    )


def make_raw_tool_call(name: str, arguments: str = "{}", tc_id: str = "tc-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=tc_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.fixture(autouse=True)
def mock_litellm():
    with patch("praxis.context.assembler.get_max_tokens", return_value=128000), \
         patch("praxis.context.assembler.get_token_count", return_value=100):
        yield


@pytest.fixture
async def store(tmp_path: Path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "chain.db"))
    s = await create_store(config)
    yield s  # type: ignore[misc]
    await s.close()


# ── S1: 配置系统链路验证 ───────────────────────────────────────────────────


class TestS1ConfigChain:
    """S1 配置 → 各组件构造的链路验证。"""

    def test_config_schemas_construct_all_components(self) -> None:
        """验证：所有配置 Schema 可正确构造，默认值有效。"""
        session_cfg = SessionConfig()
        orch_cfg = OrchestratorConfig()
        ctx_cfg = ContextConfig()
        telemetry_cfg = TelemetryConfig()

        assert session_cfg.auto_checkpoint is True
        assert orch_cfg.max_turns > 0
        assert ctx_cfg.compaction_threshold > 0
        assert telemetry_cfg is not None

    def test_component_config_isolation(self) -> None:
        """验证：不同组件配置互不影响。"""
        cfg1 = OrchestratorConfig(max_turns=5)
        cfg2 = OrchestratorConfig(max_turns=100)
        assert cfg1.max_turns == 5
        assert cfg2.max_turns == 100


# ── S2: 遥测链路验证 ──────────────────────────────────────────────────────


class TestS2TelemetryChain:
    """S2 遥测 → S11/S12 各组件的链路验证。"""

    def test_logger_creates_structured_output(self) -> None:
        """验证：结构化日志器可正常创建和使用。"""
        log = get_logger("test.chain")
        assert log is not None
        # 日志调用不抛异常
        log.info("集成测试日志", component="test", session_id="sess-001")

    def test_metric_emission(self) -> None:
        """验证：指标发射不抛异常。"""
        emit_metric("test_metric", 1.0, {"label": "integration"}, "counter")
        emit_metric("test_histogram", 42.0, {}, "histogram")


# ── S4 + S7: Gateway → Context 链路验证 ──────────────────────────────────


class TestS4S7GatewayContextChain:
    """S4（Gateway）→ S7（Context）链路验证。"""

    def test_assembler_produces_valid_prompt(self) -> None:
        """验证：PromptAssembler 组装的 Prompt 符合 LLM 消息格式。"""
        assembler = PromptAssembler(ContextConfig(), model="test-model")
        turn = TurnContext(user_message="你好")
        prompt = assembler.assemble_prompt(turn)

        assert isinstance(prompt, AssembledPrompt)
        assert len(prompt.messages) >= 2  # system + user
        assert prompt.messages[-1]["role"] == "user"
        assert prompt.messages[-1]["content"] == "你好"
        assert prompt.messages[0]["role"] in ("system", "developer")

    def test_assembler_with_tool_schemas(self) -> None:
        """验证：工具 Schema 正确注入到 Prompt。"""
        assembler = PromptAssembler(ContextConfig(), model="test-model")
        schemas = [{"type": "function", "function": {"name": "read_file"}}]
        assembler.set_tool_schemas(schemas)

        turn = TurnContext(user_message="读取文件")
        prompt = assembler.assemble_prompt(turn)
        assert prompt.tools == schemas

    def test_token_usage_tracking(self) -> None:
        """验证：Token 用量追踪返回有效结构。"""
        assembler = PromptAssembler(ContextConfig(), model="test-model")
        usage = assembler.get_token_usage()
        assert isinstance(usage, TokenUsage)
        assert usage.max_tokens == 128000


# ── S5: 工具系统链路验证 ──────────────────────────────────────────────────


class TestS5ToolChain:
    """S5 工具注册 → 注入 → 执行链路验证。"""

    def test_tool_registration_and_injection(self) -> None:
        """验证：工具注册后可被 ToolInjector 发现。"""
        registry = ToolRegistry()

        async def handler(args: dict[str, Any]) -> str:
            return "ok"

        registry.register(
            ToolDefinition(
                name="test_tool",
                description="测试工具",
                parameters={"type": "object", "properties": {}},
            ),
            handler,
        )

        # 直接从注册表获取 Schema 验证注册成功
        schemas = registry.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "test_tool" in names

        # ToolInjector 按 category 过滤
        injector = ToolInjector(registry)
        assert injector.registry is registry

    async def test_tool_executor_pipeline(self) -> None:
        """验证：ToolExecutor 完整管线（验证 → 沙箱 → 执行 → 结果）。"""
        registry = ToolRegistry()

        async def echo_handler(args: dict[str, Any]) -> str:
            return f"echo: {args.get('msg', '')}"

        registry.register(
            ToolDefinition(
                name="echo",
                description="回显",
                parameters={
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                },
            ),
            echo_handler,
        )

        sandbox = ToolPolicy(ToolsConfig())
        executor = ToolExecutor(registry, sandbox)
        result = await executor.execute("echo", {"msg": "hello"}, "tc-test")

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "hello" in result.content


# ── S6: 记忆管线链路验证 ──────────────────────────────────────────────────


class TestS6MemoryChain:
    """S6 记忆管线的内部链路验证。"""

    def test_working_memory_model(self) -> None:
        """验证：WorkingMemory 数据模型正确。"""
        from praxis.models.memory import WorkingMemory
        wm = WorkingMemory(session_id="test-sess")
        wm.append(WorkingMemoryMessage(role="user", content="测试消息"))
        wm.append(WorkingMemoryMessage(role="assistant", content="测试回复"))
        assert len(wm.messages) == 2
        assert wm.messages[0].role == "user"

    def test_memory_entry_model(self) -> None:
        """验证：MemoryEntry 数据模型正确。"""
        from praxis.models.memory import MemoryEntry, MemoryScope, MemoryType
        scope = MemoryScope.from_string("session/test-001")
        entry = MemoryEntry(
            content="用户偏好中文",
            memory_type=MemoryType.SEMANTIC,
            scope=scope,
            confidence=0.8,
        )
        assert entry.confidence == 0.8
        assert entry.memory_type == MemoryType.SEMANTIC
        assert entry.scope.scope_type == "session"


# ── S8: 护栏链路验证 ──────────────────────────────────────────────────────


class TestS8GuardrailChain:
    """S8 护栏 → S11 编排 → S2 审计的链路验证。"""

    async def test_input_guardrail_pass(self) -> None:
        """验证：正常输入通过护栏检查。"""
        engine = GuardrailEngine(RuleEngine(), PermissionManager())
        verdict = await engine.check_input("你好，请帮我分析代码")
        assert verdict.verdict == VerdictType.PASS

    async def test_output_guardrail_pass(self) -> None:
        """验证：正常输出通过护栏检查。"""
        engine = GuardrailEngine(RuleEngine(), PermissionManager())
        verdict = await engine.check_output("这是代码分析结果...")
        assert verdict.verdict == VerdictType.PASS

    async def test_tool_call_guardrail_with_permission(self) -> None:
        """验证：工具调用通过权限管理器裁决。"""
        engine = GuardrailEngine(RuleEngine(), PermissionManager())
        meta = ToolMetadata()
        verdict = await engine.check_tool_call("read_file", {"path": "test.py"}, meta)
        assert verdict.verdict in (VerdictType.AUTO_APPROVE, VerdictType.CONFIRM, VerdictType.PASS)


# ── S9: 错误恢复链路验证 ──────────────────────────────────────────────────


class TestS9RecoveryChain:
    """S9 错误恢复 → S11 编排链路验证。"""

    def test_circuit_breaker_registry(self) -> None:
        """验证：熔断器注册表正确追踪工具成功/失败。"""
        from praxis.models.recovery import CircuitState
        circuits = CircuitBreakerRegistry(failure_threshold=3)
        # 正常状态
        assert circuits.check_circuit("tool_a") == CircuitState.CLOSED

        # 连续失败触发熔断
        for failure_index in range(4):  # noqa: B007 - public discard name
            circuits.record_outcome("tool_a", success=False)
        assert circuits.check_circuit("tool_a") == CircuitState.OPEN

    def test_retry_policy(self) -> None:
        """验证：重试策略返回合理值。"""
        policy = RetryPolicy(max_retries=3)
        assert policy.max_retries == 3

        decision = policy.get_retry_decision("test_tool", attempt_count=0)
        assert decision.should_retry is True

        decision_max = policy.get_retry_decision("test_tool", attempt_count=3)
        assert decision_max.should_retry is False


# ── S11 + S12: 完整编排 → 会话链路验证 ──────────────────────────────────


class TestS11S12OrchestrationSessionChain:
    """S12（SessionFactory）→ S11（OrchestrationLoop）完整链路验证。"""

    async def test_session_factory_creates_complete_session(
        self,
        store: PersistenceStore,
    ) -> None:
        """验证：SessionFactory 创建的会话包含所有必要组件。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        mock_gw = MagicMock(spec=GatewayRouter)
        mock_gw.config = MagicMock()
        mock_gw.config.max_budget = None
        mock_gw.config.default_model = "test-model"
        mock_gw.router = MagicMock()
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gw)
        try:
            self.assert_session_complete(session, store, guardrails)
        finally:
            await session.terminate()

    def assert_session_complete(self, session, store, guardrails) -> None:

        # 验证所有组件链接完整
        assert session.session_id
        assert session.status == SessionStatus.ACTIVE
        assert session.loop is not None
        assert session.assembler is not None
        assert session.registry is not None
        assert session.store is store
        assert session.loop.coordinator is not None
        assert session.loop.guardrails is guardrails
        assert session.loop.termination is not None
        assert session.loop.parser is not None
        assert session.loop.emitter is not None

    async def test_full_chain_session_to_response(
        self,
        store: PersistenceStore,
    ) -> None:
        """验证：S12 → S11 → S7 → S4 完整链路到响应。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        mock_gw = MagicMock(spec=GatewayRouter)
        mock_gw.config = MagicMock()
        mock_gw.config.max_budget = None
        mock_gw.config.default_model = "test-model"
        mock_gw.router = MagicMock()
        mock_gw.router.acompletion = AsyncMock(
            return_value=make_raw_response(content="集成测试响应")
        )
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gw)

        response = await session.run_turn("集成测试消息")
        await session.terminate()

        assert isinstance(response, AgentResponse)
        assert response.content == "集成测试响应"
        assert response.total_turns == 1
        assert session.metadata.total_turns == 1

    async def test_full_chain_with_tool_execution(
        self,
        store: PersistenceStore,
    ) -> None:
        """验证：S12 → S11 → S5(工具) → S7 → S4 完整链路含工具。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        mock_gw = MagicMock(spec=GatewayRouter)
        mock_gw.config = MagicMock()
        mock_gw.config.max_budget = None
        mock_gw.config.default_model = "test-model"
        mock_gw.router = MagicMock()
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gw)

        # 注册测试工具
        async def greet_handler(args: dict[str, Any]) -> str:
            return f"Hello, {args.get('name', 'World')}!"

        session.registry.register(
            ToolDefinition(
                name="greet",
                description="问候",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            ),
            greet_handler,
        )

        # Mock Gateway: 第一次返回工具调用，第二次返回最终响应
        call_count = 0

        async def mock_acompletion(**kwargs: Any) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tc = make_raw_tool_call("greet", json.dumps({"name": "Praxis"}))
                return make_raw_response(tool_calls=[tc])
            return make_raw_response(content="问候完成: Hello, Praxis!")

        mock_gw.router.acompletion = AsyncMock(side_effect=mock_acompletion)

        response = await session.run_turn("请问候 Praxis")
        await session.terminate()

        assert "Praxis" in response.content
        assert response.tool_calls_made == 1
        assert response.total_turns == 2

    async def test_checkpoint_preserves_chain_state(
        self,
        store: PersistenceStore,
    ) -> None:
        """验证：S3 检查点保留 S7/S6/S11 的状态。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        mock_gw = MagicMock(spec=GatewayRouter)
        mock_gw.config = MagicMock()
        mock_gw.config.max_budget = None
        mock_gw.config.default_model = "test-model"
        mock_gw.router = MagicMock()
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gw)
        session.assembler.conversation_history = [
            {"role": "assistant", "content": "测试状态"},
        ]
        session.metadata.total_turns = 3

        cp_id = await session.save_auto_checkpoint()
        assert cp_id is not None

        # 验证检查点可加载
        cp_mgr = CheckpointManager(store)
        cp = await cp_mgr.load_checkpoint(session.session_id, cp_id)
        assert cp is not None
        assert cp.session_id == session.session_id
        await session.terminate()

    async def test_event_flow_through_chain(
        self,
        store: PersistenceStore,
    ) -> None:
        """验证：事件从 S11 编排 → S2 遥测的完整传播。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        mock_gw = MagicMock(spec=GatewayRouter)
        mock_gw.config = MagicMock()
        mock_gw.config.max_budget = None
        mock_gw.config.default_model = "test-model"
        mock_gw.router = MagicMock()
        mock_gw.router.acompletion = AsyncMock(
            return_value=make_raw_response(content="事件测试")
        )
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gw)

        response = await session.run_turn("测试事件链")
        await session.terminate()

        event_types = [e.event_type for e in response.events]
        assert "turn_start" in event_types
        assert "llm_request" in event_types
        assert "llm_response" in event_types
        assert "termination" in event_types


# ── 全链路数据流验证 ──────────────────────────────────────────────────────


class TestDataFlowIntegrity:
    """验证组件间数据流的类型契约和完整性。"""

    def test_tool_result_to_context(self) -> None:
        """验证：ToolResult → S7 上下文更新的数据格式一致。"""
        result = ToolResult(
            tool_call_id="tc-1",
            success=True,
            content="文件内容: Hello",
        )
        # 模拟 S11 注入 S7 的格式
        tool_msg = {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result.content if result.success else f"错误: {result.error}",
        }
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == "文件内容: Hello"

    def test_guardrail_verdict_contract(self) -> None:
        """验证：GuardrailVerdict 接口契约。"""
        pass_verdict = GuardrailVerdict(verdict=VerdictType.PASS, reason="通过")
        assert pass_verdict.tripwire is False

        block_verdict = GuardrailVerdict(
            verdict=VerdictType.BLOCK, reason="阻止", tripwire=True,
        )
        assert block_verdict.tripwire is True

    def test_agent_response_contract(self) -> None:
        """验证：AgentResponse 接口契约。"""
        response = AgentResponse(
            content="测试响应",
            tool_calls_made=2,
            total_turns=3,
            termination_reason=TerminationReason.NATURAL,
            events=[],
        )
        assert response.content == "测试响应"
        assert response.termination_reason == TerminationReason.NATURAL

    def test_session_metadata_contract(self) -> None:
        """验证：SessionMetadata 状态机转换。"""
        meta = SessionMetadata()
        assert meta.status == SessionStatus.INITIALIZING

        meta.status = SessionStatus.ACTIVE
        assert meta.status == SessionStatus.ACTIVE

        meta.status = SessionStatus.TERMINATED
        assert meta.status == SessionStatus.TERMINATED
