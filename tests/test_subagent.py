"""S13 子代理协调单元测试。"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import (
    ContextConfig,
    SessionConfig,
    OrchestratorConfig,
    PersistenceConfig,
    SubagentConfig,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.models.orchestrator import AgentResponse, TerminationReason
from praxis.models.subagent import (
    ConflictMarker,
    SubagentMode,
    SubagentResult,
    SubagentSpec,
    SubagentStatus,
)
from praxis.persistence.store import PersistenceStore, create_store
from praxis.subagent.aggregation import ResultAggregator
from praxis.subagent.fork import ForkManager
from praxis.subagent.handoff import HandoffManager
from praxis.subagent.isolation import IsolatedContext
from praxis.subagent.resource_control import ResourceController
from praxis.subagent.spawn import SubagentSpawner
from praxis.tools.registry import ToolRegistry


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
async def store(tmp_path: Path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "test.db"))
    s = await create_store(config)
    yield s  # type: ignore[misc]
    await s.close()


@pytest.fixture
def guardrails() -> GuardrailEngine:
    return GuardrailEngine(RuleEngine(), PermissionManager())


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    # 注册一个测试工具
    from praxis.models.tools import ToolDefinition, ToolMetadata
    defn = ToolDefinition(
        name="test_tool",
        description="测试工具",
        parameters={"type": "object", "properties": {}},
        metadata=ToolMetadata(),
    )
    reg.register(defn, AsyncMock(return_value="ok"))
    return reg


@pytest.fixture
def subagent_config() -> SubagentConfig:
    return SubagentConfig(max_concurrent=3, default_max_turns=10, default_timeout=5.0)


@pytest.fixture
def isolation(store: PersistenceStore) -> IsolatedContext:
    return IsolatedContext(
        store=store,
        orchestrator_config=OrchestratorConfig(max_turns=10),
        context_config=ContextConfig(),
    )


@pytest.fixture
def resource_ctrl(subagent_config: SubagentConfig) -> ResourceController:
    return ResourceController(subagent_config)


# ── Task 14.5: 资源管控 ─────────────────────────────────────────────────────


class TestResourceController:
    """资源管控测试。"""

    async def test_acquire_and_release(self, resource_ctrl: ResourceController) -> None:
        ok = await resource_ctrl.acquire("sub-1")
        assert ok is True
        assert resource_ctrl.available_slots == 2
        resource_ctrl.release("sub-1")
        assert resource_ctrl.available_slots == 3

    async def test_active_count(self, resource_ctrl: ResourceController) -> None:
        await resource_ctrl.acquire("sub-1")
        await resource_ctrl.acquire("sub-2")
        resource_ctrl.register_task("sub-1", MagicMock())
        resource_ctrl.register_task("sub-2", MagicMock())
        assert resource_ctrl.active_count == 2
        resource_ctrl.release("sub-1")
        assert resource_ctrl.active_count == 1

    async def test_cancel_task(self, resource_ctrl: ResourceController) -> None:
        mock_task = MagicMock()
        resource_ctrl.register_task("sub-1", mock_task)
        assert resource_ctrl.cancel_task("sub-1") is True
        mock_task.cancel.assert_called_once()

    async def test_cancel_nonexistent(self, resource_ctrl: ResourceController) -> None:
        assert resource_ctrl.cancel_task("no-exist") is False

    async def test_semaphore_limit(self) -> None:
        ctrl = ResourceController(SubagentConfig(max_concurrent=2))
        await ctrl.acquire("a")
        await ctrl.acquire("b")
        # 第三个应该会阻塞，用 wait_for 验证
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ctrl.acquire("c"), timeout=0.1)


# ── Task 14.4: 上下文隔离 ───────────────────────────────────────────────────


class TestIsolatedContext:
    """上下文隔离测试。"""

    def test_create_isolated_session(
        self,
        isolation: IsolatedContext,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        spec = SubagentSpec(
            task="测试任务",
            tool_names=["test_tool"],
        )
        session = isolation.create_isolated_session(
            spec=spec,
            guardrails=guardrails,
            parent_registry=registry,
        )
        assert session is not None
        assert session.session_id
        # 子代理应该有工具
        assert session.registry.has_tool("test_tool")

    def test_tool_subset_isolation(
        self,
        isolation: IsolatedContext,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        # 注册第二个工具
        from praxis.models.tools import ToolDefinition, ToolMetadata
        defn = ToolDefinition(
            name="extra_tool",
            description="额外工具",
            parameters={"type": "object", "properties": {}},
            metadata=ToolMetadata(),
        )
        registry.register(defn, AsyncMock(return_value="ok"))

        # 子代理只请求 test_tool
        spec = SubagentSpec(task="test", tool_names=["test_tool"])
        session = isolation.create_isolated_session(spec, guardrails, registry)
        assert session.registry.has_tool("test_tool")
        assert not session.registry.has_tool("extra_tool")

    def test_empty_tools(
        self,
        isolation: IsolatedContext,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        spec = SubagentSpec(task="no tools")
        session = isolation.create_isolated_session(spec, guardrails, registry)
        assert len(session.registry.list_tools()) == 0


# ── Task 14.4: 结果聚合 ─────────────────────────────────────────────────────


class TestResultAggregator:
    """结果聚合测试。"""

    def test_aggregate_basic(self) -> None:
        agg = ResultAggregator()
        results = [
            SubagentResult(
                subagent_id="a",
                mode=SubagentMode.FORK,
                status=SubagentStatus.COMPLETED,
                summary="结果A",
                key_findings=["发现1"],
                total_tokens=100,
                total_turns=3,
            ),
            SubagentResult(
                subagent_id="b",
                mode=SubagentMode.FORK,
                status=SubagentStatus.COMPLETED,
                summary="结果B",
                key_findings=["发现2", "发现3"],
                total_tokens=200,
                total_turns=5,
            ),
        ]
        combined = agg.aggregate(results)
        assert "结果A" in combined["combined_summary"]
        assert "结果B" in combined["combined_summary"]
        assert len(combined["key_findings"]) == 3
        assert combined["total_tokens"] == 300
        assert combined["total_turns"] == 8
        assert combined["subagent_count"] == 2

    def test_detect_conflicts(self) -> None:
        agg = ResultAggregator()
        results = [
            SubagentResult(
                subagent_id="a",
                mode=SubagentMode.FORK,
                status=SubagentStatus.COMPLETED,
                metadata={"answer_lang": "Python"},
            ),
            SubagentResult(
                subagent_id="b",
                mode=SubagentMode.FORK,
                status=SubagentStatus.COMPLETED,
                metadata={"answer_lang": "Rust"},
            ),
        ]
        conflicts = agg.detect_conflicts(results)
        assert len(conflicts) == 1
        assert conflicts[0].field == "answer_lang"
        assert set(conflicts[0].values) == {"Python", "Rust"}

    def test_no_conflicts(self) -> None:
        agg = ResultAggregator()
        results = [
            SubagentResult(
                subagent_id="a",
                mode=SubagentMode.FORK,
                status=SubagentStatus.COMPLETED,
                metadata={"answer_lang": "Python"},
            ),
            SubagentResult(
                subagent_id="b",
                mode=SubagentMode.FORK,
                status=SubagentStatus.COMPLETED,
                metadata={"answer_lang": "Python"},
            ),
        ]
        conflicts = agg.detect_conflicts(results)
        assert len(conflicts) == 0

    def test_empty_results(self) -> None:
        agg = ResultAggregator()
        combined = agg.aggregate([])
        assert combined["subagent_count"] == 0


# ── Task 14.1: Agent-as-Tool ────────────────────────────────────────────────


class TestSubagentSpawner:
    """Agent-as-Tool 测试。"""

    @patch("praxis.session.core.Session.run_turn")
    async def test_spawn_success(
        self,
        mock_run_turn: Any,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        mock_run_turn.return_value = AgentResponse(
            content="子任务完成",
            total_turns=3,
            termination_reason=TerminationReason.NATURAL,
            events=[],
        )

        spawner = SubagentSpawner(
            isolation=isolation,
            resource_ctrl=resource_ctrl,
            guardrails=guardrails,
            parent_registry=registry,
        )
        result = await spawner.spawn_agent_as_tool(
            task="分析代码",
            tool_names=["test_tool"],
            context_summary="项目上下文",
        )
        assert result.status == SubagentStatus.COMPLETED
        assert result.summary == "子任务完成"
        assert result.mode == SubagentMode.AGENT_AS_TOOL

    @patch("praxis.session.core.Session.run_turn")
    async def test_spawn_timeout(
        self,
        mock_run_turn: Any,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        async def slow(*a: Any, **kw: Any) -> None:
            await asyncio.sleep(100)
        mock_run_turn.side_effect = slow
        spawner = SubagentSpawner(
            isolation=isolation,
            resource_ctrl=resource_ctrl,
            guardrails=guardrails,
            parent_registry=registry,
        )
        result = await spawner.spawn_agent_as_tool(task="超时任务", timeout_seconds=0.05)
        assert result.status == SubagentStatus.TIMEOUT

    @patch("praxis.session.core.Session.run_turn")
    async def test_spawn_failure(
        self,
        mock_run_turn: Any,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        mock_run_turn.side_effect = RuntimeError("boom")
        spawner = SubagentSpawner(
            isolation=isolation,
            resource_ctrl=resource_ctrl,
            guardrails=guardrails,
            parent_registry=registry,
        )
        result = await spawner.spawn_agent_as_tool(task="失败任务")
        assert result.status == SubagentStatus.FAILED
        assert "boom" in result.summary


# ── Task 14.2: Handoff ──────────────────────────────────────────────────────


class TestHandoffManager:
    """Handoff 测试。"""

    @patch("praxis.session.core.Session.run_turn")
    async def test_handoff_success(
        self,
        mock_run_turn: Any,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        mock_run_turn.return_value = AgentResponse(
            content="Handoff 完成",
            total_turns=2,
            termination_reason=TerminationReason.NATURAL,
            events=[],
        )
        mgr = HandoffManager(isolation, resource_ctrl, guardrails, registry)
        result = await mgr.handoff(
            target_agent_type="code_review",
            context_summary="需要代码审查",
        )
        assert result.status == SubagentStatus.COMPLETED
        assert result.mode == SubagentMode.HANDOFF
        assert "Handoff 完成" in result.summary

    @patch("praxis.session.core.Session.run_turn")
    async def test_handoff_timeout(
        self,
        mock_run_turn: Any,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        async def slow(*a: Any, **kw: Any) -> None:
            await asyncio.sleep(100)
        mock_run_turn.side_effect = slow
        mgr = HandoffManager(isolation, resource_ctrl, guardrails, registry)
        result = await mgr.handoff("review", "上下文", timeout_seconds=0.05)
        assert result.status == SubagentStatus.TIMEOUT


# ── Task 14.3: Fork ─────────────────────────────────────────────────────────


class TestForkManager:
    """Fork 测试。"""

    @patch("praxis.session.core.Session.run_turn")
    async def test_fork_multiple(
        self,
        mock_run_turn: Any,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        mock_run_turn.return_value = AgentResponse(
            content="Fork 结果",
            total_turns=1,
            termination_reason=TerminationReason.NATURAL,
            events=[],
        )
        aggregator = ResultAggregator()
        mgr = ForkManager(
            isolation, resource_ctrl, aggregator, guardrails, registry
        )
        results = await mgr.fork(
            tasks=[
                {"task": "任务1"},
                {"task": "任务2"},
            ],
            context_summary="父上下文",
        )
        assert len(results) == 2
        assert all(r.mode == SubagentMode.FORK for r in results)

    @patch("praxis.session.core.Session.run_turn")
    async def test_fork_and_aggregate(
        self,
        mock_run_turn: Any,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        mock_run_turn.return_value = AgentResponse(
            content="结果",
            total_turns=1,
            termination_reason=TerminationReason.NATURAL,
            events=[],
        )
        aggregator = ResultAggregator()
        mgr = ForkManager(
            isolation, resource_ctrl, aggregator, guardrails, registry
        )
        combined = await mgr.fork_and_aggregate(
            tasks=[{"task": "t1"}, {"task": "t2"}],
        )
        assert combined["subagent_count"] == 2

    @patch("praxis.session.core.Session.run_turn")
    async def test_fork_partial_failure(
        self,
        mock_run_turn: Any,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        guardrails: GuardrailEngine,
        registry: ToolRegistry,
    ) -> None:
        call_count = 0

        async def side_effect(*a: Any, **kw: Any) -> AgentResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("子代理1失败")
            return AgentResponse(
                content="成功",
                total_turns=1,
                termination_reason=TerminationReason.NATURAL,
                events=[],
            )

        mock_run_turn.side_effect = side_effect
        aggregator = ResultAggregator()
        mgr = ForkManager(
            isolation, resource_ctrl, aggregator, guardrails, registry
        )
        results = await mgr.fork(tasks=[{"task": "t1"}, {"task": "t2"}])
        statuses = {r.status for r in results}
        assert SubagentStatus.FAILED in statuses
        assert SubagentStatus.COMPLETED in statuses
