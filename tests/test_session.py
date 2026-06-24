"""S12 会话管理单元测试。"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from praxis.config.schemas import (
    ContextConfig,
    SessionConfig,
    OrchestratorConfig,
    PersistenceConfig,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.session.checkpoint import CheckpointManager
from praxis.session.continuation import ContinuationManager
from praxis.session.resume import SessionResumer
from praxis.session.core import Session, SessionFactory
from praxis.session.time_travel import TimeTravelManager
from praxis.models.session import (
    CheckpointInfo,
    ContinuationPhase,
    SessionMetadata,
    SessionSnapshot,
    SessionStatus,
)
from praxis.models.persistence import Checkpoint
from praxis.persistence.store import PersistenceStore, create_store


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
def factory(store: PersistenceStore) -> SessionFactory:
    return SessionFactory(
        store=store,
        session_config=SessionConfig(),
        orchestrator_config=OrchestratorConfig(),
        context_config=ContextConfig(),
    )


# ── Task 13.1: 会话初始化 ───────────────────────────────────────────────────


class TestSessionFactory:
    """会话初始化测试。"""

    async def test_create_session(
        self,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        try:
            assert session.session_id
            assert session.status == SessionStatus.ACTIVE
            assert session.metadata.total_turns == 0
            assert session.loop is not None
            assert session.assembler is not None
            assert session.registry is not None
        finally:
            await session.terminate()

    async def test_session_has_unique_id(
        self,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        s1 = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        s2 = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        try:
            assert s1.session_id != s2.session_id
        finally:
            await s1.terminate()
            await s2.terminate()

    async def test_terminate_session(
        self,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        await session.terminate()
        assert session.status == SessionStatus.TERMINATED

    async def test_recovery_and_strategy_config_wired(
        self,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        """RecoveryConfig 与 default_strategy 应装配进 session（此前用硬编码默认值）。"""
        from praxis.config.schemas import OrchestratorConfig, RecoveryConfig
        from praxis.models.orchestrator import StrategyMode

        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(),
            orchestrator_config=OrchestratorConfig(default_strategy="plan-and-execute"),
            context_config=ContextConfig(),
            recovery_config=RecoveryConfig(
                max_retries=7, circuit_breaker_threshold=9, circuit_breaker_cooldown=123.0,
            ),
        )
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        try:
            coord = session.loop.coordinator
            assert coord.retry_policy.max_retries == 7
            assert coord.circuits.failure_threshold == 9
            assert coord.circuits.cooldown_seconds == 123.0
            assert session.loop.strategy.mode == StrategyMode.PLAN_AND_EXECUTE
        finally:
            await session.terminate()

    async def test_fallback_mappings_wired_from_config(
        self,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        """ToolsConfig.fallback_mappings 应流入协调器的降级表，激活 S9 降级链路。"""
        from praxis.config.schemas import ToolsConfig

        tools_config = ToolsConfig(fallback_mappings={"web_search": "web_fetch"})
        session = await factory.create_session(
            guardrails=guardrails,
            gateway=mock_gateway,
            tools_config=tools_config,
        )
        try:
            fallbacks = session.loop.coordinator.fallbacks
            assert fallbacks is not None
            assert fallbacks.get_fallback("web_search") == "web_fetch"
        finally:
            await session.terminate()


# ── Task 13.3: 自动检查点 ───────────────────────────────────────────────────


class TestCheckpointManager:
    """检查点管理测试。"""

    async def test_save_and_load(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="sess-1", total_turns=3)

        cp_id = await mgr.save_checkpoint(
            metadata=metadata,
            context_state={"messages": [{"role": "user", "content": "hi"}]},
            memory_state={"cursor": 5},
            loop_state={"current_turn": 3},
            description="测试检查点",
        )
        assert cp_id
        assert metadata.checkpoint_count == 1

        loaded = await mgr.load_checkpoint("sess-1", cp_id)
        assert loaded is not None
        assert loaded.session_id == "sess-1"
        assert loaded.description == "测试检查点"

    async def test_load_latest(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="sess-2", total_turns=1)

        cp1 = await mgr.save_checkpoint(metadata, {}, {}, {}, description="cp1")
        metadata.total_turns = 2
        cp2 = await mgr.save_checkpoint(metadata, {}, {}, {}, description="cp2")

        latest = await mgr.load_latest("sess-2")
        assert latest is not None
        assert latest.checkpoint_id == cp2

    async def test_list_checkpoints(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="sess-3")

        for i in range(3):
            metadata.total_turns = i + 1
            await mgr.save_checkpoint(metadata, {}, {}, {}, description=f"cp-{i}")

        infos = await mgr.list_checkpoints("sess-3")
        assert len(infos) == 3

    async def test_delete_checkpoint(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="sess-del")
        cp_id = await mgr.save_checkpoint(metadata, {}, {}, {})
        await mgr.delete_checkpoint("sess-del", cp_id)
        loaded = await mgr.load_checkpoint("sess-del", cp_id)
        assert loaded is None

    async def test_extract_snapshot(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="sess-snap", total_turns=5)
        context_state = {"messages": [{"role": "user", "content": "hello"}]}
        cp_id = await mgr.save_checkpoint(metadata, context_state, {}, {})
        cp = await mgr.load_checkpoint("sess-snap", cp_id)
        snapshot = await mgr.extract_snapshot(cp)
        assert snapshot.metadata.total_turns == 5
        assert snapshot.context_state == context_state

    async def test_load_nonexistent(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        result = await mgr.load_checkpoint("no-exist", "no-cp")
        assert result is None

    async def test_load_latest_nonexistent(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        result = await mgr.load_latest("no-exist")
        assert result is None


# ── Task 13.2: 会话恢复 ─────────────────────────────────────────────────────


class TestSessionResumer:
    """会话恢复测试。"""

    async def test_resume_from_checkpoint(
        self,
        store: PersistenceStore,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        # 保存检查点
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="resume-1", total_turns=5, total_tokens=1000)
        context_state = {
            "messages": [{"role": "user", "content": "历史消息"}],
            "file_refs": ["a.py"],
            "compaction_count": 1,
        }
        await mgr.save_checkpoint(metadata, context_state, {}, {})

        # 恢复
        resumer = SessionResumer(factory, mgr)
        session = await resumer.resume_session("resume-1", guardrails, gateway=mock_gateway)
        assert session is not None
        assert session.session_id == "resume-1"
        assert session.metadata.total_turns == 5
        assert session.metadata.total_tokens == 1000
        assert session.assembler.conversation_history == [{"role": "user", "content": "历史消息"}]
        assert session.assembler.file_refs == ["a.py"]
        assert session.metadata.continuation_phase == ContinuationPhase.WARMUP

    async def test_resume_nonexistent(
        self,
        store: PersistenceStore,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        mgr = CheckpointManager(store)
        resumer = SessionResumer(factory, mgr)
        session = await resumer.resume_session("nonexistent", guardrails, gateway=mock_gateway)
        assert session is None

    async def test_top_level_resume_agent_session(
        self,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        """顶层 resume_agent_session 应能从检查点恢复（公共入口链路）。"""
        from praxis.agent import resume_agent_session

        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="top-resume", total_turns=3)
        await mgr.save_checkpoint(
            metadata,
            {"messages": [{"role": "user", "content": "hi"}]},
            {},
            {},
        )
        session = await resume_agent_session(
            store=store,
            guardrails=guardrails,
            gateway=mock_gateway,
            session_id="top-resume",
        )
        try:
            assert session is not None
            assert session.session_id == "top-resume"
            assert session.metadata.total_turns == 3
        finally:
            if session is not None:
                await session.terminate()
        assert await resume_agent_session(
            store=store, guardrails=guardrails, gateway=mock_gateway,
            session_id="does-not-exist",
        ) is None

    async def test_resumed_session_injects_warmup_and_advances(
        self,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        """恢复的会话首轮应注入热身序列，并在轮后推进到 WORKING 阶段。"""
        from praxis.agent import resume_agent_session
        from praxis.session.continuation import WARMUP_SYSTEM_PROMPT

        mgr = CheckpointManager(store)
        await mgr.save_checkpoint(
            SessionMetadata(session_id="warm-1", total_turns=1), {}, {}, {},
        )
        session = await resume_agent_session(
            store=store, guardrails=guardrails, gateway=mock_gateway,
            session_id="warm-1",
        )
        assert session is not None
        try:
            assert session.continuation is not None
            assert session.metadata.continuation_phase == ContinuationPhase.WARMUP

            captured: dict[str, Any] = {}

            async def fake_run(user_message: str, **kwargs: Any) -> Any:
                captured.update(kwargs)
                return SimpleNamespace(total_turns=1, events=[])

            session.loop.run = fake_run  # type: ignore[assignment]
            await session.run_turn("继续")
            assert captured.get("developer_instructions") == WARMUP_SYSTEM_PROMPT
            assert session.metadata.continuation_phase == ContinuationPhase.WORKING
        finally:
            await session.terminate()

    async def test_validate_integrity(
        self,
        store: PersistenceStore,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="int-1")
        await mgr.save_checkpoint(metadata, {}, {}, {})

        resumer = SessionResumer(factory, mgr)
        session = await resumer.resume_session("int-1", guardrails, gateway=mock_gateway)
        issues = resumer.validate_integrity(session)
        assert issues == []


# ── Task 13.4: 跨上下文窗口续接 ─────────────────────────────────────────────


class TestContinuationManager:
    """跨上下文窗口续接测试。"""

    def test_init_phase_prompt(self) -> None:
        mgr = ContinuationManager()
        prompt = mgr.get_system_prompt(ContinuationPhase.INITIALIZATION)
        assert "初始化" in prompt
        assert "进度文件" in prompt

    def test_warmup_phase_prompt(self) -> None:
        mgr = ContinuationManager()
        prompt = mgr.get_system_prompt(ContinuationPhase.WARMUP)
        assert "恢复" in prompt
        assert "热身" in prompt

    def test_working_phase_no_prompt(self) -> None:
        mgr = ContinuationManager()
        prompt = mgr.get_system_prompt(ContinuationPhase.WORKING)
        assert prompt == ""

    async def test_advance_phase(
        self,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        mgr = ContinuationManager()
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        try:
            session.metadata.continuation_phase = ContinuationPhase.INITIALIZATION
            new_phase = mgr.advance_phase(session)
            assert new_phase == ContinuationPhase.WORKING
        finally:
            await session.terminate()

    async def test_advance_from_warmup(
        self,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        mgr = ContinuationManager()
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        try:
            session.metadata.continuation_phase = ContinuationPhase.WARMUP
            new_phase = mgr.advance_phase(session)
            assert new_phase == ContinuationPhase.WORKING
        finally:
            await session.terminate()

    def test_feature_list(self) -> None:
        mgr = ContinuationManager()
        features = [
            {"name": "auth", "status": "completed", "description": "认证"},
            {"name": "api", "status": "in_progress", "description": "API"},
            {"name": "ui", "status": "pending", "description": "界面"},
        ]
        mgr.set_feature_list(features)
        assert len(mgr.get_feature_list()) == 3

    def test_update_feature_status(self) -> None:
        mgr = ContinuationManager()
        mgr.set_feature_list([{"name": "auth", "status": "pending"}])
        assert mgr.update_feature_status("auth", "completed") is True
        assert mgr.get_feature_list()[0]["status"] == "completed"
        assert mgr.update_feature_status("nonexist", "done") is False

    def test_get_next_feature(self) -> None:
        mgr = ContinuationManager()
        mgr.set_feature_list([
            {"name": "a", "status": "completed"},
            {"name": "b", "status": "in_progress"},
            {"name": "c", "status": "pending"},
        ])
        nxt = mgr.get_next_feature()
        assert nxt is not None
        assert nxt["name"] == "b"

    def test_get_next_feature_pending(self) -> None:
        mgr = ContinuationManager()
        mgr.set_feature_list([
            {"name": "a", "status": "completed"},
            {"name": "b", "status": "pending"},
        ])
        nxt = mgr.get_next_feature()
        assert nxt["name"] == "b"

    def test_progress_summary(self) -> None:
        mgr = ContinuationManager()
        mgr.set_feature_list([
            {"name": "a", "status": "completed"},
            {"name": "b", "status": "in_progress"},
            {"name": "c", "status": "pending"},
        ])
        summary = mgr.get_progress_summary()
        assert "1/3 完成" in summary

    async def test_prepare_turn_init(
        self,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        mgr = ContinuationManager()
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        try:
            session.metadata.continuation_phase = ContinuationPhase.INITIALIZATION
            kwargs = mgr.prepare_turn(session)
            assert "developer_instructions" in kwargs
        finally:
            await session.terminate()


# ── Task 13.5: 时间旅行调试 ─────────────────────────────────────────────────


class TestTimeTravelManager:
    """时间旅行调试测试。"""

    async def test_list_checkpoints(self, store: PersistenceStore) -> None:
        cp_mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="tt-1")
        for i in range(3):
            metadata.total_turns = i + 1
            await cp_mgr.save_checkpoint(metadata, {}, {}, {})

        factory = MagicMock(spec=SessionFactory)
        resumer = SessionResumer(factory, cp_mgr)
        tt = TimeTravelManager(cp_mgr, resumer)

        infos = await tt.list_checkpoints("tt-1")
        assert len(infos) == 3

    async def test_rollback(
        self,
        store: PersistenceStore,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        cp_mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="tt-rb")

        # 保存 3 个检查点
        cp_ids: list[str] = []
        for i in range(3):
            metadata.total_turns = i + 1
            cp_id = await cp_mgr.save_checkpoint(
                metadata,
                {"messages": [{"role": "user", "content": f"turn-{i+1}"}]},
                {},
                {},
            )
            cp_ids.append(cp_id)

        resumer = SessionResumer(factory, cp_mgr)
        tt = TimeTravelManager(cp_mgr, resumer)

        # 回退到第一个检查点
        session = await tt.rollback("tt-rb", cp_ids[0], guardrails, gateway=mock_gateway)
        assert session is not None
        assert session.metadata.total_turns == 1
        assert session.assembler.conversation_history == [
            {"role": "user", "content": "turn-1"}
        ]

    async def test_rollback_nonexistent(
        self,
        store: PersistenceStore,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        cp_mgr = CheckpointManager(store)
        resumer = SessionResumer(factory, cp_mgr)
        tt = TimeTravelManager(cp_mgr, resumer)
        session = await tt.rollback("no-exist", "no-cp", guardrails, gateway=mock_gateway)
        assert session is None

    async def test_rollback_and_prune(
        self,
        store: PersistenceStore,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        cp_mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="tt-prune")

        cp_ids: list[str] = []
        for i in range(5):
            metadata.total_turns = i + 1
            cp_id = await cp_mgr.save_checkpoint(metadata, {}, {}, {})
            cp_ids.append(cp_id)

        resumer = SessionResumer(factory, cp_mgr)
        tt = TimeTravelManager(cp_mgr, resumer)

        # 回退到第 3 个并清理
        session = await tt.rollback_and_prune("tt-prune", cp_ids[2], guardrails, gateway=mock_gateway)
        assert session is not None
        assert session.metadata.total_turns == 3

        # 后续检查点应已删除
        remaining = await cp_mgr.list_checkpoints("tt-prune")
        remaining_ids = [r.checkpoint_id for r in remaining]
        assert cp_ids[3] not in remaining_ids
        assert cp_ids[4] not in remaining_ids
        # 前 3 个仍在
        assert cp_ids[0] in remaining_ids
        assert cp_ids[1] in remaining_ids
        assert cp_ids[2] in remaining_ids
