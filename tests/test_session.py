"""S12 会话管理单元测试。"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from praxis.config.schemas import (
    ContextConfig,
    InputConfig,
    ModelCapabilities,
    OrchestratorConfig,
    PersistenceConfig,
    SessionConfig,
)
from praxis.exceptions import (
    CheckpointCorruptionError,
    PersistenceError,
    UnsupportedInputModalityError,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.models.inputs import ImageInput, UserInput
from praxis.models.messages import ResolvedUserInput
from praxis.models.orchestrator import AgentEvent, StrategyMode
from praxis.models.responses import ModelResponse, Usage
from praxis.models.session import (
    ContinuationPhase,
    SessionMetadata,
    SessionStatus,
)
from praxis.models.tools import ToolExecutionRecord, ToolExecutionState
from praxis.orchestrator.strategy import PlanStep
from praxis.persistence.backends.sqlite import SqliteBackend
from praxis.persistence.store import PersistenceStore, create_store
from praxis.session.checkpoint import CheckpointManager
from praxis.session.continuation import ContinuationManager
from praxis.session.core import SessionFactory
from praxis.session.resume import SessionResumer
from praxis.session.time_travel import TimeTravelManager

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

    async def test_unsupported_modality_is_rejected_before_orchestration(
        self,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        mock_gateway.capabilities.return_value = ModelCapabilities()
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
            input_config=InputConfig(),
        )
        session = await factory.create_session(
            guardrails=guardrails,
            gateway=mock_gateway,
        )
        user_input = UserInput(
            text="describe",
            parts=(ImageInput.from_bytes(b"png", media_type="image/png"),),
        )
        try:
            with pytest.raises(UnsupportedInputModalityError):
                await session.run_turn(user_input)
            assert session.assembler.conversation_history == []
            assert session.loop.state.current_turn == 0
        finally:
            await session.terminate()

    async def test_run_and_stream_each_resolve_input_exactly_once(
        self,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        mock_gateway.capabilities.return_value = ModelCapabilities(image=True)
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
            input_config=InputConfig(),
        )
        session = await factory.create_session(
            guardrails=guardrails,
            gateway=mock_gateway,
        )
        user_input = UserInput(
            text="describe",
            parts=(ImageInput.from_bytes(b"png", media_type="image/png"),),
        )
        resolved = ResolvedUserInput(content="resolved", text_projection="safe")
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=resolved)

        async def stream_events(
            resolved_input: ResolvedUserInput,
            **kwargs: Any,
        ) -> AsyncGenerator[AgentEvent, None]:
            assert resolved_input is resolved
            yield session.loop.emitter.emit("turn_start", turn=0)

        try:
            session.input_resolver = resolver
            regular_response = MagicMock(total_turns=1, events=[])
            with patch.object(session.loop, "run", AsyncMock(return_value=regular_response)):
                await session.run_turn(user_input)
            resolver.resolve.assert_awaited_once_with(
                user_input,
                session.model_capabilities,
            )

            resolver.resolve.reset_mock()
            with patch.object(session.loop, "run_stream", stream_events):
                events = [event async for event in session.run_turn_stream(user_input)]
            assert events
            resolver.resolve.assert_awaited_once_with(
                user_input,
                session.model_capabilities,
            )
        finally:
            await session.terminate()

    async def test_auto_checkpoint_runs_after_multimodal_history_sanitation(
        self,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        mock_gateway.capabilities.return_value = ModelCapabilities(image=True)
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=True),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
            input_config=InputConfig(),
        )
        session = await factory.create_session(
            guardrails=guardrails,
            gateway=mock_gateway,
        )
        user_input = UserInput(
            text="describe",
            parts=(ImageInput.from_bytes(b"png", media_type="image/png"),),
        )
        response = ModelResponse(
            id="response-1",
            content="ok",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="test-model",
            finish_reason="stop",
            created=1,
        )
        checkpoint_history: list[list[dict[str, Any]]] = []

        async def capture_checkpoint() -> str:
            checkpoint_history.append([
                dict(message) for message in session.assembler.conversation_history
            ])
            return "checkpoint-1"

        try:
            with (
                patch("praxis.orchestrator.loop.chat", AsyncMock(return_value=response)),
                patch.object(
                    session,
                    "save_auto_checkpoint",
                    AsyncMock(side_effect=capture_checkpoint),
                ),
            ):
                result = await session.run_turn(user_input)

            assert result.content == "ok"
            assert checkpoint_history == [[
                {
                    "role": "user",
                    "content": (
                        "describe\n"
                        "[attachment kind=image filename=image.bin "
                        "media_type=image/png size_bytes=3]"
                    ),
                },
                {"role": "assistant", "content": "ok"},
            ]]
            assert "data:" not in str(checkpoint_history)
        finally:
            await session.terminate()

    async def test_stream_close_cascades_and_checkpoint_sees_sanitized_history(
        self,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        mock_gateway.capabilities.return_value = ModelCapabilities(image=True)
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
            input_config=InputConfig(),
        )
        session = await factory.create_session(
            guardrails=guardrails,
            gateway=mock_gateway,
        )
        user_input = UserInput(
            text="describe",
            parts=(ImageInput.from_bytes(b"png", media_type="image/png"),),
        )
        delegated_closed = asyncio.Event()
        delegated_streams: list[AsyncGenerator[AgentEvent, None]] = []
        original_run_stream = session.loop.run_stream

        async def observe_delegated_stream(
            resolved_input: ResolvedUserInput,
            stream_kwargs: dict[str, Any],
        ) -> AsyncGenerator[AgentEvent, None]:
            inner_stream = original_run_stream(resolved_input, **stream_kwargs)
            try:
                async with aclosing(inner_stream):
                    async for event in inner_stream:
                        yield event
            finally:
                delegated_closed.set()

        def observed_run_stream(
            resolved_input: ResolvedUserInput,
            **kwargs: Any,
        ) -> AsyncGenerator[AgentEvent, None]:
            delegated = observe_delegated_stream(resolved_input, kwargs)
            delegated_streams.append(delegated)
            return delegated

        outer_stream = session.run_turn_stream(user_input)
        try:
            with patch.object(session.loop, "run_stream", observed_run_stream):
                assert (await anext(outer_stream)).event_type == "turn_start"
                assert "data:" in str(session.assembler.conversation_history)
                await outer_stream.aclose()

            assert delegated_closed.is_set()
            assert "data:" not in str(session.assembler.conversation_history)
            assert "cG5n" not in str(session.assembler.conversation_history)

            checkpoint_id = await session.save_auto_checkpoint()
            assert checkpoint_id is not None
            checkpoint = await CheckpointManager(store).load_checkpoint(
                session.session_id,
                checkpoint_id,
            )
            assert checkpoint is not None
            assert "data:" not in str(checkpoint.state)
            assert "cG5n" not in str(checkpoint.state)
        finally:
            await outer_stream.aclose()
            for delegated in delegated_streams:
                await delegated.aclose()
            await session.terminate()

    async def test_stream_close_finalizes_auto_checkpoint(
        self,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=True),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)

        async def one_event(
            resolved_input: ResolvedUserInput,
            **kwargs: Any,
        ) -> AsyncGenerator[AgentEvent, None]:
            yield session.loop.emitter.emit("turn_start", turn=1)

        stream = session.run_turn_stream("stop early")
        try:
            with (
                patch.object(session.loop, "run_stream", one_event),
                patch.object(session, "save_auto_checkpoint", AsyncMock(return_value="cp")) as save,
            ):
                await anext(stream)
                await stream.aclose()
                save.assert_awaited_once()
        finally:
            await stream.aclose()
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

    async def test_auto_checkpoint_pruning(
        self,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        """超过 max_checkpoints_per_session 的旧检查点应被裁剪。"""
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=True, max_checkpoints_per_session=3),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        try:
            for checkpoint_index in range(6):  # noqa: B007 - public discard name
                await session.save_auto_checkpoint()
            mgr = CheckpointManager(store)
            infos = await mgr.list_checkpoints(session.session_id)
            assert len(infos) == 3
        finally:
            await session.terminate()

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

        await mgr.save_checkpoint(metadata, {}, {}, {}, description="cp1")
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

    async def test_delete_latest_removes_pointer_without_rollback(
        self,
        store: PersistenceStore,
    ) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="delete-latest")
        old_id = await mgr.save_checkpoint(metadata, {}, {}, {}, description="old")
        latest_id = await mgr.save_checkpoint(metadata, {}, {}, {}, description="latest")

        await mgr.delete_checkpoint("delete-latest", latest_id)

        assert await mgr.load_latest("delete-latest") is None
        assert await mgr.load_checkpoint("delete-latest", old_id) is not None
        assert [item.checkpoint_id for item in await mgr.list_checkpoints("delete-latest")] == [
            old_id,
        ]

    async def test_delete_old_checkpoint_preserves_latest(
        self,
        store: PersistenceStore,
    ) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="delete-old")
        old_id = await mgr.save_checkpoint(metadata, {}, {}, {}, description="old")
        latest_id = await mgr.save_checkpoint(metadata, {}, {}, {}, description="latest")

        await mgr.delete_checkpoint("delete-old", old_id)

        latest = await mgr.load_latest("delete-old")
        assert latest is not None
        assert latest.checkpoint_id == latest_id
        assert await mgr.load_checkpoint("delete-old", old_id) is None

    async def test_delete_latest_propagates_body_failure_after_removing_pointer(
        self,
        store: PersistenceStore,
    ) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="delete-failure")
        checkpoint_id = await mgr.save_checkpoint(metadata, {}, {}, {})
        backend = store.backend
        assert isinstance(backend, SqliteBackend)
        async with backend.engine.begin() as connection:
            await connection.exec_driver_sql(f"""
                CREATE TRIGGER fail_checkpoint_body_delete
                BEFORE DELETE ON kv_store
                WHEN OLD.namespace = 'checkpoints'
                  AND OLD.key = 'delete-failure:{checkpoint_id}'
                BEGIN
                    SELECT RAISE(FAIL, 'synthetic checkpoint delete failure');
                END
            """)

        with pytest.raises(IntegrityError, match="synthetic checkpoint delete failure"):
            await mgr.delete_checkpoint("delete-failure", checkpoint_id)

        assert await mgr.load_latest("delete-failure") is None
        assert await mgr.load_checkpoint("delete-failure", checkpoint_id) is not None

    async def test_failed_first_latest_write_leaves_safe_orphan_body(
        self,
        store: PersistenceStore,
    ) -> None:
        mgr = CheckpointManager(store)
        backend = store.backend
        assert isinstance(backend, SqliteBackend)
        async with backend.engine.begin() as connection:
            await connection.exec_driver_sql("""
                CREATE TRIGGER fail_first_latest_write
                BEFORE INSERT ON kv_store
                WHEN NEW.namespace = 'checkpoints'
                  AND NEW.key = 'save-failure:latest'
                BEGIN
                    SELECT RAISE(FAIL, 'synthetic latest write failure');
                END
            """)

        with pytest.raises(IntegrityError, match="synthetic latest write failure"):
            await mgr.save_checkpoint(
                SessionMetadata(session_id="save-failure"),
                {},
                {},
                {},
            )

        keys = await store.list_keys("checkpoints", prefix="save-failure:")
        assert len(keys) == 1
        assert not keys[0].endswith(":latest")
        assert await mgr.load_latest("save-failure") is None

    async def test_storage_failure_propagates_from_latest_lookup(
        self,
        store: PersistenceStore,
    ) -> None:
        mgr = CheckpointManager(store)
        await store.close()

        with pytest.raises(PersistenceError, match="已关闭"):
            await mgr.load_latest("storage-failure")

    async def test_extract_snapshot(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        metadata = SessionMetadata(session_id="sess-snap", total_turns=5)
        context_state = {"messages": [{"role": "user", "content": "hello"}]}
        cp_id = await mgr.save_checkpoint(metadata, context_state, {}, {})
        cp = await mgr.load_checkpoint("sess-snap", cp_id)
        snapshot = await mgr.extract_snapshot(cp)
        assert snapshot.metadata.total_turns == 5
        assert snapshot.context_state == context_state

    async def test_auto_checkpoint_restores_plan_and_execution_ledger(
        self,
        factory: SessionFactory,
        store: PersistenceStore,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        session.loop.strategy.switch_mode(StrategyMode.PLAN_AND_EXECUTE)
        session.loop.strategy.set_plan([PlanStep("write durable state", "write_file")])
        session.tool_execution_ledger["call-1"] = ToolExecutionRecord(
            tool_call_id="call-1",
            tool_name="write_file",
            argument_digest="digest",
            state=ToolExecutionState.STARTED,
            readonly=False,
            idempotent=False,
        )
        try:
            await session.save_auto_checkpoint()
            restored = await SessionResumer(
                factory,
                CheckpointManager(store),
            ).resume_session(session.session_id, guardrails, gateway=mock_gateway)

            assert restored is not None
            assert restored.loop.strategy.get_current_step() is not None
            assert restored.loop.strategy.get_current_step().description == "write durable state"
            assert (
                restored.tool_execution_ledger["call-1"].state
                is ToolExecutionState.UNCERTAIN
            )
            await restored.terminate()
        finally:
            await session.terminate()

    async def test_load_nonexistent(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        result = await mgr.load_checkpoint("no-exist", "no-cp")
        assert result is None

    async def test_load_latest_nonexistent(self, store: PersistenceStore) -> None:
        mgr = CheckpointManager(store)
        result = await mgr.load_latest("no-exist")
        assert result is None

    @pytest.mark.parametrize(
        "latest_value",
        [None, ["private-marker"], {"unexpected": "private-marker"}],
    )
    async def test_load_latest_rejects_malformed_reference(
        self,
        store: PersistenceStore,
        latest_value: object,
    ) -> None:
        mgr = CheckpointManager(store)
        await store.save("checkpoints", "malformed-latest:latest", latest_value)

        with pytest.raises(CheckpointCorruptionError) as exc_info:
            await mgr.load_latest("malformed-latest")

        assert "private-marker" not in str(exc_info.value)

    async def test_load_latest_rejects_missing_referenced_body(
        self,
        store: PersistenceStore,
    ) -> None:
        mgr = CheckpointManager(store)
        await store.save(
            "checkpoints",
            "dangling-latest:latest",
            "private-marker-missing",
        )

        with pytest.raises(CheckpointCorruptionError) as exc_info:
            await mgr.load_latest("dangling-latest")

        assert "private-marker" not in str(exc_info.value)

    async def test_load_latest_rejects_nonobject_referenced_body(
        self,
        store: PersistenceStore,
    ) -> None:
        mgr = CheckpointManager(store)
        await store.save(
            "checkpoints",
            "nonobject-latest:latest",
            "checkpoint-id",
        )
        await store.save(
            "checkpoints",
            "nonobject-latest:checkpoint-id",
            ["private-marker"],
        )

        with pytest.raises(CheckpointCorruptionError) as exc_info:
            await mgr.load_latest("nonobject-latest")

        assert "private-marker" not in str(exc_info.value)

    @pytest.mark.parametrize(
        ("checkpoint_id", "body"),
        [("null-body", None), ("list-body", ["private-marker"])],
    )
    async def test_load_checkpoint_rejects_nonobject_body(
        self,
        store: PersistenceStore,
        checkpoint_id: str,
        body: object,
    ) -> None:
        mgr = CheckpointManager(store)
        await store.save(
            "checkpoints",
            f"nonobject:{checkpoint_id}",
            body,
        )

        with pytest.raises(CheckpointCorruptionError) as exc_info:
            await mgr.load_checkpoint("nonobject", checkpoint_id)

        assert "private-marker" not in str(exc_info.value)


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

    async def test_resume_propagates_checkpoint_corruption_before_factory_path(
        self,
        store: PersistenceStore,
        factory: SessionFactory,
        guardrails: GuardrailEngine,
        mock_gateway: MagicMock,
    ) -> None:
        await store.save(
            "checkpoints",
            "corrupt-resume:latest",
            {"unexpected": "private-marker"},
        )
        mock_gateway.capabilities.reset_mock()

        with pytest.raises(CheckpointCorruptionError):
            await SessionResumer(
                factory,
                CheckpointManager(store),
            ).resume_session("corrupt-resume", guardrails, gateway=mock_gateway)

        mock_gateway.capabilities.assert_not_called()

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
