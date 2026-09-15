"""Host session identity and cancellable user-interaction contracts."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from praxis.config.schemas import (
    ContextConfig,
    InputConfig,
    MemoryConfig,
    ModelCapabilities,
    OrchestratorConfig,
    PersistenceConfig,
    SessionConfig,
    SubagentConfig,
    ToolsConfig,
)
from praxis.exceptions import InputSizeLimitError, ToolPolicyViolationError
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.memory.core import CognitiveMemory
from praxis.models.inputs import ImageInput, UserInput
from praxis.models.tools import FunctionCall, ToolCall, ToolDefinition, ToolMetadata
from praxis.persistence.store import PersistenceStore, create_store
from praxis.resources import ResourceController
from praxis.session.checkpoint import CheckpointManager
from praxis.session.core import SessionFactory
from praxis.session.resume import SessionResumer
from praxis.tools.executor import ToolExecutor
from praxis.tools.policy import ToolPolicy
from praxis.tools.registry import ToolRegistry


@pytest.fixture
async def host_store(tmp_path: Path) -> AsyncIterator[PersistenceStore]:
    store = await create_store(PersistenceConfig(sqlite_path=str(tmp_path / "host.db")))
    try:
        yield store
    finally:
        await store.close()


def host_factory(store: PersistenceStore) -> SessionFactory:
    return SessionFactory(
        store=store,
        session_config=SessionConfig(),
        orchestrator_config=OrchestratorConfig(),
        context_config=ContextConfig(),
        input_config=InputConfig(max_attachment_bytes=7),
        memory_config=MemoryConfig(background_enabled=False, dream_enabled=False),
    )


def host_guardrails() -> GuardrailEngine:
    return GuardrailEngine(RuleEngine(), PermissionManager())


async def test_resume_uses_saved_identity_for_events_and_scratchpad(
    host_store: PersistenceStore, mock_gateway: MagicMock,
) -> None:
    factory = host_factory(host_store)
    original = await factory.create_session(host_guardrails(), mock_gateway)
    assert original.memory is not None
    await original.memory.write_scratchpad("progress.json", {"step": "waiting"})
    await original.save_auto_checkpoint()
    await original.terminate()
    resumed = await SessionResumer(factory, CheckpointManager(host_store)).resume_session(
        original.session_id, host_guardrails(), mock_gateway,
    )
    assert resumed is not None
    try:
        assert resumed.session_id == original.session_id
        assert resumed.loop.emitter.emit("tool_call_start").session_id == original.session_id
        assert resumed.loop.coordinator.session_id == original.session_id
        assert resumed.memory is not None
        assert resumed.memory.session_id == original.session_id
        assert resumed.memory.default_scope.scope_id == original.session_id
        assert resumed.memory.working_memory.session_id == original.session_id
        assert await resumed.memory.read_scratchpad("progress.json") == {"step": "waiting"}
    finally:
        await resumed.terminate()


async def test_factory_accepts_host_session_identity(
    host_store: PersistenceStore, mock_gateway: MagicMock,
) -> None:
    session = await host_factory(host_store).create_session(
        host_guardrails(), mock_gateway, session_id="conversation-123",
    )
    try:
        assert session.session_id == "conversation-123"
        assert session.loop.emitter.emit("tool_call_start").session_id == "conversation-123"
        assert session.memory is not None
        assert session.memory.scratchpad.session_id == "conversation-123"
    finally:
        await session.terminate()


async def test_resume_preserves_host_tool_and_input_policy(
    host_store: PersistenceStore, mock_gateway: MagicMock, tmp_path: Path,
) -> None:
    factory = host_factory(host_store)
    tools = ToolsConfig(allowed_paths=(str(tmp_path),), default_timeout=0.125)
    mock_gateway.capabilities.return_value = ModelCapabilities(image=True)
    original = await factory.create_session(host_guardrails(), mock_gateway, tools_config=tools)
    await original.save_auto_checkpoint()
    await original.terminate()
    resumed = await SessionResumer(factory, CheckpointManager(host_store)).resume_session(
        original.session_id, host_guardrails(), mock_gateway,
        tools_config=tools, include_builtins=False,
    )
    assert resumed is not None
    try:
        assert resumed.loop.coordinator.executor.sandbox.allowed_paths == (tmp_path.resolve(),)
        assert resumed.loop.coordinator.executor.sandbox.default_timeout == 0.125
        assert resumed.input_resolver.config.max_attachment_bytes == 7
        assert resumed.registry.list_tools() == []
        with pytest.raises(ToolPolicyViolationError):
            resumed.loop.coordinator.executor.sandbox.check_path(str(tmp_path.parent / "outside"))
        with pytest.raises(InputSizeLimitError):
            await resumed.run_turn(UserInput(parts=(
                ImageInput.from_bytes(b"too-large-image", media_type="image/png"),
            )))
    finally:
        await resumed.terminate()


async def test_missing_checkpoint_returns_none(
    host_store: PersistenceStore, mock_gateway: MagicMock,
) -> None:
    resumed = await SessionResumer(
        host_factory(host_store), CheckpointManager(host_store),
    ).resume_session("missing", host_guardrails(), mock_gateway)
    assert resumed is None


@pytest.mark.parametrize("session_id", ["", "   "])
async def test_explicit_empty_identity_is_rejected(
    host_store: PersistenceStore, mock_gateway: MagicMock, session_id: str,
) -> None:
    with pytest.raises(ValueError, match="session_id"):
        session = await host_factory(host_store).create_session(
            host_guardrails(), mock_gateway, session_id=session_id,
        )
        await session.terminate()


async def test_explicit_identity_rejects_memory_from_another_session(
    host_store: PersistenceStore, mock_gateway: MagicMock,
) -> None:
    memory = CognitiveMemory(
        host_store, mock_gateway, "another-session",
        config=MemoryConfig(background_enabled=False, dream_enabled=False),
    )
    with pytest.raises(ValueError, match="session_id"):
        session = await host_factory(host_store).create_session(
            host_guardrails(), mock_gateway, session_id="host-session", memory=memory,
        )
        await session.terminate()


@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("shared_resources", [False, True])
async def test_interactive_wait_outlives_timeout_without_holding_tool_capacity(
    readonly: bool, shared_resources: bool,
) -> None:
    started = asyncio.Event()
    answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def wait_for_user(arguments: dict[str, Any]) -> str:
        started.set()
        return await answer

    async def ordinary_tool(arguments: dict[str, Any]) -> str:
        return "other session completed"

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="ask_user", description="Wait for user input", parameters={"type": "object"},
        metadata=ToolMetadata(interactive=True, readonly=readonly),
    ), wait_for_user)
    registry.register(ToolDefinition(
        name="ordinary", description="Normal work", parameters={"type": "object"},
        metadata=ToolMetadata(readonly=readonly),
    ), ordinary_tool)
    resources = ResourceController(SubagentConfig(), max_concurrent_readonly=1)
    policy = ToolPolicy(ToolsConfig(default_timeout=0.02, max_concurrent_readonly=1))
    executor = ToolExecutor(registry, policy, resources=resources if shared_resources else None)
    other = ToolExecutor(registry, policy, resources=resources) if shared_resources else executor
    waiting = asyncio.create_task(executor.execute("ask_user", {"key": "x", "namespace": "same"}))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(0.06)
        assert not waiting.done()
        result = await asyncio.wait_for(
            other.execute("ordinary", {"key": "x", "namespace": "same"}), timeout=0.5,
        )
        assert result.content == "other session completed"
        answer.set_result("submitted")
        assert (await waiting).content == "submitted"
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def test_interactive_cancellation_reaches_user_future_without_retry(
    host_store: PersistenceStore, mock_gateway: MagicMock,
) -> None:
    started = asyncio.Event()
    answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    calls = 0

    async def wait_for_user(arguments: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        started.set()
        return await answer

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="ask_user", description="Wait", parameters={"type": "object"},
        metadata=ToolMetadata(interactive=True, readonly=True, permission_level="auto_approve"),
    ), wait_for_user)
    session = await host_factory(host_store).create_session(
        host_guardrails(), mock_gateway, registry=registry,
    )
    waiting = asyncio.create_task(session.loop.coordinator.execute_single(ToolCall(
        id="wait-1", function=FunctionCall(name="ask_user", arguments="{}"),
    ), turn=1))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiting, timeout=1)
        assert answer.cancelled()
        assert calls == 1
        assert not any(e.event_type == "tool_retry" for e in session.loop.emitter.events)
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await session.terminate()


async def test_interactive_tool_still_enforces_permission_and_argument_schema(
    host_store: PersistenceStore, mock_gateway: MagicMock,
) -> None:
    calls = 0

    async def handler(arguments: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return "must not run"

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="ask_user", description="Wait", parameters={"type": "object", "required": ["text"]},
        metadata=ToolMetadata(interactive=True, permission_level="deny"),
    ), handler)
    session = await host_factory(host_store).create_session(
        host_guardrails(), mock_gateway, registry=registry,
    )
    try:
        denied = await session.loop.coordinator.execute_single(ToolCall(
            id="denied", function=FunctionCall(name="ask_user", arguments='{"text":"x"}'),
        ), turn=1)
        assert denied.skipped
        invalid = await session.loop.coordinator.executor.execute("ask_user", {})
        assert not invalid.success
        assert calls == 0
    finally:
        await session.terminate()
