"""应用级 Runtime 与 AgentSession 生命周期测试。"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config import PraxisConfig
from praxis.config.schemas import (
    GatewayConfig,
    PersistenceConfig,
    VerificationConfig,
)
from praxis.exceptions import ConcurrentSessionRunError, RuntimeStateError, SessionError
from praxis.models.orchestrator import AgentEvent, AgentResponse
from praxis.models.responses import ModelResponse, ModelResponseChunk
from praxis.models.session import SessionStatus
from praxis.models.subagent import SubagentSpec
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.runtime import HealthStatus, PraxisRuntime
from praxis.tools.registry import ToolRegistry


class FakeGateway:
    def __init__(self) -> None:
        self.config = GatewayConfig()
        self.closed = False

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        return ModelResponse(content="unused")

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelResponseChunk]:
        if False:
            yield ModelResponseChunk(id="unused")

    async def health(self) -> bool:
        return not self.closed

    def supports_vision(self, model_name: str | None = None) -> bool:
        return False

    async def close(self) -> None:
        self.closed = True


class FakeRunner:
    def __init__(self, started: asyncio.Event | None = None, release: asyncio.Event | None = None):
        self.session_id = "fake-session"
        self.status = SessionStatus.ACTIVE
        self.started = started
        self.release = release
        self.terminated = False
        self.aborted = False

    async def run_turn(self, user_message: str, **kwargs: Any) -> AgentResponse:
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        return AgentResponse(content=f"echo:{user_message}")

    async def run_turn_stream(
        self,
        user_message: str,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(event_type="content", data={"content": user_message})

    def abort(self) -> None:
        self.aborted = True

    async def terminate(self) -> None:
        self.terminated = True
        self.status = SessionStatus.TERMINATED


def runtime_config(tmp_path: Path) -> PraxisConfig:
    return PraxisConfig(
        persistence=PersistenceConfig(
            backend="filesystem",
            filesystem_path=str(tmp_path / "store"),
        ),
    )


async def test_runtime_and_session_context_lifecycle(tmp_path: Path) -> None:
    gateway = FakeGateway()
    runner = FakeRunner()

    async with PraxisRuntime(
        runtime_config(tmp_path),
        gateway=gateway,
        session_builder=lambda _: asyncio.sleep(0, result=runner),
    ) as runtime:
        assert runtime.started
        async with runtime.session() as session:
            response = await session.run("hello")
            assert response.content == "echo:hello"
            assert session.status is SessionStatus.ACTIVE
        assert runner.terminated

    assert gateway.closed
    assert not runtime.started


async def test_start_and_close_are_idempotent(tmp_path: Path) -> None:
    runtime = PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        session_builder=lambda _: asyncio.sleep(0, result=FakeRunner()),
    )
    await asyncio.gather(runtime.start(), runtime.start())
    assert runtime.started
    await asyncio.gather(runtime.close(), runtime.close())
    assert not runtime.started


async def test_session_rejects_concurrent_runs(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = FakeRunner(started, release)
    async with PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        session_builder=lambda _: asyncio.sleep(0, result=runner),
    ) as runtime:
        async with runtime.session() as session:
            first = asyncio.create_task(session.run("first"))
            await started.wait()
            with pytest.raises(ConcurrentSessionRunError):
                await session.run("second")
            release.set()
            await first


async def test_session_requires_context_manager(tmp_path: Path) -> None:
    async with PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        session_builder=lambda _: asyncio.sleep(0, result=FakeRunner()),
    ) as runtime:
        session = runtime.session()
        with pytest.raises(SessionError, match="尚未启动"):
            await session.run("hello")


async def test_health_reports_local_embedding_degradation(tmp_path: Path) -> None:
    async with PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        session_builder=lambda _: asyncio.sleep(0, result=FakeRunner()),
    ) as runtime:
        health = await runtime.health()
        assert health.status is HealthStatus.DEGRADED
        assert health.components["model"].status is HealthStatus.READY
        assert health.components["embedding"].status is HealthStatus.DEGRADED


async def test_health_reports_unavailable_visual_capability(tmp_path: Path) -> None:
    config = runtime_config(tmp_path).model_copy(
        update={"verification": VerificationConfig(visual_enabled=True)}
    )
    async with PraxisRuntime(
        config,
        gateway=FakeGateway(),
        session_builder=lambda _: asyncio.sleep(0, result=FakeRunner()),
    ) as runtime:
        health = await runtime.health()
        assert health.components["visual"].status is HealthStatus.DEGRADED
        assert "视觉能力" in health.components["visual"].detail


async def test_runtime_builds_owned_subagent_with_explicit_tool_subset(tmp_path: Path) -> None:
    parent_registry = ToolRegistry()

    async def handler(_arguments: dict[str, Any]) -> str:
        return "ok"

    for name in ("allowed", "not_allowed"):
        parent_registry.register(
            ToolDefinition(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
                metadata=ToolMetadata(),
            ),
            handler,
        )

    runtime = PraxisRuntime(runtime_config(tmp_path), gateway=FakeGateway())
    await runtime.start()
    child = await runtime.build_subagent_session(
        SubagentSpec(task="isolated", tool_names=["allowed"], max_turns=3),
        parent_registry,
    )
    try:
        assert child.registry.list_tools() == ["allowed"]
        assert child.loop.config.max_turns == 3
        assert runtime.owned_subagent_count == 1
    finally:
        await runtime.release_subagent_session(child)
        await runtime.close()

    assert runtime.owned_subagent_count == 0


async def test_runtime_rejects_session_before_start_and_restart_after_close(tmp_path: Path) -> None:
    runtime = PraxisRuntime(runtime_config(tmp_path), gateway=FakeGateway())
    assert runtime.state.value == "new"
    assert (await runtime.health()).status is HealthStatus.FAILED
    with pytest.raises(RuntimeStateError, match="尚未启动"):
        runtime.session()
    with pytest.raises(RuntimeStateError, match="尚未启动"):
        await runtime.build_subagent_session(SubagentSpec(task="x"), ToolRegistry())
    await runtime.close()
    await runtime.close()
    with pytest.raises(RuntimeStateError, match="不能重新启动"):
        await runtime.start()


async def test_runtime_start_failure_closes_partial_store(tmp_path: Path) -> None:
    store = MagicMock()
    store.close = AsyncMock()
    runtime = PraxisRuntime(runtime_config(tmp_path), gateway=FakeGateway(), store=store)
    with (
        patch("praxis.runtime.build_guardrail_engine", side_effect=RuntimeError("bad rules")),
        pytest.raises(RuntimeError, match="bad rules"),
    ):
        await runtime.start()
    assert runtime.state.value == "failed"
    assert runtime.store is None
    store.close.assert_awaited_once()


async def test_runtime_builds_default_session_and_wires_subagent_tools(tmp_path: Path) -> None:
    runtime = PraxisRuntime(runtime_config(tmp_path), gateway=FakeGateway())
    await runtime.start()
    runner = await runtime.build_session()
    try:
        assert runner.session_id
        assert runner.registry.has_tool("spawn_subagent")  # type: ignore[attr-defined]
        assert runner.registry.has_tool("fork_subagents")  # type: ignore[attr-defined]
    finally:
        await runner.terminate()
        await runtime.close()


async def test_runtime_rejects_recursive_subagent_tool_delegation(tmp_path: Path) -> None:
    runtime = PraxisRuntime(runtime_config(tmp_path), gateway=FakeGateway())
    await runtime.start()
    registry = ToolRegistry()

    async def handler(_arguments: dict[str, Any]) -> str:
        return "unused"

    registry.register(
        ToolDefinition(
            name="spawn_again",
            description="recursive",
            parameters={"type": "object"},
            metadata=ToolMetadata(category="subagent"),
        ),
        handler,
    )
    with pytest.raises(RuntimeStateError, match="不能继续委派"):
        await runtime.build_subagent_session(
            SubagentSpec(task="x", tool_names=["spawn_again"]),
            registry,
        )
    await runtime.close()


async def test_session_stream_abort_properties_and_idempotent_context(tmp_path: Path) -> None:
    runner = FakeRunner()
    async with PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        session_builder=lambda _: asyncio.sleep(0, result=runner),
    ) as runtime:
        session = runtime.session()
        assert session.session_id is None
        assert session.status is SessionStatus.INITIALIZING
        await session.__aenter__()
        await session.__aenter__()
        assert session.session_id == "fake-session"
        events = [event async for event in session.run_stream("stream")]
        assert events[0].data["content"] == "stream"
        session.abort()
        assert runner.aborted
        await session.close()
        await session.close()
        with pytest.raises(SessionError, match="已终止"):
            await session.run("after-close")


async def test_health_covers_provider_failures_and_visual_ready(tmp_path: Path) -> None:
    gateway = FakeGateway()
    gateway.health = AsyncMock(side_effect=RuntimeError("down"))  # type: ignore[method-assign]
    gateway.supports_vision = MagicMock(return_value=True)  # type: ignore[method-assign]
    store = MagicMock()
    store.list_keys = AsyncMock(side_effect=RuntimeError("storage down"))
    store.close = AsyncMock()
    embedding = MagicMock()
    embedding.close = AsyncMock()
    config = runtime_config(tmp_path).model_copy(
        update={"verification": VerificationConfig(visual_enabled=True)}
    )
    runtime = PraxisRuntime(
        config,
        gateway=gateway,
        store=store,
        embedding_provider=embedding,
        session_builder=lambda _: asyncio.sleep(0, result=FakeRunner()),
    )
    await runtime.start()
    with patch("praxis.runtime.find_spec", return_value=object()):
        health = await runtime.health()
    assert health.status is HealthStatus.FAILED
    assert health.components["model"].status is HealthStatus.FAILED
    assert health.components["storage"].status is HealthStatus.FAILED
    assert health.components["embedding"].status is HealthStatus.READY
    assert health.components["visual"].status is HealthStatus.READY
    await runtime.close()
    embedding.close.assert_awaited_once()
