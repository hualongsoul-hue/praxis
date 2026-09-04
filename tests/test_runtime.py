"""应用级 Runtime 与 AgentSession 生命周期测试。"""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config import PraxisConfig
from praxis.config.schemas import (
    GatewayConfig,
    InputConfig,
    MCPConfig,
    MemoryConfig,
    ModelCapabilities,
    PersistenceConfig,
    SkillsConfig,
    VerificationConfig,
)
from praxis.exceptions import (
    ConcurrentSessionRunError,
    RuntimeCloseError,
    RuntimeStateError,
    SessionError,
)
from praxis.models.inputs import InputValue
from praxis.models.mcp import (
    MCPElicitationRequest,
    MCPElicitationResponse,
    MCPServerConfig,
    MCPTransportType,
)
from praxis.models.orchestrator import AgentEvent, AgentResponse
from praxis.models.responses import ModelResponse, ModelResponseChunk, Usage
from praxis.models.session import SessionStatus
from praxis.models.subagent import SubagentSpec
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.runtime import HealthStatus, PraxisRuntime
from praxis.session.core import Session
from praxis.tools.registry import ToolRegistry


class FakeGateway:
    def __init__(self) -> None:
        self.config = GatewayConfig()
        self.closed = False
        self.model_capabilities = ModelCapabilities()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        return ModelResponse(
            id="fake-response",
            content="unused",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=model or self.config.default_model,
            created=0,
        )

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

    def capabilities(self, model_name: str | None = None) -> ModelCapabilities:
        return self.model_capabilities

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

    async def run_turn(self, user_message: InputValue, **kwargs: Any) -> AgentResponse:
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        return AgentResponse(content=f"echo:{user_message}")

    async def run_turn_stream(
        self,
        user_message: InputValue,
        **kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]:
        yield AgentEvent(event_type="content", data={"content": user_message})

    def abort(self) -> None:
        self.aborted = True

    async def terminate(self) -> None:
        self.terminated = True
        self.status = SessionStatus.TERMINATED


class FakeStorageBackend:
    """Minimal byte backend used to exercise the public storage protocol."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], bytes] = {}
        self.closed = False

    async def save(self, namespace: str, key: str, data: bytes) -> None:
        self.values[(namespace, key)] = data

    async def save_if_absent(self, namespace: str, key: str, data: bytes) -> bool:
        identity = (namespace, key)
        if identity in self.values:
            return False
        self.values[identity] = data
        return True

    async def load(self, namespace: str, key: str) -> bytes | None:
        return self.values.get((namespace, key))

    async def delete(self, namespace: str, key: str) -> None:
        self.values.pop((namespace, key), None)

    async def list_keys(self, namespace: str, prefix: str | None = None) -> list[str]:
        return sorted(
            key
            for item_namespace, key in self.values
            if item_namespace == namespace and (prefix is None or key.startswith(prefix))
        )

    async def clear_namespace(self, namespace: str) -> int:
        identities = [item for item in self.values if item[0] == namespace]
        for identity in identities:
            del self.values[identity]
        return len(identities)

    async def close(self) -> None:
        self.closed = True


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
        own_gateway=True,
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=runner),
    ) as runtime:
        assert runtime.started
        async with runtime.session() as session:
            response = await session.run("hello")
            assert response.content == "echo:hello"
            assert session.status is SessionStatus.ACTIVE
        assert runner.terminated

    assert gateway.closed
    assert not runtime.started


async def test_public_model_gateway_runs_default_session_without_router_internals(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path).model_copy(update={
        "memory": MemoryConfig(background_enabled=False, dream_enabled=False),
    })
    async with PraxisRuntime(config, gateway=FakeGateway()) as runtime:
        async with runtime.session() as session:
            response = await session.run("protocol request")

    assert response.content == "unused"


async def test_runtime_accepts_and_owns_public_storage_backend(tmp_path: Path) -> None:
    backend = FakeStorageBackend()
    config = runtime_config(tmp_path).model_copy(update={
        "memory": MemoryConfig(background_enabled=False, dream_enabled=False),
    })

    async with PraxisRuntime(
        config,
        gateway=FakeGateway(),
        storage_backend=backend,
        own_storage_backend=True,
    ) as runtime:
        assert runtime.store is not None
        await runtime.store.save("contract", "key", {"ok": True})
        assert await runtime.store.load("contract", "key") == {"ok": True}

    assert backend.closed is True


async def test_runtime_sessions_share_one_resource_controller(tmp_path: Path) -> None:
    config = runtime_config(tmp_path).model_copy(update={
        "memory": MemoryConfig(background_enabled=False, dream_enabled=False),
    })
    async with PraxisRuntime(config, gateway=FakeGateway()) as runtime:
        async with runtime.session() as first, runtime.session() as second:
            assert isinstance(first.runner, Session)
            assert isinstance(second.runner, Session)
            assert first.runner.loop.coordinator.executor.resources is runtime.resources
            assert second.runner.loop.coordinator.executor.resources is runtime.resources


async def test_start_and_close_are_idempotent(tmp_path: Path) -> None:
    runtime = PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=FakeRunner()),
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
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=runner),
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
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=FakeRunner()),
    ) as runtime:
        session = runtime.session()
        with pytest.raises(SessionError, match="尚未启动"):
            await session.run("hello")


async def test_health_reports_local_embedding_degradation(tmp_path: Path) -> None:
    async with PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=FakeRunner()),
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
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=FakeRunner()),
    ) as runtime:
        health = await runtime.health()
        assert health.components["visual"].status is HealthStatus.DEGRADED
        assert "视觉能力" in health.components["visual"].detail


async def test_runtime_builds_owned_subagent_with_explicit_tool_subset(tmp_path: Path) -> None:
    parent_registry = ToolRegistry()

    async def handler(arguments: dict[str, Any]) -> str:
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

    config = runtime_config(tmp_path).model_copy(
        update={"inputs": InputConfig(max_attachment_bytes=123)}
    )
    runtime = PraxisRuntime(config, gateway=FakeGateway())
    await runtime.start()
    child = await runtime.build_subagent_session(
        SubagentSpec(task="isolated", tool_names=["allowed"], max_turns=3),
        parent_registry,
    )
    try:
        assert child.registry.list_tools() == ["allowed"]
        assert child.loop.config.max_turns == 3
        assert child.input_resolver.config.max_attachment_bytes == 123
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
    store.load = AsyncMock(return_value=None)
    store.save = AsyncMock()
    store.list_keys = AsyncMock(return_value=[])
    store.close = AsyncMock()
    gateway = FakeGateway()
    gateway.close = AsyncMock()  # type: ignore[method-assign]
    audit = MagicMock()
    audit.flush = AsyncMock()
    audit.close = AsyncMock()
    embedding = MagicMock()
    embedding.close = AsyncMock()
    runtime = PraxisRuntime(
        runtime_config(tmp_path),
        gateway=gateway,
        store=store,
        audit_sink=audit,
        embedding_provider=embedding,
        own_gateway=True,
        own_store=True,
        own_audit_sink=True,
        own_embedding_provider=True,
    )
    with (
        patch("praxis.runtime.build_guardrail_engine", side_effect=RuntimeError("bad rules")),
        pytest.raises(RuntimeError, match="bad rules"),
    ):
        await runtime.start()
    assert runtime.state.value == "failed"
    store.close.assert_awaited_once()
    gateway.close.assert_awaited_once()
    audit.flush.assert_awaited_once()
    audit.close.assert_awaited_once()
    embedding.close.assert_awaited_once()


async def test_runtime_close_collects_failures_and_closes_every_resource(
    tmp_path: Path,
) -> None:
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    store.save = AsyncMock()
    store.list_keys = AsyncMock(return_value=[])
    store.close = AsyncMock()
    gateway = FakeGateway()
    gateway.close = AsyncMock(side_effect=RuntimeError("gateway close failed"))  # type: ignore[method-assign]
    audit = MagicMock()
    audit.flush = AsyncMock(side_effect=RuntimeError("audit flush failed"))
    audit.close = AsyncMock()
    embedding = MagicMock()
    embedding.close = AsyncMock()
    runtime = PraxisRuntime(
        runtime_config(tmp_path),
        gateway=gateway,
        store=store,
        audit_sink=audit,
        embedding_provider=embedding,
        own_gateway=True,
        own_store=True,
        own_audit_sink=True,
        own_embedding_provider=True,
    )
    await runtime.start()

    with pytest.raises(RuntimeCloseError) as captured:
        await runtime.close()

    assert runtime.state.value == "closed"
    assert len(captured.value.failures) == 2
    store.close.assert_awaited_once()
    gateway.close.assert_awaited_once()
    audit.close.assert_awaited_once()
    embedding.close.assert_awaited_once()


async def test_session_close_cancels_active_run_promptly(tmp_path: Path) -> None:
    started = asyncio.Event()
    runner = FakeRunner(started, asyncio.Event())
    runtime = PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=runner),
    )
    await runtime.start()
    session = runtime.session()
    await session.__aenter__()
    running = asyncio.create_task(session.run("blocked"))
    await started.wait()

    await asyncio.wait_for(session.close(), timeout=0.5)

    assert running.cancelled()
    assert runner.terminated
    await runtime.close()


async def test_failed_session_close_can_be_retried(tmp_path: Path) -> None:
    runner = FakeRunner()
    runner.terminate = AsyncMock(side_effect=[RuntimeError("first close failed"), None])  # type: ignore[method-assign]
    async with PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=runner),
    ) as runtime:
        session = runtime.session()
        await session.__aenter__()
        with pytest.raises(RuntimeError, match="first close failed"):
            await session.close()
        assert session.closed is False

        await session.close()

        assert session.closed is True
        assert runner.terminate.await_count == 2


async def test_runtime_builds_default_session_and_wires_subagent_tools(tmp_path: Path) -> None:
    runtime = PraxisRuntime(runtime_config(tmp_path), gateway=FakeGateway())
    await runtime.start()
    runner = await runtime.build_session()
    try:
        assert runner.session_id
        assert runner.registry.has_tool("spawn_subagent")  # type: ignore[attr-defined]
        assert runner.registry.has_tool("fork_subagents")  # type: ignore[attr-defined]
        assert runner.skill_manager is not None  # type: ignore[attr-defined]
        assert runner.verifier_registry is not None  # type: ignore[attr-defined]
    finally:
        await runner.terminate()
        await runtime.close()


async def test_runtime_rejects_recursive_subagent_tool_delegation(tmp_path: Path) -> None:
    runtime = PraxisRuntime(runtime_config(tmp_path), gateway=FakeGateway())
    await runtime.start()
    registry = ToolRegistry()

    async def handler(arguments: dict[str, Any]) -> str:
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
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=runner),
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
    gateway.model_capabilities = ModelCapabilities(image=True)
    store = MagicMock()
    store.load = AsyncMock(return_value=None)
    store.save = AsyncMock()

    async def health_sensitive_list(namespace: str, prefix: str = "") -> list[str]:
        if namespace == "_praxis_health":
            raise RuntimeError("storage down")
        return []

    store.list_keys = AsyncMock(side_effect=health_sensitive_list)
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
        own_embedding_provider=True,
        session_builder=lambda runtime_instance: asyncio.sleep(0, result=FakeRunner()),
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


async def test_runtime_wires_configured_skills_verification_mcp_and_callbacks(
    tmp_path: Path,
) -> None:
    manager = MagicMock()
    manager.list_connected_servers.return_value = ["local"]
    elicitation_handler = AsyncMock()
    sampling_review_handler = AsyncMock(return_value=True)
    config = runtime_config(tmp_path).model_copy(
        update={
            "skills": SkillsConfig(
                skill_paths=[str(Path("examples/skills"))],
                auto_discover=True,
                max_skills_in_context=2,
            ),
            "verification": VerificationConfig(
                computational_enabled=False,
                inferential_enabled=False,
                visual_enabled=False,
            ),
            "mcp": MCPConfig(
                enabled=True,
                connect_timeout=7,
                sampling_enabled=True,
                servers=[
                    MCPServerConfig(
                        name="local",
                        command="python",
                        timeout=30,
                    )
                ],
            ),
        }
    )
    runtime = PraxisRuntime(
        config,
        gateway=FakeGateway(),
        mcp_elicitation_handler=elicitation_handler,
        mcp_sampling_review_handler=sampling_review_handler,
    )

    with patch(
        "praxis.tools.mcp.wiring.connect_mcp_servers",
        AsyncMock(return_value=manager),
    ) as connect:
        await runtime.start()
        runner = await runtime.build_session()
        try:
            assert runner.skill_manager is not None  # type: ignore[attr-defined]
            assert runner.skill_manager.skills  # type: ignore[attr-defined]
            assert runner.registry.has_tool("load_skill")  # type: ignore[attr-defined]
            registry = runner.verifier_registry  # type: ignore[attr-defined]
            assert registry is not None
            assert registry.computational_enabled is False
            assert registry.inferential_enabled is False
            assert runner.mcp_manager is manager  # type: ignore[attr-defined]
            assert runner.mcp_sampling_manager.review_handler is sampling_review_handler  # type: ignore[attr-defined]
            assert runner.mcp_elicitation_manager.handler is elicitation_handler  # type: ignore[attr-defined]
            connected_config = connect.await_args.args[1][0]
            assert connected_config.timeout == 7
            health = await runtime.health()
            assert health.components["mcp"].status is HealthStatus.DEGRADED
        finally:
            await runner.terminate()
            await runtime.close()


async def test_runtime_uses_real_mcp_stdio_capabilities_and_closes_them(
    tmp_path: Path,
) -> None:
    reviews: list[tuple[int, str]] = []
    elicitations: list[str] = []

    async def review(messages: list[dict[str, Any]], server_name: str) -> bool:
        reviews.append((len(messages), server_name))
        return True

    async def elicit(request: MCPElicitationRequest) -> MCPElicitationResponse:
        elicitations.append(request.message)
        return MCPElicitationResponse(accepted=True, data={"answer": "approved"})

    server_path = Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
    config = runtime_config(tmp_path).model_copy(
        update={
            "mcp": MCPConfig(
                enabled=True,
                sampling_enabled=True,
                servers=[
                    MCPServerConfig(
                        name="runtime-stdio",
                        transport=MCPTransportType.STDIO,
                        command=sys.executable,
                        args=[str(server_path)],
                    )
                ],
            )
        }
    )

    async with PraxisRuntime(
        config,
        gateway=FakeGateway(),
        mcp_elicitation_handler=elicit,
        mcp_sampling_review_handler=review,
    ) as runtime:
        async with runtime.session() as agent_session:
            assert isinstance(agent_session.runner, Session)
            runner = agent_session.runner
            manager = runner.mcp_manager
            assert manager.list_connected_servers() == ["runtime-stdio"]
            assert (
                await manager.tools_bridge.call_tool(
                    "runtime-stdio",
                    "echo",
                    {"message": "runtime"},
                )
                == "runtime"
            )
            assert (
                await manager.resources_bridge.read_resource(
                    "runtime-stdio",
                    "test://fixture",
                )
            ).text == "fixture resource"
            prompt = await manager.prompts_bridge.get_prompt(
                "runtime-stdio",
                "greet",
                {"name": "Praxis"},
            )
            assert prompt[0].content == "Hello, Praxis!"
            assert (
                await manager.tools_bridge.call_tool(
                    "runtime-stdio",
                    "request_sampling",
                    {},
                )
                == "unused"
            )
            assert (
                await manager.tools_bridge.call_tool(
                    "runtime-stdio",
                    "request_elicitation",
                    {},
                )
                == "approved"
            )
            assert (await runtime.health()).components["mcp"].status is HealthStatus.READY

        assert manager.list_connected_servers() == []
        assert not runner.registry.has_tool("mcp_runtime-stdio_echo")

    assert reviews == [(1, "runtime-stdio")]
    assert elicitations == ["Provide approval"]


async def test_runtime_injects_owned_embedding_provider_into_memory(tmp_path: Path) -> None:
    embedding = MagicMock()
    embedding.embed = AsyncMock(return_value=[1.0, 0.0])
    embedding.close = AsyncMock()
    runtime = PraxisRuntime(
        runtime_config(tmp_path),
        gateway=FakeGateway(),
        embedding_provider=embedding,
        own_embedding_provider=True,
    )
    await runtime.start()
    runner = await runtime.build_session()
    try:
        memory = runner.memory  # type: ignore[attr-defined]
        assert memory is not None
        assert memory.vector_store.embed_func == embedding.embed
    finally:
        await runner.terminate()
        await runtime.close()
    embedding.close.assert_awaited_once()
