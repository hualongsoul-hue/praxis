"""应用级 PraxisRuntime 与并发安全的 AgentSession。"""

import asyncio
import math
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, aclosing
from datetime import UTC, datetime
from importlib.util import find_spec
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from praxis.config.settings import PraxisConfig
from praxis.exceptions import (
    ConcurrentSessionRunError,
    RuntimeCloseError,
    RuntimeStateError,
    SessionError,
)
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine, build_guardrail_engine
from praxis.lifecycle import AsyncResourceOwner, TaskSupervisor
from praxis.memory.store import ScopedMemoryStore
from praxis.memory.vector import VectorStore
from praxis.models.inputs import InputValue
from praxis.models.mcp import MCPElicitationRequest, MCPElicitationResponse
from praxis.models.orchestrator import AgentEvent, AgentResponse
from praxis.models.runtime import ComponentHealth, HealthStatus, RuntimeHealth, RuntimeState
from praxis.models.session import SessionStatus
from praxis.models.subagent import SubagentSpec
from praxis.models.verification import VerificationStatus
from praxis.persistence.store import PersistenceStore, StorageBackend, create_store
from praxis.protocols import ApprovalHandler, AuditSink, EmbeddingProvider, ModelGateway
from praxis.resources import ResourceController
from praxis.session.core import Session, SessionFactory
from praxis.skills.manager import build_skill_manager
from praxis.telemetry.audit import AuditService
from praxis.telemetry.metrics import MetricsCollector, use_metrics
from praxis.tools.registry import ToolRegistry
from praxis.verification.registry import VerifierRegistry

MCPElicitationHandler = Callable[
    [MCPElicitationRequest],
    Awaitable[MCPElicitationResponse],
]
MCPSamplingReviewHandler = Callable[
    [list[dict[str, Any]], str],
    Awaitable[bool],
]


class SessionRunner(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def status(self) -> SessionStatus: ...

    async def run_turn(self, user_input: InputValue, **kwargs: Any) -> AgentResponse: ...

    def run_turn_stream(
        self,
        user_input: InputValue,
        **kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]: ...

    def abort(self) -> None: ...

    async def terminate(self) -> None: ...


SessionBuilder = Callable[["PraxisRuntime"], Awaitable[SessionRunner]]


async def probe_component_health(
    check: Callable[[], Awaitable[str]],
    *,
    required: bool,
    failure_status: HealthStatus = HealthStatus.FAILED,
) -> ComponentHealth:
    """Run one bounded probe and return metadata without leaking exception details."""
    started = perf_counter()
    probed_at = datetime.now(UTC)
    try:
        detail = await check()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        reason = type(exc).__name__
        return ComponentHealth(
            status=failure_status,
            detail=f"探针失败: {reason}",
            reason=reason,
            required=required,
            last_probe_at=probed_at,
            latency_ms=(perf_counter() - started) * 1000,
        )
    return ComponentHealth(
        status=HealthStatus.READY,
        detail=detail,
        reason=detail,
        required=required,
        last_probe_at=probed_at,
        latency_ms=(perf_counter() - started) * 1000,
    )


class PraxisRuntime:
    """持有一个 Agent 服务实例的全部资源和生命周期。"""

    def __init__(
        self,
        config: PraxisConfig,
        *,
        gateway: ModelGateway | None = None,
        store: PersistenceStore | None = None,
        audit_sink: AuditSink | None = None,
        approval_handler: ApprovalHandler | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        session_builder: SessionBuilder | None = None,
        mcp_elicitation_handler: MCPElicitationHandler | None = None,
        mcp_sampling_review_handler: MCPSamplingReviewHandler | None = None,
        own_gateway: bool = False,
        own_store: bool = False,
        own_audit_sink: bool = False,
        own_embedding_provider: bool = False,
        vector_store: VectorStore | None = None,
        own_vector_store: bool = False,
        storage_backend: StorageBackend | None = None,
        own_storage_backend: bool = False,
    ) -> None:
        if store is not None and storage_backend is not None:
            raise ValueError("store 与 storage_backend 不能同时提供")
        if storage_backend is not None:
            store = PersistenceStore(storage_backend, own_backend=own_storage_backend)
        self.config = config.model_copy(deep=True)
        self.gateway = gateway
        self.store = store
        self.audit_sink = audit_sink
        self.approval_handler = approval_handler
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.metrics = MetricsCollector(enabled=self.config.telemetry.metrics_enabled)
        self.supervisor = TaskSupervisor()
        self.resources = ResourceController(
            self.config.subagent,
            supervisor=self.supervisor,
            max_concurrent_readonly=self.config.tools.max_concurrent_readonly,
        )
        self.session_builder = session_builder
        self.mcp_elicitation_handler = mcp_elicitation_handler
        self.mcp_sampling_review_handler = mcp_sampling_review_handler
        self.guardrails: GuardrailEngine | None = None
        self.sessions: set[AgentSession] = set()
        self.subagent_sessions: set[Session] = set()
        self.runtime_state = RuntimeState.NEW
        self.lifecycle_lock = asyncio.Lock()
        self.resource_owner = AsyncResourceOwner()
        self.gateway_owned = gateway is None or own_gateway
        self.store_owned = storage_backend is not None or store is None or own_store
        self.audit_sink_owned = audit_sink is None or own_audit_sink
        self.embedding_provider_owned = own_embedding_provider
        self.vector_store_owned = vector_store is None or own_vector_store

    @property
    def state(self) -> RuntimeState:
        return self.runtime_state

    @property
    def started(self) -> bool:
        return self.runtime_state is RuntimeState.ACTIVE

    async def __aenter__(self) -> "PraxisRuntime":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.close()

    async def start(self) -> None:
        async with self.lifecycle_lock:
            if self.runtime_state is RuntimeState.ACTIVE:
                return
            if self.runtime_state in {
                RuntimeState.STOPPING,
                RuntimeState.CLOSED,
                RuntimeState.FAILED,
            }:
                raise RuntimeStateError("已关闭的 Runtime 不能重新启动")
            self.runtime_state = RuntimeState.STARTING
            try:
                if self.store is None:
                    self.store = await create_store(self.config.persistence)
                if self.store_owned:
                    self.resource_owner.register("storage", self.store.close)
                if self.gateway is None:
                    self.gateway = GatewayRouter(self.config.gateway)
                if self.gateway_owned:
                    self.resource_owner.register("gateway", self.gateway.close)
                if self.embedding_provider is not None and self.embedding_provider_owned:
                    self.resource_owner.register("embedding", self.embedding_provider.close)
                if self.vector_store is None:
                    memory_config = self.config.memory
                    self.vector_store = VectorStore(
                        ScopedMemoryStore(self.store),
                        embed_func=(
                            self.embedding_provider.embed
                            if self.embedding_provider is not None
                            else None
                        ),
                        api_base=memory_config.embedding_api_base,
                        api_key=(
                            os.getenv(memory_config.embedding_api_key_env, "")
                            if memory_config.embedding_api_key_env
                            else ""
                        ),
                        model=memory_config.embedding_model,
                        timeout=memory_config.embedding_timeout,
                        dimensions=memory_config.embedding_dimensions,
                        provider_id=(
                            memory_config.embedding_model
                            or (
                                "injected-provider"
                                if self.embedding_provider is not None
                                else "local-lexical-v1"
                            )
                        ),
                        max_memories=memory_config.max_memories,
                    )
                if self.vector_store_owned:
                    self.resource_owner.register("memory-index", self.vector_store.aclose)
                await self.vector_store.start()
                if self.audit_sink is None:
                    self.audit_sink = AuditService(
                        self.store,
                        enabled=self.config.telemetry.audit_enabled,
                    )
                if self.audit_sink_owned:
                    self.resource_owner.register("audit", self.close_audit_sink)
                self.guardrails = build_guardrail_engine(
                    self.config.guardrails,
                    audit_sink=self.audit_sink,
                )
                self.runtime_state = RuntimeState.ACTIVE
            except BaseException as start_error:
                self.runtime_state = RuntimeState.FAILED
                cleanup_failures = await self.resource_owner.close()
                for failure in cleanup_failures:
                    start_error.add_note(
                        f"启动回滚失败: {type(failure).__name__}: {failure}"
                    )
                raise

    async def close_audit_sink(self) -> None:
        """Flush and close an owned audit sink without skipping close on flush failure."""
        if self.audit_sink is None:
            return
        flush_error: BaseException | None = None
        try:
            await self.audit_sink.flush()
        except BaseException as exc:
            flush_error = exc
        try:
            await self.audit_sink.close()
        except BaseException as close_error:
            if flush_error is not None:
                close_error.add_note(
                    f"审计 flush 同时失败: {type(flush_error).__name__}: {flush_error}"
                )
            raise
        if flush_error is not None:
            raise flush_error

    def session(self) -> "AgentSession":
        if not self.started:
            raise RuntimeStateError("Runtime 尚未启动")
        return AgentSession(self)

    @property
    def owned_subagent_count(self) -> int:
        """Return the number of isolated child sessions owned by this Runtime."""
        return len(self.subagent_sessions)

    async def build_session(self) -> SessionRunner:
        if self.session_builder is not None:
            return await self.session_builder(self)
        if self.store is None or self.gateway is None or self.guardrails is None:
            raise RuntimeStateError("Runtime 组件尚未就绪")
        factory = SessionFactory(
            store=self.store,
            session_config=self.config.session,
            orchestrator_config=self.config.orchestrator,
            context_config=self.config.context,
            input_config=self.config.inputs,
            memory_config=self.config.memory,
            recovery_config=self.config.recovery,
            approval_handler=self.approval_handler,
            audit_sink=self.audit_sink,
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            resources=self.resources,
        )
        session = await factory.create_session(
            guardrails=self.guardrails,
            gateway=self.gateway,
            model=self.config.gateway.default_model,
            tools_config=self.config.tools,
        )
        await self.configure_session_extensions(session)
        from praxis.subagent.tools import wire_subagent

        wire_subagent(
            session=session,
            store=self.store,
            guardrails=self.guardrails,
            gateway=self.gateway,
            orchestrator_config=self.config.orchestrator,
            context_config=self.config.context,
            input_config=self.config.inputs,
            subagent_config=self.config.subagent,
            model=self.config.gateway.default_model,
            runtime=self,
            resource_controller=self.resources,
        )
        return session

    async def configure_session_extensions(self, session: Session) -> None:
        """Attach configured skills, verification, and optional MCP to one session."""

        if self.store is None or self.gateway is None:
            raise RuntimeStateError("Runtime 扩展组件尚未就绪")

        skill_manager = await build_skill_manager(
            self.config.skills,
            session.registry,
            self.store,
        )
        skill_manager.register_disclosure_tools()
        session.skill_manager = skill_manager
        session.loop.skill_manager = skill_manager

        verifier_registry = VerifierRegistry.from_config(
            self.config.verification,
            gateway=self.gateway,
            policy=session.loop.coordinator.executor.sandbox,
        )
        session.verifier_registry = verifier_registry
        session.loop.verifier_registry = verifier_registry

        if self.config.mcp.enabled:
            await self.configure_session_mcp(session)

    async def configure_session_mcp(self, session: Session) -> None:
        """Connect configured MCP servers with session-owned transport lifetimes."""

        if self.gateway is None:
            raise RuntimeStateError("模型网关尚未就绪")

        from praxis.tools.mcp.elicitation import ElicitationManager
        from praxis.tools.mcp.sampling import SamplingManager
        from praxis.tools.mcp.wiring import connect_mcp_servers

        sampling_manager = (
            SamplingManager(self.gateway) if self.config.mcp.sampling_enabled else None
        )
        if (
            sampling_manager is not None
            and self.mcp_sampling_review_handler is not None
        ):
            sampling_manager.set_review_handler(self.mcp_sampling_review_handler)

        elicitation_manager = ElicitationManager()
        if self.mcp_elicitation_handler is not None:
            elicitation_manager.set_handler(self.mcp_elicitation_handler)

        server_configs = [
            server.model_copy(
                update={
                    "timeout": min(server.timeout, self.config.mcp.connect_timeout),
                }
            )
            for server in self.config.mcp.servers
        ]
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            manager = await connect_mcp_servers(
                session.registry,
                server_configs,
                stack,
                sampling_manager=sampling_manager,
                elicitation_manager=elicitation_manager,
                task_supervisor=self.supervisor,
            )
        except BaseException:
            await stack.aclose()
            raise

        session.mcp_manager = manager
        session.mcp_sampling_manager = sampling_manager
        session.mcp_elicitation_manager = elicitation_manager
        session.attach_mcp_stack(stack)

    async def build_subagent_session(
        self,
        spec: SubagentSpec,
        parent_registry: ToolRegistry,
    ) -> Session:
        """Create an isolated child session with only explicitly delegated tools."""
        if not self.started:
            raise RuntimeStateError("Runtime 尚未启动")
        if self.store is None or self.gateway is None or self.guardrails is None:
            raise RuntimeStateError("Runtime 组件尚未就绪")

        child_registry = ToolRegistry()
        for tool_name in dict.fromkeys(spec.tool_names):
            entry = parent_registry.get_entry(tool_name)
            if entry.definition.metadata.category == "subagent":
                raise RuntimeStateError("子代理不能继续委派子代理工具")
            child_registry.register(entry.definition, entry.handler)

        factory = SessionFactory(
            store=self.store,
            session_config=self.config.session.model_copy(update={"auto_checkpoint": False}),
            orchestrator_config=self.config.orchestrator.model_copy(
                update={"max_turns": spec.max_turns}
            ),
            context_config=self.config.context,
            input_config=self.config.inputs,
            memory_config=self.config.memory,
            recovery_config=self.config.recovery,
            approval_handler=self.approval_handler,
            audit_sink=self.audit_sink,
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            resources=self.resources,
        )
        child = await factory.create_session(
            guardrails=self.guardrails,
            gateway=self.gateway,
            registry=child_registry,
            model=self.config.gateway.default_model,
            tools_config=self.config.tools,
            include_builtins=False,
        )
        self.subagent_sessions.add(child)
        return child

    async def release_subagent_session(self, session: Session) -> None:
        """Terminate and forget a child session; safe to call more than once."""
        try:
            if session in self.subagent_sessions:
                session.abort()
                await session.terminate()
        finally:
            self.subagent_sessions.discard(session)

    def register_session(self, session: "AgentSession") -> None:
        self.sessions.add(session)

    def unregister_session(self, session: "AgentSession") -> None:
        self.sessions.discard(session)

    async def health(self) -> RuntimeHealth:
        components: dict[str, ComponentHealth] = {}
        if not self.started or self.gateway is None or self.store is None:
            return RuntimeHealth(
                status=HealthStatus.FAILED,
                runtime_state=self.runtime_state,
                components={
                    "runtime": ComponentHealth(
                        status=HealthStatus.FAILED,
                        detail="Runtime 未就绪",
                    ),
                },
            )
        gateway = self.gateway
        store = self.store

        probe_timeout = self.config.gateway.health_probe_timeout

        async def check_model() -> str:
            async with asyncio.timeout(probe_timeout):
                if not await gateway.health():
                    raise RuntimeError("model unavailable")
            return "模型端点实时探测通过"

        components["model"] = await probe_component_health(
            check_model,
            required=True,
        )

        async def check_storage() -> str:
            key = f"probe-{uuid4().hex}"
            payload = {"status": "ok"}
            saved = False
            async with asyncio.timeout(probe_timeout):
                try:
                    await store.save("runtime_health", key, payload)
                    saved = True
                    loaded = await store.load("runtime_health", key)
                    if loaded != payload:
                        raise RuntimeError("storage round-trip mismatch")
                finally:
                    if saved:
                        await store.delete("runtime_health", key)
            return "存储读写删除探测通过"

        components["storage"] = await probe_component_health(
            check_storage,
            required=True,
        )

        async def check_embedding() -> str:
            if self.vector_store is None:
                raise RuntimeError("embedding store unavailable")
            async with asyncio.timeout(self.config.memory.embedding_timeout):
                vector = await self.vector_store.embed_func("praxis health probe")
            self.vector_store.validate_vector(vector)
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("embedding contains non-finite values")
            return f"嵌入探测通过，维度 {len(vector)}"

        embedding_is_local = (
            self.embedding_provider is None
            and self.config.memory.embedding_api_base is None
        )
        embedding_health = await probe_component_health(
            check_embedding,
            required=False,
            failure_status=HealthStatus.DEGRADED,
        )
        if embedding_is_local and embedding_health.status is HealthStatus.READY:
            embedding_health = embedding_health.model_copy(
                update={
                    "status": HealthStatus.DEGRADED,
                    "detail": "本地词法嵌入探测通过；未配置语义嵌入 Provider",
                    "reason": "local lexical fallback",
                }
            )
        components["embedding"] = embedding_health

        if not self.config.verification.visual_enabled:
            components["visual"] = ComponentHealth(
                status=HealthStatus.READY,
                detail="视觉验证未启用",
                required=False,
            )
        elif find_spec("playwright") is None:
            components["visual"] = ComponentHealth(
                status=HealthStatus.DEGRADED,
                detail="缺少可选依赖 praxis[visual]",
                required=False,
            )
        elif not self.gateway.capabilities().image:
            components["visual"] = ComponentHealth(
                status=HealthStatus.DEGRADED,
                detail="默认模型未声明视觉能力",
                required=False,
            )
        else:
            components["visual"] = ComponentHealth(
                status=HealthStatus.READY,
                detail="视觉验证能力就绪",
                required=False,
            )

        concrete_sessions = [
            session.runner
            for session in self.sessions
            if isinstance(session.runner, Session)
        ]
        skills_loaded = any(
            session.skill_manager is not None
            for session in concrete_sessions
        )
        components["skills"] = ComponentHealth(
            status=HealthStatus.READY,
            detail=(
                "技能系统已装配"
                if skills_loaded
                else "技能系统将在创建会话时装配"
            ),
            required=False,
        )
        async def check_verification() -> str:
            config = self.config.verification
            if not config.computational_enabled:
                return "计算验证已禁用"
            registry = VerifierRegistry.from_config(config, gateway=self.gateway)
            schema_entry = registry.get("schema")
            if schema_entry is None:
                raise RuntimeError("schema verifier missing")
            async with asyncio.timeout(probe_timeout):
                result = await schema_entry.verifier.verify(
                    {
                        "data": {"healthy": True},
                        "schema": {
                            "type": "object",
                            "required": ["healthy"],
                            "properties": {"healthy": {"const": True}},
                        },
                    }
                )
            if result.status is not VerificationStatus.PASS:
                raise RuntimeError("schema verifier self-test failed")
            return "Schema 验证器自检通过"

        components["verification"] = await probe_component_health(
            check_verification,
            required=self.config.verification.computational_enabled,
        )

        if not self.config.mcp.enabled:
            components["mcp"] = ComponentHealth(
                status=HealthStatus.READY,
                detail="MCP 未启用",
                required=False,
            )
        else:
            expected_servers = {server.name for server in self.config.mcp.servers}
            healthy_session_count = 0
            for concrete_session in concrete_sessions:
                manager = concrete_session.mcp_manager
                session_ready = manager is not None
                for server_name in sorted(expected_servers):
                    server_status = (
                        manager.get_server_status(server_name)
                        if manager is not None
                        else None
                    )
                    server_ready = server_status is not None and server_status.value == "connected"
                    components[
                        f"mcp:{concrete_session.session_id}:{server_name}"
                    ] = ComponentHealth(
                        status=(
                            HealthStatus.READY if server_ready else HealthStatus.DEGRADED
                        ),
                        detail=(
                            f"MCP Server 状态: {server_status.value}"
                            if server_status is not None
                            else "MCP Manager 未装配"
                        ),
                        reason=(server_status.value if server_status is not None else "missing"),
                        required=False,
                    )
                    session_ready = session_ready and server_ready
                if session_ready:
                    healthy_session_count += 1
            mcp_ready = bool(concrete_sessions) and (
                healthy_session_count == len(concrete_sessions)
            )
            components["mcp"] = ComponentHealth(
                status=HealthStatus.READY if mcp_ready else HealthStatus.DEGRADED,
                detail=(
                    f"{healthy_session_count}/{len(concrete_sessions)} 个活动会话的 MCP 连接完整"
                ),
                reason="all session connections ready" if mcp_ready else "session connection missing",
                required=False,
            )

        components["background_tasks"] = ComponentHealth(
            status=HealthStatus.READY if self.supervisor.healthy else HealthStatus.FAILED,
            detail=(
                "后台任务正常"
                if self.supervisor.healthy
                else f"{len(self.supervisor.failures)} 个后台任务失败"
            ),
        )

        required_failed = any(
            item.required and item.status is HealthStatus.FAILED
            for item in components.values()
        )
        degraded = any(item.status is HealthStatus.DEGRADED for item in components.values())
        status = (
            HealthStatus.FAILED
            if required_failed
            else HealthStatus.DEGRADED
            if degraded
            else HealthStatus.READY
        )
        return RuntimeHealth(
            status=status,
            runtime_state=self.runtime_state,
            components=components,
        )

    async def close(self) -> None:
        async with self.lifecycle_lock:
            if self.runtime_state is RuntimeState.CLOSED:
                return
            self.runtime_state = RuntimeState.STOPPING
            failures: list[BaseException] = []
            sessions = list(self.sessions)
            for session in sessions:
                try:
                    session.abort()
                except BaseException as exc:
                    failures.append(exc)
            try:
                await self.resources.close()
            except BaseException as exc:
                failures.append(exc)
            try:
                await self.supervisor.close()
            except BaseException as exc:
                failures.append(exc)
            child_sessions = list(self.subagent_sessions)
            if child_sessions:
                results = await asyncio.gather(
                    *(self.release_subagent_session(session) for session in child_sessions),
                    return_exceptions=True,
                )
                failures.extend(item for item in results if isinstance(item, BaseException))
            if sessions:
                results = await asyncio.gather(
                    *(session.close() for session in sessions),
                    return_exceptions=True,
                )
                failures.extend(item for item in results if isinstance(item, BaseException))
            failures.extend(await self.resource_owner.close())
            self.runtime_state = RuntimeState.CLOSED
            if failures:
                raise RuntimeCloseError(tuple(failures))


class AgentSession:
    """异步上下文管理的会话；同一实例禁止并发执行轮次。"""

    def __init__(self, runtime: PraxisRuntime) -> None:
        self.runtime = runtime
        self.runner: SessionRunner | None = None
        self.run_lock = asyncio.Lock()
        self.close_lock = asyncio.Lock()
        self.closed = False
        self.closing = False
        self.active_task: asyncio.Task[Any] | None = None

    @property
    def session_id(self) -> str | None:
        return self.runner.session_id if self.runner is not None else None

    @property
    def status(self) -> SessionStatus:
        if self.runner is None:
            return SessionStatus.INITIALIZING
        return self.runner.status

    async def __aenter__(self) -> "AgentSession":
        if self.runner is None:
            with use_metrics(self.runtime.metrics):
                self.runner = await self.runtime.build_session()
            self.runtime.register_session(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.close()

    def require_runner(self) -> SessionRunner:
        if self.runner is None:
            raise SessionError("AgentSession 尚未启动，请使用 async with")
        if self.closed or self.closing or self.runner.status is SessionStatus.TERMINATED:
            raise SessionError("AgentSession 已终止")
        return self.runner

    async def run(self, user_input: InputValue, **kwargs: Any) -> AgentResponse:
        runner = self.require_runner()
        if self.run_lock.locked():
            raise ConcurrentSessionRunError("同一 AgentSession 不能并发执行两个轮次")
        async with self.run_lock:
            with use_metrics(self.runtime.metrics):
                task = asyncio.create_task(
                    runner.run_turn(user_input, **kwargs),
                    name=f"praxis-session-{runner.session_id}",
                )
                self.active_task = task
                try:
                    return await task
                finally:
                    self.active_task = None

    async def run_stream(
        self,
        user_input: InputValue,
        **kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]:
        runner = self.require_runner()
        if self.run_lock.locked():
            raise ConcurrentSessionRunError("同一 AgentSession 不能并发执行两个轮次")
        async with self.run_lock:
            with use_metrics(self.runtime.metrics):
                self.active_task = asyncio.current_task()
                try:
                    stream = runner.run_turn_stream(user_input, **kwargs)
                    async with aclosing(stream):
                        async for event in stream:
                            yield event
                finally:
                    self.active_task = None

    def abort(self) -> None:
        if self.runner is None:
            raise SessionError("AgentSession 尚未启动，请使用 async with")
        self.runner.abort()
        task = self.active_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def close(self) -> None:
        async with self.close_lock:
            if self.closed:
                return
            self.closing = True
            try:
                if self.runner is not None:
                    self.runner.abort()
                    task = self.active_task
                    if task is not None and task is not asyncio.current_task() and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    async with self.run_lock:
                        await self.runner.terminate()
                self.closed = True
                self.runtime.unregister_session(self)
            finally:
                self.closing = False


__all__ = [
    "AgentSession",
    "ComponentHealth",
    "HealthStatus",
    "MCPElicitationHandler",
    "MCPSamplingReviewHandler",
    "PraxisRuntime",
    "RuntimeHealth",
    "RuntimeState",
]
