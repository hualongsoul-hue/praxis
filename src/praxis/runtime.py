"""应用级 PraxisRuntime 与并发安全的 AgentSession。"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from importlib.util import find_spec
from typing import Any, Protocol, cast

from praxis.config.settings import PraxisConfig
from praxis.exceptions import ConcurrentSessionRunError, RuntimeStateError, SessionError
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine, build_guardrail_engine
from praxis.lifecycle import TaskSupervisor
from praxis.models.inputs import InputValue
from praxis.models.orchestrator import AgentEvent, AgentResponse
from praxis.models.runtime import ComponentHealth, HealthStatus, RuntimeHealth, RuntimeState
from praxis.models.session import SessionStatus
from praxis.models.subagent import SubagentSpec
from praxis.persistence.store import PersistenceStore, create_store
from praxis.protocols import ApprovalHandler, AuditSink, EmbeddingProvider, ModelGateway
from praxis.session.core import Session, SessionFactory
from praxis.telemetry.audit import AuditService
from praxis.telemetry.metrics import MetricsCollector, use_metrics
from praxis.tools.registry import ToolRegistry


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
    ) -> None:
        self.config = config.model_copy(deep=True)
        self.gateway = gateway
        self.store = store
        self.audit_sink = audit_sink
        self.approval_handler = approval_handler
        self.embedding_provider = embedding_provider
        self.metrics = MetricsCollector()
        self.supervisor = TaskSupervisor()
        self.session_builder = session_builder
        self.guardrails: GuardrailEngine | None = None
        self.sessions: set[AgentSession] = set()
        self.subagent_sessions: set[Session] = set()
        self.runtime_state = RuntimeState.NEW
        self.lifecycle_lock = asyncio.Lock()

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
            if self.runtime_state in {RuntimeState.STOPPING, RuntimeState.CLOSED}:
                raise RuntimeStateError("已关闭的 Runtime 不能重新启动")
            self.runtime_state = RuntimeState.STARTING
            try:
                if self.store is None:
                    self.store = await create_store(self.config.persistence)
                if self.gateway is None:
                    self.gateway = GatewayRouter(self.config.gateway)
                if self.audit_sink is None:
                    self.audit_sink = AuditService(
                        self.store,
                        enabled=self.config.telemetry.audit_enabled,
                    )
                self.guardrails = build_guardrail_engine(
                    self.config.guardrails,
                    audit_sink=self.audit_sink,
                )
                self.runtime_state = RuntimeState.ACTIVE
            except BaseException:
                self.runtime_state = RuntimeState.FAILED
                if self.store is not None:
                    await self.store.close()
                    self.store = None
                raise

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
        )
        session = await factory.create_session(
            guardrails=self.guardrails,
            gateway=cast(Any, self.gateway),
            model=self.config.gateway.default_model,
            tools_config=self.config.tools,
        )
        from praxis.subagent.tools import wire_subagent

        wire_subagent(
            session=session,
            store=self.store,
            guardrails=self.guardrails,
            gateway=cast(Any, self.gateway),
            orchestrator_config=self.config.orchestrator,
            context_config=self.config.context,
            input_config=self.config.inputs,
            subagent_config=self.config.subagent,
            model=self.config.gateway.default_model,
            runtime=self,
            supervisor=self.supervisor,
        )
        return session

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
        )
        child = await factory.create_session(
            guardrails=self.guardrails,
            gateway=cast(Any, self.gateway),
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

        try:
            model_ready = await self.gateway.health()
        except Exception as exc:
            model_ready = False
            model_detail = f"模型网关探针失败: {type(exc).__name__}"
        else:
            model_detail = "模型网关配置就绪" if model_ready else "模型网关不可用"
        components["model"] = ComponentHealth(
            status=HealthStatus.READY if model_ready else HealthStatus.FAILED,
            detail=model_detail,
        )

        try:
            await self.store.list_keys("_praxis_health")
        except Exception as exc:
            components["storage"] = ComponentHealth(
                status=HealthStatus.FAILED,
                detail=f"存储探针失败: {type(exc).__name__}",
            )
        else:
            components["storage"] = ComponentHealth(
                status=HealthStatus.READY,
                detail="存储可读",
            )

        if self.embedding_provider is None and self.config.memory.embedding_api_base is None:
            components["embedding"] = ComponentHealth(
                status=HealthStatus.DEGRADED,
                detail="使用确定性本地词法检索",
                required=False,
            )
        else:
            components["embedding"] = ComponentHealth(
                status=HealthStatus.READY,
                detail="嵌入 Provider 已配置",
                required=False,
            )

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
            sessions = list(self.sessions)
            if sessions:
                await asyncio.gather(
                    *(session.close() for session in sessions),
                    return_exceptions=True,
                )
            await self.supervisor.close()
            child_sessions = list(self.subagent_sessions)
            if child_sessions:
                await asyncio.gather(
                    *(self.release_subagent_session(session) for session in child_sessions),
                    return_exceptions=True,
                )
            if self.audit_sink is not None:
                await self.audit_sink.close()
            if self.embedding_provider is not None:
                await self.embedding_provider.close()
            if self.gateway is not None:
                await self.gateway.close()
            if self.store is not None:
                await self.store.close()
            self.runtime_state = RuntimeState.CLOSED


class AgentSession:
    """异步上下文管理的会话；同一实例禁止并发执行轮次。"""

    def __init__(self, runtime: PraxisRuntime) -> None:
        self.runtime = runtime
        self.runner: SessionRunner | None = None
        self.run_lock = asyncio.Lock()
        self.closed = False

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
        if self.closed or self.runner.status is SessionStatus.TERMINATED:
            raise SessionError("AgentSession 已终止")
        return self.runner

    async def run(self, user_input: InputValue, **kwargs: Any) -> AgentResponse:
        runner = self.require_runner()
        if self.run_lock.locked():
            raise ConcurrentSessionRunError("同一 AgentSession 不能并发执行两个轮次")
        async with self.run_lock:
            with use_metrics(self.runtime.metrics):
                return await runner.run_turn(user_input, **kwargs)

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
                stream = runner.run_turn_stream(user_input, **kwargs)
                async with aclosing(stream):
                    async for event in stream:
                        yield event

    def abort(self) -> None:
        self.require_runner().abort()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.runner is not None:
            self.runner.abort()
            async with self.run_lock:
                await self.runner.terminate()
        self.runtime.unregister_session(self)


__all__ = [
    "AgentSession",
    "ComponentHealth",
    "HealthStatus",
    "PraxisRuntime",
    "RuntimeHealth",
    "RuntimeState",
]
