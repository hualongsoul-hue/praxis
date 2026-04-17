"""会话初始化。

create_session 创建新会话时初始化所有子系统实例，
注入配置，加载项目级记忆/工具/权限，生成会话 ID 和初始检查点。
"""

from collections.abc import AsyncIterator
from typing import Any

from praxis.config.subsystems import (
    ContextConfig,
    LifecycleConfig,
    OrchestratorConfig,
    ToolsConfig,
)
from praxis.context.assembler import PromptAssembler
from praxis.context.tool_injection import ToolInjector
from praxis.guardrails.engine import GuardrailEngine
from praxis.models.lifecycle import (
    ContinuationPhase,
    SessionMetadata,
    SessionStatus,
)
from praxis.models.orchestrator import AgentEvent, AgentResponse
from praxis.orchestrator.events import EventEmitter
from praxis.orchestrator.loop import OrchestrationLoop
from praxis.orchestrator.parser import OutputParser
from praxis.orchestrator.strategy import LoopStrategy
from praxis.orchestrator.termination import TerminationManager
from praxis.orchestrator.tool_coordination import ToolCoordinator
from praxis.persistence.store import PersistenceStore
from praxis.recovery.circuit_breaker import CircuitBreakerRegistry
from praxis.recovery.retry import RetryPolicy
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric
from praxis.tools.executor import ToolExecutor
from praxis.tools.registry import ToolRegistry
from praxis.tools.sandbox import Sandbox

log = get_logger("lifecycle.session")


class Session:
    """Agent 会话。

    持有当前会话的所有子系统实例和状态。
    """

    def __init__(
        self,
        metadata: SessionMetadata,
        loop: OrchestrationLoop,
        assembler: PromptAssembler,
        injector: ToolInjector,
        registry: ToolRegistry,
        store: PersistenceStore,
        config: LifecycleConfig,
    ) -> None:
        self.metadata = metadata
        self.loop = loop
        self.assembler = assembler
        self.injector = injector
        self.registry = registry
        self.store = store
        self.config = config

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def status(self) -> SessionStatus:
        return self.metadata.status

    async def run_turn(self, user_message: str, **kwargs: Any) -> AgentResponse:
        """执行一轮对话。"""
        self.metadata.status = SessionStatus.ACTIVE
        response = await self.loop.run(user_message, **kwargs)
        self.metadata.total_turns += response.total_turns
        self.metadata.total_tokens += sum(
            e.data.get("token_count", 0) for e in response.events
            if e.event_type == "llm_request"
        )
        return response

    async def run_turn_stream(
        self,
        user_message: str,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        """流式执行一轮对话。"""
        self.metadata.status = SessionStatus.ACTIVE
        async for event in self.loop.run_stream(user_message, **kwargs):
            yield event

    def abort(self) -> None:
        """中断当前会话。"""
        self.loop.abort()

    def terminate(self) -> None:
        """终止会话。"""
        self.metadata.status = SessionStatus.TERMINATED
        log.info("会话已终止", session_id=self.session_id)


class SessionFactory:
    """会话工厂。

    负责创建新会话，初始化所有子系统并注入依赖。
    """

    def __init__(
        self,
        store: PersistenceStore,
        lifecycle_config: LifecycleConfig,
        orchestrator_config: OrchestratorConfig,
        context_config: ContextConfig,
    ) -> None:
        self.store = store
        self.lifecycle_config = lifecycle_config
        self.orchestrator_config = orchestrator_config
        self.context_config = context_config

    def create_session(
        self,
        guardrails: GuardrailEngine,
        registry: ToolRegistry | None = None,
        model: str = "default",
    ) -> Session:
        """创建新会话。

        初始化所有子系统实例并注入配置。

        Args:
            guardrails: 护栏引擎（外部传入，因权限配置项目级别）。
            registry: 工具注册表（可选，None 时创建新实例）。
            model: LLM 模型名。

        Returns:
            初始化完毕的 Session。
        """
        metadata = SessionMetadata(status=SessionStatus.INITIALIZING)

        # S5: 工具系统
        if registry is None:
            registry = ToolRegistry()
        sandbox = Sandbox(ToolsConfig())
        executor = ToolExecutor(registry, sandbox)
        injector = ToolInjector(registry)

        # S7: 上下文引擎
        assembler = PromptAssembler(self.context_config, model=model)

        # S9: 错误恢复
        circuits = CircuitBreakerRegistry()
        retry_policy = RetryPolicy()

        # S11: 编排循环
        emitter = EventEmitter()
        parser = OutputParser()
        termination = TerminationManager(self.orchestrator_config)
        strategy = LoopStrategy()

        coordinator = ToolCoordinator(
            executor=executor,
            registry=registry,
            guardrails=guardrails,
            circuit_registry=circuits,
            retry_policy=retry_policy,
            emitter=emitter,
        )

        # GatewayRouter 需要外部传入（因模型配置不在此层管理）
        # loop 在运行时设置 gateway
        loop = OrchestrationLoop(
            config=self.orchestrator_config,
            gateway=None,  # type: ignore[arg-type]
            assembler=assembler,
            tool_coordinator=coordinator,
            guardrails=guardrails,
            termination=termination,
            strategy=strategy,
            parser=parser,
            emitter=emitter,
        )

        metadata.status = SessionStatus.ACTIVE

        emit_metric("session_created", 1.0, {"session_id": metadata.session_id}, "counter")
        log.info("会话已创建", session_id=metadata.session_id)

        return Session(
            metadata=metadata,
            loop=loop,
            assembler=assembler,
            injector=injector,
            registry=registry,
            store=self.store,
            config=self.lifecycle_config,
        )
