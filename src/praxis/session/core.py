"""会话初始化。

create_session 创建新会话时初始化所有组件实例，
注入配置，加载项目级记忆/工具/权限，生成会话 ID 和初始检查点。
"""

from collections.abc import AsyncIterator
from typing import Any

from praxis.config.schemas import (
    ContextConfig,
    MemoryConfig,
    SessionConfig,
    OrchestratorConfig,
    ToolsConfig,
)
from praxis.context.assembler import PromptAssembler
from praxis.context.compaction import ContextCompactor
from praxis.context.masking import ObservationMasker
from praxis.context.tool_injection import ToolInjector
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.session.checkpoint import CheckpointManager
from praxis.memory.core import CognitiveMemory
from praxis.models.session import (
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
from praxis.recovery.fallback import FallbackRegistry
from praxis.recovery.retry import RetryPolicy
from praxis.skills.manager import SkillManager
from praxis.telemetry.audit import configure_audit
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric
from praxis.tools.builtins.registration import register_builtins
from praxis.tools.executor import ToolExecutor
from praxis.tools.registry import ToolRegistry
from praxis.tools.sandbox import Sandbox
from praxis.verification.registry import VerifierRegistry

log = get_logger("session.core")


class Session:
    """Agent 会话。

    持有当前会话的所有组件实例和状态。
    """

    def __init__(
        self,
        metadata: SessionMetadata,
        loop: OrchestrationLoop,
        assembler: PromptAssembler,
        injector: ToolInjector,
        registry: ToolRegistry,
        store: PersistenceStore,
        config: SessionConfig,
        memory: CognitiveMemory | None = None,
        skill_manager: SkillManager | None = None,
        verifier_registry: VerifierRegistry | None = None,
    ) -> None:
        self.metadata = metadata
        self.loop = loop
        self.assembler = assembler
        self.injector = injector
        self.registry = registry
        self.store = store
        self.config = config
        self.memory = memory
        self.skill_manager = skill_manager
        self.verifier_registry = verifier_registry

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

        # 自动检查点
        if self.config.auto_checkpoint:
            await self.save_auto_checkpoint()

        return response

    async def save_auto_checkpoint(self) -> str | None:
        """自动保存检查点（S3 持久化 S6/S7/S11 状态）。"""
        checkpoint_mgr = CheckpointManager(self.store)

        context_state = {
            "messages": self.assembler.conversation_history,
            "file_refs": self.assembler.file_refs,
            "compaction_count": self.assembler.compaction_count,
            "tool_schemas": self.assembler.tool_schemas,
        }
        memory_state = self.memory.export_state() if self.memory is not None else {}
        loop_state = self.loop.state.model_dump(mode="json")

        checkpoint_id = await checkpoint_mgr.save_checkpoint(
            metadata=self.metadata,
            context_state=context_state,
            memory_state=memory_state,
            loop_state=loop_state,
            file_refs=self.assembler.file_refs,
            description=f"Auto checkpoint after turn {self.metadata.total_turns}",
        )
        return checkpoint_id

    async def run_turn_stream(
        self,
        user_message: str,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        """流式执行一轮对话。"""
        self.metadata.status = SessionStatus.ACTIVE
        turn_tokens = 0
        async for event in self.loop.run_stream(user_message, **kwargs):
            if event.event_type == "llm_request":
                turn_tokens += event.data.get("token_count", 0)
            yield event

        # 与 run_turn 对齐：累加统计并触发自动检查点
        self.metadata.total_turns += self.loop.state.current_turn
        self.metadata.total_tokens += turn_tokens
        if self.config.auto_checkpoint:
            await self.save_auto_checkpoint()

    def abort(self) -> None:
        """中断当前会话。"""
        self.loop.abort()

    async def terminate(self) -> None:
        """终止会话：停止后台记忆 Worker 与 Dream 调度器。"""
        if self.memory is not None:
            await self.memory.stop()
        self.metadata.status = SessionStatus.TERMINATED
        log.info("会话已终止", session_id=self.session_id)


class SessionFactory:
    """会话工厂。

    负责创建新会话，初始化所有组件并注入依赖。
    """

    def __init__(
        self,
        store: PersistenceStore,
        session_config: SessionConfig,
        orchestrator_config: OrchestratorConfig,
        context_config: ContextConfig,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        self.store = store
        self.session_config = session_config
        self.orchestrator_config = orchestrator_config
        self.context_config = context_config
        self.memory_config = memory_config or MemoryConfig()
        # 配置 S2 审计持久化通道（护栏裁决事件写入 store 的 "audit" 命名空间）
        configure_audit(store)

    async def create_session(
        self,
        guardrails: GuardrailEngine,
        gateway: GatewayRouter,
        registry: ToolRegistry | None = None,
        model: str = "default",
        memory: CognitiveMemory | None = None,
        skill_manager: SkillManager | None = None,
        verifier_registry: VerifierRegistry | None = None,
        tools_config: ToolsConfig | None = None,
        include_builtins: bool = True,
    ) -> Session:
        """创建新会话。

        初始化所有组件实例并注入配置。

        Args:
            guardrails: 护栏引擎（外部传入，因权限配置项目级别）。
            gateway: S4 LLM 网关路由器（LiteLLM Router 封装）。
            registry: 工具注册表（可选，None 时创建新实例）。
            model: LLM 模型名。
            memory: S6 记忆管线（可选）。
            skill_manager: S14 技能管理器（可选）。
            verifier_registry: S10 验证器注册表（可选）。
            tools_config: S5 工具系统配置（沙箱/超时等）；None 使用默认。
            include_builtins: 是否自动注册内置工具（仅在创建新 registry 时生效）。

        Returns:
            初始化完毕的 Session。
        """
        metadata = SessionMetadata(status=SessionStatus.INITIALIZING)

        # S6: 记忆系统——自动创建并启动（后台 Worker + Dream 调度器）
        if memory is None:
            memory = CognitiveMemory(
                store=self.store,
                gateway=gateway,
                session_id=metadata.session_id,
                config=self.memory_config,
            )
        await memory.start()

        # S5: 工具系统
        created_new_registry = registry is None
        if created_new_registry:
            registry = ToolRegistry()
        sandbox = Sandbox(tools_config or ToolsConfig())
        if created_new_registry and include_builtins:
            register_builtins(registry, sandbox, store=self.store)
        executor = ToolExecutor(registry, sandbox)
        injector = ToolInjector(registry)

        # S7: 上下文引擎
        assembler = PromptAssembler(self.context_config, model=model)

        # S9: 错误恢复（熔断 / 重试 / 降级）
        circuits = CircuitBreakerRegistry()
        retry_policy = RetryPolicy()
        fallbacks = FallbackRegistry()

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
            fallback_registry=fallbacks,
        )

        # S14: 披露工具注册（使 LLM 可触发第二/三层技能披露）
        if skill_manager is not None:
            skill_manager.register_disclosure_tools()

        # S7: 上下文压缩与遮蔽（Token 压力下自动启用）
        compactor = ContextCompactor(self.context_config, gateway=gateway, model=model)
        masker = ObservationMasker(self.context_config, model=model)

        loop = OrchestrationLoop(
            config=self.orchestrator_config,
            gateway=gateway,
            assembler=assembler,
            tool_coordinator=coordinator,
            guardrails=guardrails,
            termination=termination,
            strategy=strategy,
            parser=parser,
            emitter=emitter,
            tool_injector=injector,
            memory=memory,
            verifier_registry=verifier_registry,
            skill_manager=skill_manager,
            compactor=compactor,
            masker=masker,
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
            config=self.session_config,
            memory=memory,
            skill_manager=skill_manager,
            verifier_registry=verifier_registry,
        )
