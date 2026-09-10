"""会话初始化。

create_session 创建新会话时初始化所有组件实例，
注入配置，加载项目级记忆/工具/权限，生成会话 ID 和初始检查点。
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, aclosing
from typing import Any

from praxis.config.schemas import (
    ContextConfig,
    InputConfig,
    MemoryConfig,
    ModelCapabilities,
    OrchestratorConfig,
    RecoveryConfig,
    SessionConfig,
    ToolsConfig,
)
from praxis.context.assembler import PromptAssembler
from praxis.context.compaction import ContextCompactor
from praxis.context.jit_retrieval import JITRetriever
from praxis.context.masking import ObservationMasker
from praxis.context.tool_injection import ToolInjector
from praxis.exceptions import SessionError
from praxis.guardrails.engine import GuardrailEngine
from praxis.input_resolver import InputResolver
from praxis.memory.core import CognitiveMemory
from praxis.memory.vector import VectorStore
from praxis.models.inputs import InputValue
from praxis.models.orchestrator import AgentEvent, AgentResponse, StrategyMode
from praxis.models.session import (
    SessionMetadata,
    SessionStatus,
)
from praxis.models.tools import ToolExecutionRecord
from praxis.orchestrator.events import EventEmitter
from praxis.orchestrator.loop import OrchestrationLoop
from praxis.orchestrator.parser import OutputParser
from praxis.orchestrator.strategy import LoopStrategy
from praxis.orchestrator.termination import TerminationManager
from praxis.orchestrator.tool_coordination import ToolCoordinator
from praxis.persistence.store import PersistenceStore
from praxis.protocols import ApprovalHandler, AuditSink, EmbeddingProvider, ModelGateway
from praxis.recovery.circuit_breaker import CircuitBreakerRegistry
from praxis.recovery.fallback import FallbackRegistry
from praxis.recovery.retry import RetryPolicy
from praxis.resources import ResourceController
from praxis.session.checkpoint import CheckpointManager
from praxis.skills.manager import SkillManager
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric
from praxis.tools.builtins.jit_ops import register_jit_tools
from praxis.tools.builtins.memory_ops import register_memory_tools
from praxis.tools.builtins.registration import register_builtins
from praxis.tools.executor import ToolExecutor
from praxis.tools.policy import ToolPolicy
from praxis.tools.registry import ToolRegistry
from praxis.verification.registry import VerifierRegistry

log = get_logger("session.core")


def strategy_mode(name: str) -> StrategyMode:
    """将 OrchestratorConfig.default_strategy 字符串映射为 StrategyMode。"""
    try:
        return StrategyMode(name)
    except ValueError:
        return StrategyMode.REACT


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
        input_resolver: InputResolver,
        model_capabilities: ModelCapabilities,
        memory: CognitiveMemory | None = None,
        skill_manager: SkillManager | None = None,
        verifier_registry: VerifierRegistry | None = None,
        resources: ResourceController | None = None,
        continuation: Any = None,
    ) -> None:
        self.metadata = metadata
        self.loop = loop
        self.assembler = assembler
        self.injector = injector
        self.registry = registry
        self.store = store
        self.config = config
        self.input_resolver = input_resolver
        self.model_capabilities = model_capabilities
        self.memory = memory
        self.skill_manager = skill_manager
        self.verifier_registry = verifier_registry
        self.resources = resources
        # S12 跨窗口续接管理器（可选）；存在时按续接阶段注入热身/初始化序列。
        self.continuation = continuation
        # MCP 连接（可选）：管理器 + 持有传输生命周期的退出栈，terminate 时关闭。
        self.mcp_manager: Any = None
        self.mcp_sampling_manager: Any = None
        self.mcp_elicitation_manager: Any = None
        self.mcp_auth_manager: Any = None
        self.mcp_stack: Any = None
        self.tool_execution_ledger: dict[str, ToolExecutionRecord] = {}
        self.checkpoint_lock = asyncio.Lock()
        self.loop.coordinator.configure_execution_tracking(
            self.tool_execution_ledger,
            self.persist_tool_execution,
        )

    async def persist_tool_execution(self, record: ToolExecutionRecord) -> None:
        """Persist a side-effect boundary before orchestration may continue."""
        if self.config.auto_checkpoint:
            await self.save_auto_checkpoint(
                description=f"Tool {record.tool_call_id}: {record.state.value}",
            )

    def attach_mcp_stack(self, stack: AsyncExitStack) -> None:
        """转移 MCP 连接退出栈的所有权，随 Session 统一关闭。"""
        self.mcp_stack = stack

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def status(self) -> SessionStatus:
        return self.metadata.status

    def apply_continuation(self, kwargs: dict[str, Any]) -> None:
        """续接阶段（初始化/热身）按需注入开发者指令，不覆盖显式入参。"""
        if self.continuation is None:
            return
        for key, value in self.continuation.prepare_turn(self).items():
            kwargs.setdefault(key, value)

    async def run_turn(self, user_input: InputValue, **kwargs: Any) -> AgentResponse:
        """执行一轮对话。"""
        self.metadata.status = SessionStatus.ACTIVE
        self.apply_continuation(kwargs)
        resolved = await self.input_resolver.resolve(
            user_input,
            self.model_capabilities,
        )
        response = await self.loop.run(resolved, **kwargs)
        if self.continuation is not None:
            self.continuation.advance_phase(self)
        self.metadata.total_turns += response.total_turns
        self.metadata.total_tokens += sum(
            e.data.get("token_count", 0) for e in response.events
            if e.event_type == "llm_request"
        )

        # 自动检查点
        if self.config.auto_checkpoint:
            await self.save_auto_checkpoint()

        return response

    async def save_auto_checkpoint(self, description: str | None = None) -> str | None:
        """自动保存检查点（S3 持久化 S6/S7/S11 状态）。"""
        async with self.checkpoint_lock:
            checkpoint_mgr = CheckpointManager(self.store)
            context_state = {
                "messages": self.assembler.conversation_history,
                "file_refs": self.assembler.file_refs,
                "compaction_count": self.assembler.compaction_count,
                "tool_schemas": self.assembler.tool_schemas,
            }
            memory_state = self.memory.export_state() if self.memory is not None else {}
            loop_state = self.loop.state.model_dump(mode="json")
            recovery_state = {
                "circuits": self.loop.coordinator.circuits.export_state(),
                "retry": self.loop.coordinator.retry_policy.export_state(),
            }
            checkpoint_id = await checkpoint_mgr.save_checkpoint(
                metadata=self.metadata,
                context_state=context_state,
                memory_state=memory_state,
                loop_state=loop_state,
                strategy_state=self.loop.strategy.export_state(),
                recovery_state=recovery_state,
                tool_execution_ledger={
                    key: value.model_dump(mode="json")
                    for key, value in self.tool_execution_ledger.items()
                },
                file_refs=self.assembler.file_refs,
                description=description or (
                    f"Auto checkpoint after turn {self.metadata.total_turns}"
                ),
            )
            await self.prune_checkpoints(checkpoint_mgr)
            return checkpoint_id

    async def prune_checkpoints(self, checkpoint_mgr: CheckpointManager) -> None:
        """按 max_checkpoints_per_session 保留最新若干个，删除最旧的多余检查点。"""
        limit = self.config.max_checkpoints_per_session
        if limit <= 0:
            return
        infos = await checkpoint_mgr.list_checkpoints(self.session_id)
        if len(infos) <= limit:
            return
        infos.sort(key=lambda c: c.created_at)  # 旧→新
        for stale in infos[: len(infos) - limit]:
            await checkpoint_mgr.delete_checkpoint(self.session_id, stale.checkpoint_id)

    async def run_turn_stream(
        self,
        user_input: InputValue,
        **kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]:
        """流式执行一轮对话。"""
        self.metadata.status = SessionStatus.ACTIVE
        self.apply_continuation(kwargs)
        resolved = await self.input_resolver.resolve(
            user_input,
            self.model_capabilities,
        )
        turn_tokens = 0
        completed = False
        try:
            stream = self.loop.run_stream(resolved, **kwargs)
            async with aclosing(stream):
                async for event in stream:
                    if event.event_type == "llm_request":
                        turn_tokens += event.data.get("token_count", 0)
                    yield event
            completed = True
        finally:
            if completed and self.continuation is not None:
                self.continuation.advance_phase(self)
            self.metadata.total_turns += self.loop.state.current_turn
            self.metadata.total_tokens += turn_tokens
            if self.config.auto_checkpoint:
                await self.save_auto_checkpoint(
                    description=(
                        "Auto checkpoint after stream completion"
                        if completed
                        else "Auto checkpoint after stream interruption"
                    ),
                )

    def abort(self) -> None:
        """中断当前会话。"""
        self.loop.abort()

    async def terminate(self) -> None:
        """终止会话：停止后台记忆 Worker、Dream 调度器并关闭 MCP 连接。"""
        if self.metadata.status is SessionStatus.TERMINATED:
            return
        failures: list[BaseException] = []
        if self.resources is not None:
            try:
                await self.resources.cancel_group(self.session_id)
            except BaseException as exc:
                failures.append(exc)
        if self.memory is not None:
            try:
                await self.memory.stop()
            except BaseException as exc:
                failures.append(exc)
        if self.mcp_manager is not None:
            try:
                await self.mcp_manager.close()
            except BaseException as exc:
                failures.append(exc)
        if self.mcp_stack is not None:
            try:
                await self.mcp_stack.aclose()
            except BaseException as exc:
                failures.append(exc)
            else:
                self.mcp_stack = None
        if failures:
            raise SessionError(
                f"会话终止期间发生 {len(failures)} 个错误",
                details={"failure_types": [type(item).__name__ for item in failures]},
            ) from failures[0]
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
        input_config: InputConfig | None = None,
        memory_config: MemoryConfig | None = None,
        recovery_config: RecoveryConfig | None = None,
        approval_handler: ApprovalHandler | None = None,
        audit_sink: AuditSink | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        resources: ResourceController | None = None,
        runtime_id: str | None = None,
    ) -> None:
        self.store = store
        self.session_config = session_config
        self.orchestrator_config = orchestrator_config
        self.context_config = context_config
        self.input_config = input_config or InputConfig()
        self.memory_config = memory_config or MemoryConfig()
        self.recovery_config = recovery_config or RecoveryConfig()
        self.approval_handler = approval_handler
        self.audit_sink = audit_sink
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.resources = resources
        self.runtime_id = runtime_id

    async def create_session(
        self,
        guardrails: GuardrailEngine,
        gateway: ModelGateway,
        registry: ToolRegistry | None = None,
        model: str = "default",
        memory: CognitiveMemory | None = None,
        skill_manager: SkillManager | None = None,
        verifier_registry: VerifierRegistry | None = None,
        tools_config: ToolsConfig | None = None,
        include_builtins: bool = True,
        jit_retriever: JITRetriever | None = None,
    ) -> Session:
        """创建新会话。

        初始化所有组件实例并注入配置。

        Args:
            guardrails: 护栏策略模板；为本会话复制规则，临时授权不继承。
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
        guardrails = guardrails.for_session()
        metadata = SessionMetadata(status=SessionStatus.INITIALIZING)
        model_capabilities = gateway.capabilities(model)
        input_resolver = InputResolver(self.input_config)

        # S6: 记忆系统——自动创建并启动（后台 Worker + Dream 调度器）
        if memory is None:
            memory = CognitiveMemory(
                store=self.store,
                gateway=gateway,
                session_id=metadata.session_id,
                config=self.memory_config,
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
            )
        # S5: 工具系统
        resolved_tools_config = tools_config or ToolsConfig()
        created_new_registry = registry is None
        if created_new_registry:
            registry = ToolRegistry()
        sandbox = ToolPolicy(resolved_tools_config)
        if created_new_registry and include_builtins:
            register_builtins(registry, sandbox, store=self.store)
            register_memory_tools(registry, memory)
            # S7: JIT 懒加载工具（仅在 JITRetriever 配置了 ContentLoader 时）
            if jit_retriever is not None:
                register_jit_tools(registry, jit_retriever)
        executor = ToolExecutor(registry, sandbox, resources=self.resources)
        injector = ToolInjector(registry)

        # S7: 上下文引擎
        assembler = PromptAssembler(self.context_config, model=model)

        # S9: 错误恢复（熔断 / 重试 / 降级）——从 RecoveryConfig 装配
        rc = self.recovery_config
        circuits = CircuitBreakerRegistry(
            failure_threshold=rc.circuit_breaker_threshold,
            cooldown_seconds=rc.circuit_breaker_cooldown,
        )
        retry_policy = RetryPolicy(
            max_retries=rc.max_retries,
            initial_delay=rc.base_delay,
            max_delay=rc.max_delay,
        )
        fallbacks = FallbackRegistry()
        # 从配置加载工具降级映射，激活协调器中的降级链路（否则映射恒为空）
        if resolved_tools_config.fallback_mappings:
            fallbacks.load_mappings(resolved_tools_config.fallback_mappings)

        # S11: 编排循环
        emitter = EventEmitter(
            runtime_id=self.runtime_id,
            session_id=metadata.session_id,
        )
        parser = OutputParser()
        termination = TerminationManager(self.orchestrator_config)
        strategy = LoopStrategy(mode=strategy_mode(self.orchestrator_config.default_strategy))

        coordinator = ToolCoordinator(
            executor=executor,
            registry=registry,
            guardrails=guardrails,
            circuit_registry=circuits,
            retry_policy=retry_policy,
            emitter=emitter,
            fallback_registry=fallbacks,
            approval_handler=self.approval_handler,
            approval_timeout=resolved_tools_config.approval_timeout,
            audit_sink=self.audit_sink,
            session_id=metadata.session_id,
            status_callback=lambda status: setattr(metadata, "status", status),
        )

        # S10: 为验证器注册表注入网关，启用推理型/视觉型验证（否则二者不可用）
        if verifier_registry is not None and verifier_registry.gateway is None:
            verifier_registry.gateway = gateway

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
            compaction_min_history=self.context_config.compaction_min_history,
            jit_retriever=jit_retriever,
            model=model,
        )

        metadata.status = SessionStatus.ACTIVE

        # 注意：不以 session_id 作为指标标签（会造成无界基数）；session_id 见日志/追踪
        emit_metric("session_created", 1.0, {}, "counter")
        log.info("会话已创建", session_id=metadata.session_id)

        session = Session(
            metadata=metadata,
            loop=loop,
            assembler=assembler,
            injector=injector,
            registry=registry,
            store=self.store,
            config=self.session_config,
            input_resolver=input_resolver,
            model_capabilities=model_capabilities,
            memory=memory,
            skill_manager=skill_manager,
            verifier_registry=verifier_registry,
            resources=self.resources,
        )
        try:
            await memory.start()
        except BaseException as error:
            try:
                await session.terminate()
            except BaseException as cleanup_error:
                error.add_note(f"记忆启动回滚失败: {type(cleanup_error).__name__}")
            raise
        return session
