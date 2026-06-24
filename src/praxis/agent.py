"""Praxis Agent — 框架主入口。

提供创建完整 Agent 会话的统一接口，
内部编排 SessionFactory（S4~S12、S14）与 wire_subagent（S13）的装配。

用法::

    from praxis.agent import create_agent_session

    session = create_agent_session(
        store=store,
        guardrails=guardrails,
        subagent_config=SubagentConfig(),   # 传入即启用子代理
    )
    response = await session.run_turn("你好")
"""

from contextlib import AsyncExitStack

from praxis.config.schemas import (
    ContextConfig,
    MemoryConfig,
    OrchestratorConfig,
    SessionConfig,
    SubagentConfig,
    ToolsConfig,
)
from praxis.models.mcp import MCPServerConfig
from praxis.tools.mcp.wiring import connect_mcp_servers
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.memory.core import CognitiveMemory
from praxis.persistence.store import PersistenceStore
from praxis.session.checkpoint import CheckpointManager
from praxis.session.continuation import ContinuationManager
from praxis.session.core import Session, SessionFactory
from praxis.session.resume import SessionResumer
from praxis.skills.manager import SkillManager
from praxis.subagent.tools import wire_subagent
from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolRegistry
from praxis.verification.registry import VerifierRegistry

log = get_logger("agent")


async def create_agent_session(
    store: PersistenceStore,
    guardrails: GuardrailEngine,
    gateway: GatewayRouter,
    session_config: SessionConfig | None = None,
    orchestrator_config: OrchestratorConfig | None = None,
    context_config: ContextConfig | None = None,
    memory_config: MemoryConfig | None = None,
    subagent_config: SubagentConfig | None = None,
    tools_config: ToolsConfig | None = None,
    registry: ToolRegistry | None = None,
    model: str = "default",
    memory: CognitiveMemory | None = None,
    skill_manager: SkillManager | None = None,
    verifier_registry: VerifierRegistry | None = None,
    include_builtins: bool = True,
    mcp_servers: list[MCPServerConfig] | None = None,
) -> Session:
    """创建带完整 S1~S14 集成的 Agent 会话。

    先通过 SessionFactory 创建基础会话（S4~S12、S14），
    若提供 subagent_config 则调用 wire_subagent 将 S13 注册为工具。

    Args:
        store: S3 持久化存储。
        guardrails: S8 护栏引擎。
        gateway: S4 LLM 网关路由器。
        session_config: S12 会话配置（None 使用默认值）。
        orchestrator_config: S11 编排配置（None 使用默认值）。
        context_config: S7 上下文配置（None 使用默认值）。
        subagent_config: S13 子代理配置（None 则不启用子代理工具）。
        registry: S5 工具注册表（None 则创建新实例）。
        model: LLM 模型名。
        memory: S6 记忆管线。
        skill_manager: S14 技能管理器。
        verifier_registry: S10 验证器注册表。

    Returns:
        初始化完毕的 Session，含完整组件集成。
    """
    session_config = session_config or SessionConfig()
    orchestrator_config = orchestrator_config or OrchestratorConfig()
    context_config = context_config or ContextConfig()

    factory = SessionFactory(
        store=store,
        session_config=session_config,
        orchestrator_config=orchestrator_config,
        context_config=context_config,
        memory_config=memory_config,
    )

    session = await factory.create_session(
        guardrails=guardrails,
        gateway=gateway,
        registry=registry,
        model=model,
        memory=memory,
        skill_manager=skill_manager,
        verifier_registry=verifier_registry,
        tools_config=tools_config,
        include_builtins=include_builtins,
    )

    if subagent_config is not None:
        wire_subagent(
            session=session,
            store=store,
            guardrails=guardrails,
            gateway=gateway,
            orchestrator_config=orchestrator_config,
            context_config=context_config,
            subagent_config=subagent_config,
            model=model,
        )

    # MCP：连接配置的 Server，将其工具注册进会话注册表（生命周期随会话关闭）
    if mcp_servers:
        stack = AsyncExitStack()
        session.mcp_manager = await connect_mcp_servers(
            session.registry, mcp_servers, stack,
        )
        session._mcp_stack = stack

    log.info(
        "Agent 会话已创建",
        session_id=session.session_id,
        subagent_enabled=subagent_config is not None,
        mcp_servers=len(mcp_servers) if mcp_servers else 0,
    )
    return session


async def resume_agent_session(
    store: PersistenceStore,
    guardrails: GuardrailEngine,
    gateway: GatewayRouter,
    session_id: str,
    checkpoint_id: str | None = None,
    session_config: SessionConfig | None = None,
    orchestrator_config: OrchestratorConfig | None = None,
    context_config: ContextConfig | None = None,
    memory_config: MemoryConfig | None = None,
    registry: ToolRegistry | None = None,
    model: str = "default",
    memory: CognitiveMemory | None = None,
    skill_manager: SkillManager | None = None,
    verifier_registry: VerifierRegistry | None = None,
) -> Session | None:
    """从检查点恢复一个 Agent 会话（跨上下文窗口续接）。

    通过 SessionResumer 重建无状态组件并恢复 S6/S7/S11 状态。

    Args:
        store: S3 持久化存储。
        guardrails: S8 护栏引擎。
        gateway: S4 LLM 网关路由器。
        session_id: 待恢复的会话 ID。
        checkpoint_id: 指定检查点 ID；None 时加载最新检查点。
        其余参数同 create_agent_session。

    Returns:
        恢复后的 Session；检查点不存在时返回 None。
    """
    factory = SessionFactory(
        store=store,
        session_config=session_config or SessionConfig(),
        orchestrator_config=orchestrator_config or OrchestratorConfig(),
        context_config=context_config or ContextConfig(),
        memory_config=memory_config,
    )
    resumer = SessionResumer(factory, CheckpointManager(store))
    session = await resumer.resume_session(
        session_id=session_id,
        guardrails=guardrails,
        gateway=gateway,
        registry=registry,
        model=model,
        checkpoint_id=checkpoint_id,
        memory=memory,
        skill_manager=skill_manager,
        verifier_registry=verifier_registry,
    )
    if session is not None:
        # 恢复后处于 WARMUP 阶段：附加续接管理器，使下一轮注入标准热身序列
        session.continuation = ContinuationManager()
        log.info("Agent 会话已恢复", session_id=session_id)
    return session
