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

from praxis.config.schemas import (
    ContextConfig,
    OrchestratorConfig,
    SessionConfig,
    SubagentConfig,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.memory.pipeline import MemoryPipeline
from praxis.persistence.store import PersistenceStore
from praxis.session.core import Session, SessionFactory
from praxis.skills.manager import SkillManager
from praxis.subagent.tools import wire_subagent
from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolRegistry
from praxis.verification.registry import VerifierRegistry

log = get_logger("agent")


def create_agent_session(
    store: PersistenceStore,
    guardrails: GuardrailEngine,
    session_config: SessionConfig | None = None,
    orchestrator_config: OrchestratorConfig | None = None,
    context_config: ContextConfig | None = None,
    subagent_config: SubagentConfig | None = None,
    registry: ToolRegistry | None = None,
    model: str = "default",
    memory: MemoryPipeline | None = None,
    skill_manager: SkillManager | None = None,
    verifier_registry: VerifierRegistry | None = None,
) -> Session:
    """创建带完整 S1~S14 集成的 Agent 会话。

    先通过 SessionFactory 创建基础会话（S4~S12、S14），
    若提供 subagent_config 则调用 wire_subagent 将 S13 注册为工具。

    Args:
        store: S3 持久化存储。
        guardrails: S8 护栏引擎。
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
    )

    session = factory.create_session(
        guardrails=guardrails,
        registry=registry,
        model=model,
        memory=memory,
        skill_manager=skill_manager,
        verifier_registry=verifier_registry,
    )

    if subagent_config is not None:
        wire_subagent(
            session=session,
            store=store,
            guardrails=guardrails,
            orchestrator_config=orchestrator_config,
            context_config=context_config,
            subagent_config=subagent_config,
            model=model,
        )

    log.info(
        "Agent 会话已创建",
        session_id=session.session_id,
        subagent_enabled=subagent_config is not None,
    )
    return session
