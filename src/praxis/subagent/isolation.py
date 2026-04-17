"""上下文隔离。

每个子代理拥有独立组件实例集，
子代理间不共享可变状态，
遥测关联父追踪链路。
"""

from praxis.config.schemas import (
    ContextConfig,
    SessionConfig,
    OrchestratorConfig,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.session.core import Session, SessionFactory
from praxis.models.subagent import SubagentSpec
from praxis.persistence.store import PersistenceStore
from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolRegistry

log = get_logger("subagent.isolation")


class IsolatedContext:
    """隔离上下文。

    为子代理创建完全独立的组件实例集。
    """

    def __init__(
        self,
        store: PersistenceStore,
        orchestrator_config: OrchestratorConfig,
        context_config: ContextConfig,
    ) -> None:
        self.store = store
        self.orchestrator_config = orchestrator_config
        self.context_config = context_config

    def create_isolated_session(
        self,
        spec: SubagentSpec,
        guardrails: GuardrailEngine,
        parent_registry: ToolRegistry,
        model: str = "default",
    ) -> Session:
        """创建隔离的子代理会话。

        Args:
            spec: 子代理规格。
            guardrails: 护栏引擎（共享只读实例）。
            parent_registry: 父代理工具注册表（用于过滤工具子集）。
            model: LLM 模型名。

        Returns:
            隔离的子代理 Session。
        """
        # 创建独立工具注册表（仅包含指定工具子集）
        child_registry = ToolRegistry()
        for tool_name in spec.tool_names:
            if parent_registry.has_tool(tool_name):
                entry = parent_registry.get_entry(tool_name)
                child_registry.register(entry.definition, entry.handler)

        # 为子代理定制编排配置
        sub_orch_config = OrchestratorConfig(
            max_turns=spec.max_turns,
            default_strategy=self.orchestrator_config.default_strategy,
            stream_events=False,
        )

        factory = SessionFactory(
            store=self.store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=sub_orch_config,
            context_config=self.context_config,
        )

        session = factory.create_session(
            guardrails=guardrails,
            registry=child_registry,
            model=model,
        )

        log.info(
            "隔离上下文已创建",
            subagent_id=spec.subagent_id,
            tool_count=len(spec.tool_names),
        )
        return session
