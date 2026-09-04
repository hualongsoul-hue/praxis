"""子代理工具注册。

将 S13 的三种执行模型（Agent-as-Tool、Fork、Handoff）注册为 S5 工具，
使 LLM 可通过工具调用触发子代理委托。
"""

import json
from typing import Any

from praxis.config.schemas import (
    ContextConfig,
    InputConfig,
    OrchestratorConfig,
    SubagentConfig,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.lifecycle import TaskSupervisor
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.persistence.store import PersistenceStore
from praxis.protocols import ModelGateway
from praxis.session.core import Session
from praxis.subagent.aggregation import ResultAggregator
from praxis.subagent.fork import ForkManager
from praxis.subagent.handoff import HandoffManager
from praxis.subagent.isolation import IsolatedContext, RuntimeSubagentFactory
from praxis.subagent.resource_control import ResourceController
from praxis.subagent.spawn import SubagentSpawner
from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolHandler, ToolRegistry

log = get_logger("subagent.tools")

SPAWN_TOOL = ToolDefinition(
    name="spawn_subagent",
    description=(
        "创建专家子代理处理有界子任务。"
        "子代理拥有独立上下文窗口和工具子集，可消耗大量 Token 进行深度工作，"
        "最终返回精炼摘要（≤1500 Token）。"
        "适用于需要深度分析、独立研究的子问题。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "子任务描述",
            },
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "授予子代理的工具名称列表（为主代理工具集的子集）",
            },
            "context_summary": {
                "type": "string",
                "description": "传递给子代理的上下文摘要",
            },
        },
        "required": ["task"],
    },
    metadata=ToolMetadata(
        category="subagent",
        permission_level="confirm",
        readonly=False,
        timeout_seconds=300.0,
        tags=["subagent", "delegation"],
    ),
)

FORK_TOOL = ToolDefinition(
    name="fork_subagents",
    description=(
        "并行创建多个子代理分别执行不同任务，独立工作后聚合结果。"
        "适用于需要并行探索多条路径或同时处理多个独立子问题的场景。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "子任务描述"},
                        "tool_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "工具名称列表",
                        },
                    },
                    "required": ["task"],
                },
                "description": "并行任务列表",
            },
            "context_summary": {
                "type": "string",
                "description": "共享上下文摘要",
            },
        },
        "required": ["tasks"],
    },
    metadata=ToolMetadata(
        category="subagent",
        permission_level="confirm",
        readonly=False,
        timeout_seconds=600.0,
        tags=["subagent", "fork", "parallel"],
    ),
)

HANDOFF_TOOL = ToolDefinition(
    name="handoff_to_expert",
    description=(
        "将控制权移交给专家代理。主代理暂停，专家代理接管完全控制权，"
        "传递精炼上下文摘要。完成后控制权返回主代理。"
        "适用于需要特定领域专家处理的任务。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "target_agent_type": {
                "type": "string",
                "description": "目标专家代理类型（如 researcher、code_reviewer）",
            },
            "context_summary": {
                "type": "string",
                "description": "传递给专家代理的精炼上下文摘要",
            },
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "授予专家代理的工具名称列表",
            },
        },
        "required": ["target_agent_type", "context_summary"],
    },
    metadata=ToolMetadata(
        category="subagent",
        permission_level="confirm",
        readonly=False,
        timeout_seconds=300.0,
        tags=["subagent", "handoff"],
    ),
)


def create_spawn_handler(spawner: SubagentSpawner) -> ToolHandler:
    """创建 spawn_subagent 工具处理函数。"""

    async def handler(arguments: dict[str, Any]) -> str:
        result = await spawner.spawn_agent_as_tool(
            task=arguments["task"],
            tool_names=arguments.get("tool_names"),
            context_summary=arguments.get("context_summary", ""),
        )
        return json.dumps(result.model_dump(), ensure_ascii=False)

    return handler


def create_fork_handler(fork_manager: ForkManager) -> ToolHandler:
    """创建 fork_subagents 工具处理函数。"""

    async def handler(arguments: dict[str, Any]) -> str:
        aggregated = await fork_manager.fork_and_aggregate(
            tasks=arguments["tasks"],
            context_summary=arguments.get("context_summary", ""),
        )
        return json.dumps(aggregated, ensure_ascii=False, default=str)

    return handler


def create_handoff_handler(handoff_manager: HandoffManager) -> ToolHandler:
    """创建 handoff_to_expert 工具处理函数。"""

    async def handler(arguments: dict[str, Any]) -> str:
        result = await handoff_manager.handoff(
            target_agent_type=arguments["target_agent_type"],
            context_summary=arguments["context_summary"],
            tool_names=arguments.get("tool_names"),
        )
        return json.dumps(result.model_dump(), ensure_ascii=False)

    return handler


def register_subagent_tools(
    registry: ToolRegistry,
    spawner: SubagentSpawner,
    fork_manager: ForkManager,
    handoff_manager: HandoffManager,
) -> None:
    """将 S13 三种执行模型注册为工具。

    Args:
        registry: 工具注册表。
        spawner: Agent-as-Tool 创建器。
        fork_manager: Fork 管理器。
        handoff_manager: Handoff 管理器。
    """
    registry.register(SPAWN_TOOL, create_spawn_handler(spawner))
    registry.register(FORK_TOOL, create_fork_handler(fork_manager))
    registry.register(HANDOFF_TOOL, create_handoff_handler(handoff_manager))


def wire_subagent(
    session: Session,
    store: PersistenceStore,
    guardrails: GuardrailEngine,
    gateway: ModelGateway,
    orchestrator_config: OrchestratorConfig,
    context_config: ContextConfig,
    input_config: InputConfig,
    subagent_config: SubagentConfig,
    model: str = "default",
    runtime: RuntimeSubagentFactory | None = None,
    supervisor: TaskSupervisor | None = None,
) -> None:
    """将 S13 子代理三种执行模型注册到会话的工具注册表中。

    子代理通过 IsolatedContext + SessionFactory 创建独立子会话，
    子会话不包含 S13 工具（天然防止递归嵌套）。

    Args:
        session: 目标会话。
        store: S3 持久化存储（子代理共享）。
        guardrails: S8 护栏引擎（子代理共享只读实例）。
        gateway: S4 LLM 网关路由器（子代理共享）。
        orchestrator_config: S11 编排配置（子代理继承策略类型）。
        context_config: S7 上下文配置（子代理继承）。
        subagent_config: S13 子代理配置。
        model: LLM 模型名。
    """
    isolation = IsolatedContext(
        store=store,
        gateway=gateway,
        orchestrator_config=orchestrator_config,
        context_config=context_config,
        input_config=input_config,
        runtime=runtime,
    )
    resource_ctrl = ResourceController(subagent_config, supervisor=supervisor)
    aggregator = ResultAggregator()

    spawner = SubagentSpawner(
        isolation=isolation,
        resource_ctrl=resource_ctrl,
        guardrails=guardrails,
        parent_registry=session.registry,
        model=model,
    )
    fork_manager = ForkManager(
        isolation=isolation,
        resource_ctrl=resource_ctrl,
        aggregator=aggregator,
        guardrails=guardrails,
        parent_registry=session.registry,
        model=model,
    )
    handoff_manager = HandoffManager(
        isolation=isolation,
        resource_ctrl=resource_ctrl,
        guardrails=guardrails,
        parent_registry=session.registry,
        model=model,
    )

    register_subagent_tools(
        session.registry, spawner, fork_manager, handoff_manager,
    )

    log.info(
        "S13 子代理工具已注册",
        tools=["spawn_subagent", "fork_subagents", "handoff_to_expert"],
    )
