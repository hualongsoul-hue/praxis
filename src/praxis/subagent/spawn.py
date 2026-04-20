"""Agent-as-Tool 执行模型。

spawn_agent_as_tool 创建专家子代理，
独立 S11 循环和上下文窗口，工具集为主代理子集，
执行后返回精炼摘要。
"""

import asyncio
from typing import Any

from praxis.guardrails.engine import GuardrailEngine
from praxis.models.orchestrator import TerminationReason
from praxis.models.subagent import (
    SubagentMode,
    SubagentResult,
    SubagentSpec,
    SubagentStatus,
)
from praxis.subagent.isolation import IsolatedContext
from praxis.subagent.resource_control import ResourceController
from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolRegistry

log = get_logger("subagent.spawn")

SUBAGENT_SYSTEM_PROMPT = (
    "你是一个专家子代理，负责处理特定子任务。\n"
    "请专注于分配给你的任务，完成后提供精炼摘要。\n"
    "摘要应包含关键发现和结论，控制在 1500 Token 以内。"
)


class SubagentSpawner:
    """Agent-as-Tool 子代理创建器。

    创建独立的子代理实例执行子任务，返回精炼摘要。
    """

    def __init__(
        self,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        guardrails: GuardrailEngine,
        parent_registry: ToolRegistry,
        model: str = "default",
    ) -> None:
        self.isolation = isolation
        self.resource_ctrl = resource_ctrl
        self.guardrails = guardrails
        self.parent_registry = parent_registry
        self.model = model

    async def spawn_agent_as_tool(
        self,
        task: str,
        tool_names: list[str] | None = None,
        context_summary: str = "",
        max_turns: int = 50,
        timeout_seconds: float = 300.0,
    ) -> SubagentResult:
        """创建并运行 Agent-as-Tool 子代理。

        Args:
            task: 子任务描述。
            tool_names: 授予的工具子集。
            context_summary: 上下文摘要。
            max_turns: 最大轮次。
            timeout_seconds: 超时秒数。

        Returns:
            子代理执行结果。
        """
        spec = SubagentSpec(
            task=task,
            mode=SubagentMode.AGENT_AS_TOOL,
            tool_names=tool_names or [],
            context_summary=context_summary,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            system_prompt_override=SUBAGENT_SYSTEM_PROMPT,
        )

        await self.resource_ctrl.acquire(spec.subagent_id)

        try:
            result = await self.run_subagent(spec)
        finally:
            self.resource_ctrl.release(spec.subagent_id)

        return result

    async def run_subagent(self, spec: SubagentSpec) -> SubagentResult:
        """运行子代理的核心逻辑。"""
        session = await self.isolation.create_isolated_session(
            spec=spec,
            guardrails=self.guardrails,
            parent_registry=self.parent_registry,
            model=self.model,
        )

        # 构造任务消息
        user_message = spec.task
        if spec.context_summary:
            user_message = f"上下文摘要:\n{spec.context_summary}\n\n任务:\n{spec.task}"

        try:
            response = await asyncio.wait_for(
                session.run_turn(
                    user_message,
                    system_prompt_override=spec.system_prompt_override,
                ),
                timeout=spec.timeout_seconds,
            )

            return SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.AGENT_AS_TOOL,
                status=SubagentStatus.COMPLETED,
                summary=response.content,
                total_turns=response.total_turns,
                total_tokens=sum(
                    e.data.get("token_count", 0) for e in response.events
                    if e.event_type == "llm_request"
                ),
            )
        except asyncio.TimeoutError:
            session.abort()
            log.warning("子代理超时", subagent_id=spec.subagent_id)
            return SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.AGENT_AS_TOOL,
                status=SubagentStatus.TIMEOUT,
                summary="子代理执行超时，部分结果可能不完整。",
            )
        except Exception as exc:
            log.error("子代理执行失败", subagent_id=spec.subagent_id, error=str(exc))
            return SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.AGENT_AS_TOOL,
                status=SubagentStatus.FAILED,
                summary=f"执行失败: {exc}",
            )
