"""Handoff 执行模型。

将控制权转移到专家代理，传递精炼上下文摘要（非完整历史），
完成后返回主代理。
"""

import asyncio

from praxis.guardrails.engine import GuardrailEngine
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

log = get_logger("subagent.handoff")

HANDOFF_SYSTEM_PROMPT = (
    "你是一个接管控制权的专家代理。\n"
    "主代理已将完整控制权移交给你，请处理分配的任务。\n"
    "完成后提供完整的执行报告和结论。"
)


class HandoffManager:
    """Handoff 执行管理器。

    专家代理接管完全控制权，主代理暂停，
    传递精炼的上下文摘要而非完整历史，
    移交完成后控制权返回主代理。
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

    async def handoff(
        self,
        target_agent_type: str,
        context_summary: str,
        tool_names: list[str] | None = None,
        max_turns: int = 50,
        timeout_seconds: float = 300.0,
    ) -> SubagentResult:
        """执行 Handoff，将控制权移交专家代理。

        Args:
            target_agent_type: 目标代理类型名。
            context_summary: 精炼上下文摘要。
            tool_names: 授予的工具子集。
            max_turns: 最大轮次。
            timeout_seconds: 超时秒数。

        Returns:
            专家代理执行结果。
        """
        spec = SubagentSpec(
            task=f"Handoff to {target_agent_type}",
            mode=SubagentMode.HANDOFF,
            tool_names=tool_names or [],
            context_summary=context_summary,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            system_prompt_override=HANDOFF_SYSTEM_PROMPT,
        )

        log.info(
            "Handoff 开始",
            target=target_agent_type,
            subagent_id=spec.subagent_id,
        )

        await self.resource_ctrl.acquire(spec.subagent_id)

        try:
            result = await self.run_handoff(spec)
        finally:
            self.resource_ctrl.release(spec.subagent_id)

        log.info(
            "Handoff 完成",
            target=target_agent_type,
            status=result.status.value,
        )
        return result

    async def run_handoff(self, spec: SubagentSpec) -> SubagentResult:
        """运行 Handoff 代理。"""
        session = self.isolation.create_isolated_session(
            spec=spec,
            guardrails=self.guardrails,
            parent_registry=self.parent_registry,
            model=self.model,
        )

        user_message = f"上下文摘要:\n{spec.context_summary}\n\n请接管并处理。"

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
                mode=SubagentMode.HANDOFF,
                status=SubagentStatus.COMPLETED,
                summary=response.content,
                total_turns=response.total_turns,
            )
        except asyncio.TimeoutError:
            session.abort()
            return SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.HANDOFF,
                status=SubagentStatus.TIMEOUT,
                summary="Handoff 执行超时。",
            )
        except Exception as exc:
            log.error("Handoff 执行失败", error=str(exc))
            return SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.HANDOFF,
                status=SubagentStatus.FAILED,
                summary=f"执行失败: {exc}",
            )
