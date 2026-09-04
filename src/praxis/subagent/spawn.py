"""Agent-as-Tool 执行模型。

spawn_agent_as_tool 创建专家子代理，
独立 S11 循环和上下文窗口，工具集为主代理子集，
执行后返回精炼摘要。
"""

import asyncio

from praxis.guardrails.engine import GuardrailEngine
from praxis.models.subagent import (
    SubagentMode,
    SubagentResult,
    SubagentSpec,
    SubagentStatus,
)
from praxis.resources import ResourceController
from praxis.subagent.isolation import IsolatedContext
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
        owner_id: str = "",
    ) -> None:
        self.isolation = isolation
        self.resource_ctrl = resource_ctrl
        self.guardrails = guardrails
        self.parent_registry = parent_registry
        self.model = model
        self.owner_id = owner_id

    async def spawn_agent_as_tool(
        self,
        task: str,
        tool_names: list[str] | None = None,
        context_summary: str = "",
        max_turns: int | None = None,
        timeout_seconds: float | None = None,
    ) -> SubagentResult:
        """创建并运行 Agent-as-Tool 子代理。

        Args:
            task: 子任务描述。
            tool_names: 授予的工具子集。
            context_summary: 上下文摘要。
            max_turns: 最大轮次；None 时取 SubagentConfig.default_max_turns。
            timeout_seconds: 超时秒数；None 时取 SubagentConfig.default_timeout。

        Returns:
            子代理执行结果。
        """
        cfg = self.resource_ctrl.config
        spec = SubagentSpec(
            task=task,
            mode=SubagentMode.AGENT_AS_TOOL,
            tool_names=tool_names or [],
            context_summary=context_summary,
            max_turns=max_turns if max_turns is not None else cfg.default_max_turns,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else cfg.default_timeout,
            system_prompt_override=SUBAGENT_SYSTEM_PROMPT,
        )

        await self.resource_ctrl.acquire(spec.subagent_id)

        try:
            execution_task = self.resource_ctrl.create_task(
                spec.subagent_id,
                self.run_subagent(spec),
                owner_id=self.owner_id,
            )
            result = await execution_task
        except asyncio.CancelledError:
            result = SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.AGENT_AS_TOOL,
                status=SubagentStatus.CANCELLED,
                summary="子代理执行已取消。",
            )
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
        except TimeoutError:
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
        finally:
            await self.isolation.close_session(session)
