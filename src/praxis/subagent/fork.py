"""Fork 执行模型。

子代理获得父上下文只读副本，并行独立执行，
collect_results 聚合多个子代理输出。
"""

import asyncio
from typing import Any

from praxis.guardrails.engine import GuardrailEngine
from praxis.models.subagent import (
    SubagentMode,
    SubagentResult,
    SubagentSpec,
    SubagentStatus,
)
from praxis.subagent.aggregation import ResultAggregator
from praxis.subagent.isolation import IsolatedContext
from praxis.subagent.resource_control import ResourceController
from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolRegistry

log = get_logger("subagent.fork")


class ForkManager:
    """Fork 执行管理器。

    支持并行创建多个子代理，独立执行后聚合结果。
    """

    def __init__(
        self,
        isolation: IsolatedContext,
        resource_ctrl: ResourceController,
        aggregator: ResultAggregator,
        guardrails: GuardrailEngine,
        parent_registry: ToolRegistry,
        model: str = "default",
    ) -> None:
        self.isolation = isolation
        self.resource_ctrl = resource_ctrl
        self.aggregator = aggregator
        self.guardrails = guardrails
        self.parent_registry = parent_registry
        self.model = model

    async def fork(
        self,
        tasks: list[dict[str, Any]],
        context_summary: str = "",
        timeout_seconds: float = 300.0,
    ) -> list[SubagentResult]:
        """并行 Fork 多个子代理执行不同任务。

        Args:
            tasks: 任务列表，每项包含 task（描述）和可选 tool_names。
            context_summary: 父上下文只读摘要。
            timeout_seconds: 全局超时秒数。

        Returns:
            各子代理结果列表。
        """
        default_max_turns = self.resource_ctrl.config.default_max_turns
        specs: list[SubagentSpec] = []
        for t in tasks:
            specs.append(SubagentSpec(
                task=t["task"],
                mode=SubagentMode.FORK,
                tool_names=t.get("tool_names", []),
                context_summary=context_summary,
                max_turns=t.get("max_turns", default_max_turns),
                timeout_seconds=timeout_seconds,
            ))

        log.info("Fork 启动", fork_count=len(specs))

        execution_tasks = [
            self.resource_ctrl.create_task(spec.subagent_id, self.run_single_fork(spec))
            for spec in specs
        ]
        results = await asyncio.gather(*execution_tasks, return_exceptions=False)
        return list(results)

    async def run_single_fork(self, spec: SubagentSpec) -> SubagentResult:
        """运行单个 Fork 子代理。"""
        await self.resource_ctrl.acquire(spec.subagent_id)

        session = None
        try:
            session = await self.isolation.create_isolated_session(
                spec=spec,
                guardrails=self.guardrails,
                parent_registry=self.parent_registry,
                model=self.model,
            )

            user_message = spec.task
            if spec.context_summary:
                user_message = (
                    f"父代理上下文（只读）:\n{spec.context_summary}\n\n任务:\n{spec.task}"
                )

            response = await asyncio.wait_for(
                session.run_turn(user_message),
                timeout=spec.timeout_seconds,
            )
            return SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.FORK,
                status=SubagentStatus.COMPLETED,
                summary=response.content,
                total_turns=response.total_turns,
            )
        except TimeoutError:
            return SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.FORK,
                status=SubagentStatus.TIMEOUT,
                summary="Fork 子代理超时。",
            )
        except asyncio.CancelledError:
            return SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.FORK,
                status=SubagentStatus.CANCELLED,
                summary="Fork 子代理已取消。",
            )
        except Exception as exc:
            log.error("Fork 子代理失败", subagent_id=spec.subagent_id, error=str(exc))
            return SubagentResult(
                subagent_id=spec.subagent_id,
                mode=SubagentMode.FORK,
                status=SubagentStatus.FAILED,
                summary=f"执行失败: {exc}",
            )
        finally:
            if session is not None:
                await self.isolation.close_session(session)
            self.resource_ctrl.release(spec.subagent_id)

    async def fork_and_aggregate(
        self,
        tasks: list[dict[str, Any]],
        context_summary: str = "",
        timeout_seconds: float = 300.0,
    ) -> dict[str, object]:
        """Fork 多个子代理并聚合结果。

        Args:
            tasks: 任务列表。
            context_summary: 父上下文摘要。
            timeout_seconds: 超时秒数。

        Returns:
            聚合结果。
        """
        results = await self.fork(tasks, context_summary, timeout_seconds)
        return self.aggregator.aggregate(results)
