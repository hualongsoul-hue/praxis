"""资源管控。

并发数上限（默认 5），独立 Token 预算和轮次上限，
超时控制（超时强制终止返回部分结果）。
"""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from praxis.config.schemas import SubagentConfig
from praxis.lifecycle import TaskSupervisor
from praxis.telemetry.logger import get_logger

log = get_logger("subagent.resource_control")
_T = TypeVar("_T")


class ResourceController:
    """子代理资源管控器。

    限制并发数、追踪运行中子代理、提供超时控制。
    """

    def __init__(
        self,
        config: SubagentConfig,
        supervisor: TaskSupervisor | None = None,
    ) -> None:
        self.config = config
        self.supervisor = supervisor
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.running: dict[str, asyncio.Task[Any]] = {}

    async def acquire(self, subagent_id: str) -> bool:
        """获取并发槽位。

        Args:
            subagent_id: 子代理 ID。

        Returns:
            是否成功获取。
        """
        await self.semaphore.acquire()
        log.info("并发槽位已分配", subagent_id=subagent_id)
        return True

    def release(self, subagent_id: str) -> None:
        """释放并发槽位。"""
        self.semaphore.release()
        self.running.pop(subagent_id, None)
        log.info("并发槽位已释放", subagent_id=subagent_id)

    def register_task(self, subagent_id: str, task: asyncio.Task[Any]) -> None:
        """注册运行中的子代理任务。"""
        self.running[subagent_id] = task

    def create_task(
        self,
        subagent_id: str,
        coroutine: Coroutine[Any, Any, _T],
    ) -> asyncio.Task[_T]:
        """Create and register a child task under the Runtime supervisor."""
        if self.supervisor is None:
            task = asyncio.create_task(coroutine, name=f"praxis-subagent-{subagent_id}")
        else:
            task = self.supervisor.create_task(
                coroutine,
                name=f"praxis-subagent-{subagent_id}",
            )
        self.register_task(subagent_id, task)
        task.add_done_callback(
            lambda completed: self._discard_completed(subagent_id, completed)
        )
        return task

    def _discard_completed(
        self,
        subagent_id: str,
        task: asyncio.Task[Any],
    ) -> None:
        if self.running.get(subagent_id) is task:
            self.running.pop(subagent_id, None)

    def cancel_task(self, subagent_id: str) -> bool:
        """取消指定子代理任务。"""
        task = self.running.get(subagent_id)
        if task is None:
            return False
        task.cancel()
        return True

    @property
    def active_count(self) -> int:
        """当前运行中的子代理数量。"""
        return len(self.running)

    @property
    def available_slots(self) -> int:
        """当前可用并发槽位数（基于信号量实际剩余值）。"""
        return self.semaphore._value
