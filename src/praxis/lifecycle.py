"""Runtime 后台任务监督器。"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")
AsyncCloser = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class OwnedResource:
    """One named async resource whose lifecycle was transferred to an owner."""

    name: str
    close: AsyncCloser


class AsyncResourceOwner:
    """Close registered resources in reverse order without short-circuiting."""

    def __init__(self) -> None:
        self.resources: list[OwnedResource] = []
        self.closed = False

    def register(self, name: str, close: AsyncCloser) -> None:
        if self.closed:
            raise RuntimeError("资源所有者已关闭")
        self.resources.append(OwnedResource(name=name, close=close))

    async def close(self) -> tuple[BaseException, ...]:
        if self.closed:
            return ()
        self.closed = True
        resources = list(reversed(self.resources))
        self.resources.clear()
        failures: list[BaseException] = []
        for resource in resources:
            try:
                await resource.close()
            except BaseException as exc:
                failures.append(exc)
        return tuple(failures)


class TaskSupervisor:
    """跟踪、取消并回收 Runtime 创建的全部后台任务。"""

    def __init__(self) -> None:
        self.tasks: set[asyncio.Task[Any]] = set()
        self.task_failures: list[BaseException] = []
        self.closed = False

    @property
    def healthy(self) -> bool:
        return not self.task_failures and not self.closed

    @property
    def failures(self) -> tuple[BaseException, ...]:
        return tuple(self.task_failures)

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        name: str,
    ) -> asyncio.Task[T]:
        if self.closed:
            coroutine.close()
            raise RuntimeError("任务监督器已关闭")
        task = asyncio.create_task(coroutine, name=name)
        self.tasks.add(task)
        task.add_done_callback(self.handle_done)
        return task

    def handle_done(self, task: asyncio.Task[Any]) -> None:
        self.tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.task_failures.append(error)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
