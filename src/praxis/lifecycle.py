"""Runtime 后台任务监督器。"""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


class TaskSupervisor:
    """跟踪、取消并回收 Runtime 创建的全部后台任务。"""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._failures: list[BaseException] = []
        self._closed = False

    @property
    def healthy(self) -> bool:
        return not self._failures and not self._closed

    @property
    def failures(self) -> tuple[BaseException, ...]:
        return tuple(self._failures)

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, _T],
        *,
        name: str,
    ) -> asyncio.Task[_T]:
        if self._closed:
            coroutine.close()
            raise RuntimeError("任务监督器已关闭")
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._failures.append(error)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
