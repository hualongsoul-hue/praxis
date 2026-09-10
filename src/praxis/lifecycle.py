"""Runtime 后台任务监督器。"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, TypeVar

T = TypeVar("T")
AsyncCloser = Callable[[], Awaitable[None]]


async def close_async_stream(stream: object) -> None:
    """Await a provider's optional close contract in the stream's owning task."""
    close = getattr(stream, "aclose", None)
    if not callable(close):
        close = getattr(stream, "close", None)
    if callable(close):
        result = close()
        if isawaitable(result):
            await result


async def close_provider_stream(stream: object) -> None:
    """Finish a transport close before propagating cancellation, preserving context.

    Only for low-level providers, not task-affine source generators. Python 3.12's
    explicit task context lets provider correlation tokens reset in their original
    Context while the caller is suspended waiting for cancellation-safe cleanup.
    """
    caller = asyncio.current_task()
    task = asyncio.create_task(
        close_async_stream(stream),
        name="praxis-provider-close",
        context=caller.get_context() if caller is not None else None,
    )
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
        except BaseException:
            break
    if cancellation is not None:
        if not task.cancelled() and (error := task.exception()) is not None:
            cancellation.add_note(f"Provider 关闭失败: {type(error).__name__}")
        raise cancellation
    task.result()


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
