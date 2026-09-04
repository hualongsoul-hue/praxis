"""Runtime-wide concurrency leases and child-task ownership."""

import asyncio
import os
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from praxis.config.schemas import SubagentConfig
from praxis.lifecycle import TaskSupervisor
from praxis.telemetry.logger import get_logger

log = get_logger("resources")
T = TypeVar("T")


@dataclass(slots=True)
class KeyedWriteLock:
    """Reference-counted lock for one mutable external resource."""

    lock: asyncio.Lock
    users: int = 0


class ResourceController:
    """Runtime-wide subagent, readonly-tool, write, and task-group limits."""

    def __init__(
        self,
        config: SubagentConfig,
        supervisor: TaskSupervisor | None = None,
        max_concurrent_readonly: int = 10,
    ) -> None:
        if max_concurrent_readonly < 1:
            raise ValueError("max_concurrent_readonly 必须大于 0")
        self.config = config
        self.supervisor = supervisor
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.read_semaphore = asyncio.Semaphore(max_concurrent_readonly)
        self.available_permits = config.max_concurrent
        self.running: dict[str, asyncio.Task[Any]] = {}
        self.task_owners: dict[str, str] = {}
        self.owner_tasks: dict[str, set[asyncio.Task[Any]]] = {}
        self.write_locks: dict[str, KeyedWriteLock] = {}
        self.write_locks_guard = asyncio.Lock()

    @asynccontextmanager
    async def read_lease(self) -> AsyncGenerator[None, None]:
        """Acquire one Runtime-wide readonly tool permit."""
        async with self.read_semaphore:
            yield

    @asynccontextmanager
    async def write_lease(self, resource_key: str) -> AsyncGenerator[None, None]:
        """Serialize writes targeting the same normalized resource key."""
        async with self.write_locks_guard:
            state = self.write_locks.get(resource_key)
            if state is None:
                state = KeyedWriteLock(lock=asyncio.Lock())
                self.write_locks[resource_key] = state
            state.users += 1
        try:
            async with state.lock:
                yield
        finally:
            async with self.write_locks_guard:
                state.users -= 1
                if state.users == 0 and not state.lock.locked():
                    self.write_locks.pop(resource_key, None)

    @staticmethod
    def write_resource_key(name: str, arguments: dict[str, Any]) -> str:
        """Derive a stable lock key without including mutable payload content."""
        for field in ("file_path", "path", "directory", "root", "cwd"):
            value = arguments.get(field)
            if isinstance(value, str) and value:
                return f"path:{os.path.normcase(os.path.abspath(value))}"
        namespace = arguments.get("namespace")
        key = arguments.get("key")
        if isinstance(namespace, str) and isinstance(key, str):
            return f"storage:{namespace}:{key}"
        url = arguments.get("url")
        if isinstance(url, str) and url:
            return f"url:{url}"
        return f"tool:{name}"

    async def acquire(self, subagent_id: str) -> bool:
        """Acquire one Runtime-wide subagent permit."""
        await self.semaphore.acquire()
        self.available_permits -= 1
        log.info("并发槽位已分配", subagent_id=subagent_id)
        return True

    def release(self, subagent_id: str) -> None:
        """Release a previously acquired subagent permit."""
        self.semaphore.release()
        self.available_permits += 1
        task = self.running.get(subagent_id)
        if task is not None:
            self.discard_completed(subagent_id, task)
        log.info("并发槽位已释放", subagent_id=subagent_id)

    def register_task(
        self,
        subagent_id: str,
        task: asyncio.Task[Any],
        owner_id: str = "",
    ) -> None:
        """Register a child task and its parent Session group."""
        self.running[subagent_id] = task
        if owner_id:
            self.task_owners[subagent_id] = owner_id
            self.owner_tasks.setdefault(owner_id, set()).add(task)

    def create_task(
        self,
        subagent_id: str,
        coroutine: Coroutine[Any, Any, T],
        owner_id: str = "",
    ) -> asyncio.Task[T]:
        """Create and register a child task under the Runtime supervisor."""
        if self.supervisor is None:
            task = asyncio.create_task(coroutine, name=f"praxis-subagent-{subagent_id}")
        else:
            task = self.supervisor.create_task(
                coroutine,
                name=f"praxis-subagent-{subagent_id}",
            )
        self.register_task(subagent_id, task, owner_id)
        task.add_done_callback(
            lambda completed: self.discard_completed(subagent_id, completed)
        )
        return task

    def discard_completed(
        self,
        subagent_id: str,
        task: asyncio.Task[Any],
    ) -> None:
        """Remove one completed task from all ownership indexes."""
        if self.running.get(subagent_id) is task:
            self.running.pop(subagent_id, None)
        owner_id = self.task_owners.pop(subagent_id, "")
        if owner_id:
            tasks = self.owner_tasks.get(owner_id)
            if tasks is not None:
                tasks.discard(task)
                if not tasks:
                    self.owner_tasks.pop(owner_id, None)

    def cancel_task(self, subagent_id: str) -> bool:
        """Cancel one tracked child task."""
        task = self.running.get(subagent_id)
        if task is None:
            return False
        task.cancel()
        return True

    async def cancel_group(self, owner_id: str) -> None:
        """Cancel and join every child task created by one parent Session."""
        tasks = list(self.owner_tasks.get(owner_id, set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        """Cancel and join all tasks still tracked by the controller."""
        tasks = list({*self.running.values()})
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def active_count(self) -> int:
        """Return the number of tracked child tasks."""
        return len(self.running)

    @property
    def available_slots(self) -> int:
        """Return currently available subagent permits."""
        return self.available_permits
