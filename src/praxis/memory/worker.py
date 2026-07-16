"""后台消息提取 Worker。

随 CognitiveMemory.start() 启动的独立 asyncio.Task，消费 pending 消息队列：
message_id 游标、批量阈值、背压、失败重试。
"""

import asyncio
from collections.abc import Awaitable, Callable

from praxis.models.memory import MemoryScope, WorkingMemoryMessage
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.worker")


# 签名：(messages, scope) -> 处理消息数
ProcessFn = Callable[[list[WorkingMemoryMessage], MemoryScope], Awaitable[int]]


class BackgroundWorker:
    """后台提取/整合 Worker。

    - pending 队列仅追加，成功消费后才推进 last_processed_message_id 游标
    - 失败时保留 pending，下次循环重试
    - 优雅关闭：设置 shutdown 事件 → 等待当前处理完成 → 取消任务
    """

    def __init__(
        self,
        process_fn: ProcessFn,
        scope: MemoryScope,
        batch_threshold: int = 3,
        interval_seconds: float = 10.0,
    ) -> None:
        self.process_fn = process_fn
        self.scope = scope
        self.batch_threshold = max(1, batch_threshold)
        self.interval_seconds = interval_seconds

        self.pending: list[WorkingMemoryMessage] = []
        self.last_processed_message_id: str | None = None
        self.signal: asyncio.Event = asyncio.Event()
        self.idle_event: asyncio.Event = asyncio.Event()
        self.idle_event.set()

        self.task: asyncio.Task[None] | None = None
        self.running: bool = False

    def notify(self, message: WorkingMemoryMessage) -> None:
        """由 append_message 调用，入队并通知 Worker。不阻塞。"""
        self.pending.append(message)
        self.signal.set()

    def clear_pending(self) -> None:
        """会话级清理：清空 pending 与游标（不影响已持久化记忆）。"""
        self.pending.clear()
        self.last_processed_message_id = None

    def start(self) -> None:
        """启动 Worker。若已在运行则幂等。"""
        if self.task is not None and not self.task.done():
            return
        self.running = True
        self.task = asyncio.create_task(self.run_loop())
        log.info("记忆后台 Worker 已启动", scope=self.scope.to_string())

    async def stop(self) -> None:
        """停止 Worker：等待当前处理完成 → 取消任务。"""
        self.running = False
        self.signal.set()

        if self.task is not None and not self.task.done():
            # 等待最多一次循环让当前批处理结束
            try:
                await asyncio.wait_for(self.idle_event.wait(), timeout=30.0)
            except TimeoutError:
                log.warning("等待后台 Worker 空闲超时，强制取消")
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

        log.info("记忆后台 Worker 已停止")

    async def run_loop(self) -> None:
        """主循环：等待信号或超时 → 检查批量阈值 → 处理 pending。"""
        while self.running:
            try:
                await asyncio.wait_for(
                    self.signal.wait(),
                    timeout=self.interval_seconds,
                )
            except TimeoutError:
                pass
            self.signal.clear()

            if not self.running:
                break

            if len(self.pending) < self.batch_threshold:
                continue

            await self.process_once()

    async def process_once(self) -> int:
        """消费一批 pending 消息。成功后按 message_id 推进游标。"""
        snapshot = list(self.pending)
        if not snapshot:
            return 0

        self.idle_event.clear()
        try:
            count = await self.process_fn(snapshot, self.scope)
        except Exception as exc:
            log.error(
                "后台记忆处理失败，保留 pending 以便重试",
                error=str(exc),
                error_type=type(exc).__name__,
                pending_size=len(snapshot),
            )
            emit_metric(
                "memory_background_error",
                1.0,
                {"error_type": type(exc).__name__},
                "counter",
            )
            return 0
        finally:
            self.idle_event.set()

        # 成功消费后：按 message_id 推进游标，从 pending 中移除已处理项
        last_id = snapshot[-1].message_id
        processed_ids = {m.message_id for m in snapshot}
        self.pending = [m for m in self.pending if m.message_id not in processed_ids]
        self.last_processed_message_id = last_id

        log.info(
            "后台处理完成",
            extracted=count,
            cursor=last_id,
            remaining=len(self.pending),
        )
        emit_metric(
            "memory_background_cycle",
            float(count),
            {},
            "counter",
        )
        return count
