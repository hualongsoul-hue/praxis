"""后台异步自治任务。

asyncio.Task 随 S6 实例创建自动启动，消息游标持久化，
可配置批量阈值，背压控制（不阻塞 append_message），
优雅关闭（clear_session 等待完成），失败容错（记入 S2，下次重试）。
"""

import asyncio
from typing import Any

from praxis.config.schemas import MemoryConfig
from praxis.memory.pipeline import MemoryPipeline
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.background")


class BackgroundProcessor:
    """后台记忆处理器。

    管理异步任务的生命周期：启动、运行、暂停、恢复、关闭。
    """

    def __init__(
        self,
        pipeline: MemoryPipeline,
        config: MemoryConfig,
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.task: asyncio.Task[None] | None = None
        self.signal: asyncio.Event = asyncio.Event()
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.running: bool = False
        self.processing: bool = False

    def start(self) -> None:
        """启动后台处理任务并订阅 pipeline.append。"""
        if self.task is not None and not self.task.done():
            return
        self.running = True
        self.shutdown_event.clear()
        self.pipeline.on_append = self.notify
        self.task = asyncio.create_task(self.run_loop())
        log.info("后台记忆处理器已启动")

    async def stop(self, wait: bool = True) -> None:
        """停止后台处理任务并解绑 pipeline 订阅。

        Args:
            wait: 是否等待当前处理完成后再停止。
        """
        self.running = False
        self.signal.set()
        if self.pipeline.on_append is self.notify:
            self.pipeline.on_append = None

        if self.task is not None and not self.task.done():
            if wait and self.processing:
                await self.shutdown_event.wait()
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

        log.info("后台记忆处理器已停止")

    def notify(self) -> None:
        """通知有新消息待处理。

        由 append_message 调用，不阻塞。
        """
        self.signal.set()

    async def run_loop(self) -> None:
        """后台任务主循环。

        等待信号 → 检查批量阈值 → 执行提取+整合 → 推进游标。
        """
        while self.running:
            try:
                await asyncio.wait_for(
                    self.signal.wait(),
                    timeout=self.config.background_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

            self.signal.clear()

            if not self.running:
                break

            pending_count = len(self.pipeline.pending_messages)
            if pending_count < self.config.background_batch_threshold:
                continue

            self.processing = True
            try:
                count = await self.pipeline.process_pending()
                if count > 0:
                    log.info("后台处理完成", extracted=count)
                    emit_metric(
                        "memory_background_cycle",
                        float(count),
                        {},
                        "counter",
                    )
            except Exception as exc:
                log.error(
                    "后台记忆处理失败",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                emit_metric(
                    "memory_background_error",
                    1.0,
                    {"error_type": type(exc).__name__},
                    "counter",
                )
            finally:
                self.processing = False

        self.shutdown_event.set()

    def export_state(self) -> dict[str, Any]:
        """导出后台处理器状态。"""
        pipeline_state = self.pipeline.export_state()
        return {
            "pipeline": pipeline_state,
            "running": self.running,
        }

    async def import_state(self, snapshot: dict[str, Any]) -> None:
        """从检查点恢复状态并重启。"""
        pipeline_state = snapshot.get("pipeline", {})
        await self.pipeline.import_state(pipeline_state)
        if snapshot.get("running", False):
            self.start()
