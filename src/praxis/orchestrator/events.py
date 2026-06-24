"""事件发射系统。

在编排循环关键节点发射结构化事件，
通过 S2 记录并通过流式接口实时推送。
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from praxis.models.orchestrator import AgentEvent
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("orchestrator.events")

EVENT_TYPES = frozenset({
    "turn_start",
    "llm_request",
    "content_delta",
    "reasoning_delta",
    "llm_response",
    "tool_call_start",
    "tool_call_end",
    "tool_retry",
    "verification_result",
    "turn_end",
    "termination",
})


class EventEmitter:
    """事件发射器。

    收集编排循环事件，支持同步记录和异步流式推送。
    """

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.listeners: list["EventListener"] = []

    def emit(
        self,
        event_type: str,
        turn: int = 0,
        data: dict[str, Any] | None = None,
    ) -> AgentEvent:
        """发射事件。

        Args:
            event_type: 事件类型。
            turn: 当前轮次号。
            data: 事件数据。

        Returns:
            发射的事件。
        """
        event = AgentEvent(
            event_type=event_type,
            turn=turn,
            data=data or {},
            timestamp=time.time(),
        )
        self.events.append(event)

        # 仅以 event_type 作标签；turn 是无界值，不入指标标签（避免基数爆炸）
        emit_metric(
            "orchestrator_event",
            1.0,
            {"event_type": event_type},
            "counter",
        )

        for listener in self.listeners:
            listener.on_event(event)

        return event

    def add_listener(self, listener: "EventListener") -> None:
        """添加事件监听器。"""
        self.listeners.append(listener)

    def remove_listener(self, listener: "EventListener") -> None:
        """移除事件监听器。"""
        if listener in self.listeners:
            self.listeners.remove(listener)

    def get_events(self) -> list[AgentEvent]:
        """获取所有事件。"""
        return list(self.events)

    def clear(self) -> None:
        """清空事件列表。"""
        self.events.clear()


class EventListener:
    """事件监听器基类。"""

    def on_event(self, event: AgentEvent) -> None:
        """事件回调。子类覆盖此方法处理事件。"""


class StreamCollector(EventListener):
    """流式收集器。

    将事件收集到异步队列中，供 run_stream 消费。
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    def on_event(self, event: AgentEvent) -> None:
        self.queue.put_nowait(event)

    def close(self) -> None:
        """发送终止信号。"""
        self.queue.put_nowait(None)

    async def iter_events(self) -> AsyncIterator[AgentEvent]:
        """异步迭代事件流。"""
        while True:
            event = await self.queue.get()
            if event is None:
                break
            yield event
