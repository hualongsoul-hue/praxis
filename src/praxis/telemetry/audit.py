"""Runtime 实例级、同步持久化、只追加的审计服务。"""

import asyncio
from typing import Any, cast

from praxis.exceptions import TelemetryError
from praxis.models.telemetry import AuditEvent
from praxis.persistence.store import PersistenceStore
from praxis.telemetry.redaction import redact_observability_value

AUDIT_NAMESPACE = "audit"
def redact_audit_value(value: object) -> object:
    """Apply the shared immutable observability redaction policy."""
    return redact_observability_value(value)


class AuditService:
    """关键事件在返回前落盘；同一事件 ID 永不覆盖。"""

    def __init__(self, store: PersistenceStore, *, enabled: bool = True) -> None:
        self.store = store
        self.enabled_state = enabled
        self.closed = False
        self.lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.enabled_state and not self.closed

    async def record(self, event: AuditEvent) -> None:
        if not self.enabled:
            return
        safe_event = event.model_copy(
            update={"details": cast(dict[str, Any], redact_audit_value(event.details))},
            deep=True,
        )
        async with self.lock:
            try:
                created = await self.store.save_if_absent(
                    AUDIT_NAMESPACE,
                    safe_event.event_id,
                    safe_event.model_dump(mode="json"),
                )
            except Exception as exc:
                raise TelemetryError("审计事件持久化失败") from exc
            if not created:
                raise TelemetryError(
                    "审计事件已存在，拒绝覆盖",
                    details={"event_id": safe_event.event_id},
                )

    async def query(self, prefix: str | None = None) -> list[AuditEvent]:
        keys = await self.store.list_keys(AUDIT_NAMESPACE, prefix)
        events: list[AuditEvent] = []
        for key in keys:
            data = await self.store.load(AUDIT_NAMESPACE, key)
            if data is not None:
                events.append(AuditEvent.model_validate(data))
        return sorted(events, key=lambda item: item.timestamp)

    async def flush(self) -> None:
        """写入是同步的；获取锁即可确认在途事件已经完成。"""
        async with self.lock:
            return

    async def close(self) -> None:
        if self.closed:
            return
        await self.flush()
        self.closed = True


class NullAuditSink:
    """显式禁用审计时使用的无状态实现。"""

    async def record(self, event: AuditEvent) -> None:
        return

    async def flush(self) -> None:
        return

    async def close(self) -> None:
        return
