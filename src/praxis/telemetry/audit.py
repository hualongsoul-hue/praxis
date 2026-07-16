"""Runtime 实例级、同步持久化、只追加的审计服务。"""

import asyncio
import re
from typing import Any, cast

from praxis.exceptions import TelemetryError
from praxis.models.telemetry import AuditEvent
from praxis.persistence.store import PersistenceStore

AUDIT_NAMESPACE = "audit"
_SECRET_KEYS = frozenset({"api_key", "authorization", "credential", "password", "secret"})
_SECRET_VALUE = re.compile(r"(?i)\b(?:sk|key|token)-[a-z0-9_-]{8,}\b")


def redact_audit_value(value: object) -> object:
    """递归清除审计详情中的凭据；不会修改调用方对象。"""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if str(key).casefold() in _SECRET_KEYS
            else redact_audit_value(item)
            for key, item in cast(dict[object, object], value).items()
        }
    if isinstance(value, list):
        return [redact_audit_value(item) for item in cast(list[object], value)]
    if isinstance(value, tuple):
        return [redact_audit_value(item) for item in cast(tuple[object, ...], value)]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    return value


class AuditService:
    """关键事件在返回前落盘；同一事件 ID 永不覆盖。"""

    def __init__(self, store: PersistenceStore, *, enabled: bool = True) -> None:
        self._store = store
        self._enabled = enabled
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._closed

    async def record(self, event: AuditEvent) -> None:
        if not self.enabled:
            return
        safe_event = event.model_copy(
            update={"details": cast(dict[str, Any], redact_audit_value(event.details))},
            deep=True,
        )
        async with self._lock:
            try:
                created = await self._store.save_if_absent(
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
        keys = await self._store.list_keys(AUDIT_NAMESPACE, prefix)
        events: list[AuditEvent] = []
        for key in keys:
            data = await self._store.load(AUDIT_NAMESPACE, key)
            if data is not None:
                events.append(AuditEvent.model_validate(data))
        return sorted(events, key=lambda item: item.timestamp)

    async def flush(self) -> None:
        """写入是同步的；获取锁即可确认在途事件已经完成。"""
        async with self._lock:
            return

    async def close(self) -> None:
        if self._closed:
            return
        await self.flush()
        self._closed = True


class NullAuditSink:
    """显式禁用审计时使用的无状态实现。"""

    async def record(self, event: AuditEvent) -> None:
        return

    async def flush(self) -> None:
        return

    async def close(self) -> None:
        return
