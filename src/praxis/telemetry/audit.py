"""审计日志。

记录不可篡改的操作记录（工具调用、LLM 调用、权限决策），
独立于常规日志，通过 S3 持久化引擎存储。

审计写入使用 fire-and-forget 背景任务，避免阻塞热路径（如护栏裁决）。
"""

import asyncio

from praxis.models.telemetry import AuditEvent
from praxis.persistence.store import PersistenceStore
from praxis.telemetry.logger import get_logger

log = get_logger("telemetry.audit")

AUDIT_NAMESPACE = "audit"

audit_store: PersistenceStore | None = None
pending_tasks: set[asyncio.Task[None]] = set()


def configure_audit(store: PersistenceStore) -> None:
    """配置审计日志的持久化存储。由 S12 生命周期管理在初始化时调用。"""
    global audit_store
    audit_store = store


async def write_audit(store: PersistenceStore, event: AuditEvent) -> None:
    """实际写入审计事件到存储（后台任务体）。"""
    try:
        await store.save(
            AUDIT_NAMESPACE,
            event.event_id,
            event.model_dump(mode="json"),
        )
    except Exception as exc:
        log.warning("审计写入失败", event_id=event.event_id, error=str(exc))


async def record_audit(event: AuditEvent) -> None:
    """记录一条审计事件（fire-and-forget，不阻塞调用方）。

    Args:
        event: 审计事件对象。
    """
    if audit_store is None:
        return
    task = asyncio.create_task(write_audit(audit_store, event))
    pending_tasks.add(task)
    task.add_done_callback(pending_tasks.discard)


async def flush_audit() -> None:
    """等待所有 pending 审计写入完成（测试或优雅关闭时调用）。"""
    if not pending_tasks:
        return
    await asyncio.gather(*list(pending_tasks), return_exceptions=True)


async def query_audit(
    prefix: str | None = None,
) -> list[AuditEvent]:
    """查询审计事件。

    Args:
        prefix: 可选事件 ID 前缀过滤。

    Returns:
        匹配的审计事件列表。
    """
    if audit_store is None:
        return []
    keys = await audit_store.list_keys(AUDIT_NAMESPACE, prefix)
    events: list[AuditEvent] = []
    for key in keys:
        data = await audit_store.load(AUDIT_NAMESPACE, key)
        if data is not None:
            events.append(AuditEvent.model_validate(data))
    return events
