"""审计日志。

记录不可篡改的操作记录（工具调用、LLM 调用、权限决策），
独立于常规日志，通过 S3 持久化引擎存储。
"""

from praxis.models.telemetry import AuditEvent
from praxis.persistence.store import PersistenceStore

AUDIT_NAMESPACE = "audit"

audit_store: PersistenceStore | None = None


def configure_audit(store: PersistenceStore) -> None:
    """配置审计日志的持久化存储。由 S12 生命周期管理在初始化时调用。"""
    global audit_store
    audit_store = store


async def record_audit(event: AuditEvent) -> None:
    """记录一条审计事件。

    如果审计存储未配置，事件将被静默丢弃（不影响业务流程）。

    Args:
        event: 审计事件对象。
    """
    if audit_store is None:
        return
    await audit_store.save(
        AUDIT_NAMESPACE,
        event.event_id,
        event.model_dump(mode="json"),
    )


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
