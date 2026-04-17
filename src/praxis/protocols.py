"""Praxis 通用接口协议。

定义跨子系统共享的 Protocol，用于类型检查和依赖注入。
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Exportable(Protocol):
    """可导出/导入状态的子系统协议。

    实现者：S6（记忆系统）、S7（上下文引擎）。
    消费者：S12（生命周期管理）在检查点保存/恢复时调用。
    """

    async def export_state(self) -> dict[str, Any]: ...

    async def import_state(self, state: dict[str, Any]) -> None: ...


@runtime_checkable
class SessionAware(Protocol):
    """感知会话生命周期的子系统协议。

    实现者：S6（记忆系统）。
    消费者：S12（生命周期管理）在会话终止时调用。
    """

    async def clear_session(self) -> None: ...
