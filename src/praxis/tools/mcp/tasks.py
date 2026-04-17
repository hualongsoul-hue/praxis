"""MCP Tasks（实验性）。

长时间运行的服务器端操作：状态跟踪、进度通知、取消。
"""

from typing import Any

from praxis.models.mcp import MCPTaskInfo, MCPTaskStatus
from praxis.telemetry.logger import get_logger

log = get_logger("tools.mcp.tasks")


class MCPTaskManager:
    """MCP Task 管理器。

    跟踪 MCP Server 端长时间运行的操作。
    """

    def __init__(self) -> None:
        self.tasks: dict[str, MCPTaskInfo] = {}

    def register_task(
        self,
        task_id: str,
        server_name: str,
    ) -> MCPTaskInfo:
        """注册新 Task。

        Args:
            task_id: Task 标识。
            server_name: 服务器名称。

        Returns:
            Task 信息。
        """
        task = MCPTaskInfo(
            task_id=task_id,
            server_name=server_name,
            status=MCPTaskStatus.PENDING,
        )
        self.tasks[task_id] = task
        log.info("Task 已注册", task_id=task_id, server=server_name)
        return task

    def update_status(
        self,
        task_id: str,
        status: MCPTaskStatus,
        progress: float | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> MCPTaskInfo | None:
        """更新 Task 状态。

        Args:
            task_id: Task 标识。
            status: 新状态。
            progress: 进度（0.0~1.0）。
            result: 完成结果。
            error: 错误信息。

        Returns:
            更新后的 Task 信息，不存在时返回 None。
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        task.status = status
        if progress is not None:
            task.progress = progress
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error

        log.info("Task 状态更新", task_id=task_id, status=status.value)
        return task

    def get_task(self, task_id: str) -> MCPTaskInfo | None:
        """获取 Task 信息。"""
        return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """取消 Task。

        Args:
            task_id: Task 标识。

        Returns:
            是否成功取消。
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False
        if task.status in (MCPTaskStatus.COMPLETED, MCPTaskStatus.FAILED, MCPTaskStatus.CANCELLED):
            return False
        task.status = MCPTaskStatus.CANCELLED
        log.info("Task 已取消", task_id=task_id)
        return True

    def list_tasks(
        self,
        server_name: str | None = None,
        status: MCPTaskStatus | None = None,
    ) -> list[MCPTaskInfo]:
        """列出 Task。

        Args:
            server_name: 可选服务器过滤。
            status: 可选状态过滤。

        Returns:
            匹配的 Task 列表。
        """
        results: list[MCPTaskInfo] = []
        for task in self.tasks.values():
            if server_name and task.server_name != server_name:
                continue
            if status and task.status != status:
                continue
            results.append(task)
        return results

    def cleanup_completed(self) -> int:
        """清理已完成/失败/取消的 Task。

        Returns:
            清理数量。
        """
        terminal = {MCPTaskStatus.COMPLETED, MCPTaskStatus.FAILED, MCPTaskStatus.CANCELLED}
        to_remove = [tid for tid, t in self.tasks.items() if t.status in terminal]
        for tid in to_remove:
            del self.tasks[tid]
        return len(to_remove)
