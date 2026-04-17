"""跨上下文窗口续接。

两阶段模式（初始化 + 增量进度），
标准热身序列（检查目录 → 读进度文件 → 验证基础功能 → 开始工作），
长时间任务支持（一次一功能、干净状态、增量前进）。
"""

from typing import Any

from praxis.lifecycle.session import Session
from praxis.models.lifecycle import ContinuationPhase, SessionStatus
from praxis.telemetry.logger import get_logger

log = get_logger("lifecycle.continuation")

INIT_SYSTEM_PROMPT = (
    "你正在启动一个新的工作会话。请执行以下初始化步骤：\n"
    "1. 检查工作目录结构\n"
    "2. 创建或更新进度文件（progress.json）\n"
    "3. 列出待完成的功能清单\n"
    "4. 建立初始 Git 提交（如适用）\n"
    "完成后报告初始化状态。"
)

WARMUP_SYSTEM_PROMPT = (
    "你正在恢复一个工作会话。请执行标准热身序列：\n"
    "1. 检查工作目录当前状态\n"
    "2. 读取进度文件和待办列表\n"
    "3. 验证基础功能仍然正常\n"
    "4. 确认当前工作位置后继续未完成的工作\n"
    "请报告恢复状态和下一步计划。"
)


class ContinuationManager:
    """跨上下文窗口续接管理器。

    管理初始化和增量进度两个阶段的系统提示注入和状态转换。
    """

    def __init__(self) -> None:
        self.feature_list: list[dict[str, Any]] = []

    def get_system_prompt(self, phase: ContinuationPhase) -> str:
        """获取当前阶段的系统提示。

        Args:
            phase: 续接阶段。

        Returns:
            阶段专用系统提示。
        """
        if phase == ContinuationPhase.INITIALIZATION:
            return INIT_SYSTEM_PROMPT
        if phase == ContinuationPhase.WARMUP:
            return WARMUP_SYSTEM_PROMPT
        return ""

    def prepare_turn(self, session: Session) -> dict[str, str]:
        """根据会话当前阶段准备轮次参数。

        Args:
            session: 当前会话。

        Returns:
            额外的 run_turn 参数（如 system_prompt_override）。
        """
        phase = session.metadata.continuation_phase
        kwargs: dict[str, str] = {}

        if phase in (ContinuationPhase.INITIALIZATION, ContinuationPhase.WARMUP):
            prompt = self.get_system_prompt(phase)
            if prompt:
                kwargs["developer_instructions"] = prompt
            log.info("续接阶段准备", phase=phase.value)

        return kwargs

    def advance_phase(self, session: Session) -> ContinuationPhase:
        """推进续接阶段。

        初始化 → 工作，热身 → 工作。

        Args:
            session: 当前会话。

        Returns:
            新阶段。
        """
        current = session.metadata.continuation_phase
        if current in (ContinuationPhase.INITIALIZATION, ContinuationPhase.WARMUP):
            session.metadata.continuation_phase = ContinuationPhase.WORKING
            log.info("续接阶段推进", from_phase=current.value, to_phase="working")
        return session.metadata.continuation_phase

    def set_feature_list(self, features: list[dict[str, Any]]) -> None:
        """设置 JSON 功能列表。

        Args:
            features: 功能列表，每项包含 name, status, description。
        """
        self.feature_list = features

    def get_feature_list(self) -> list[dict[str, Any]]:
        """获取功能列表。"""
        return list(self.feature_list)

    def update_feature_status(self, feature_name: str, status: str) -> bool:
        """更新功能状态。

        Args:
            feature_name: 功能名称。
            status: 新状态（pending/in_progress/completed）。

        Returns:
            是否找到并更新。
        """
        for feature in self.feature_list:
            if feature.get("name") == feature_name:
                feature["status"] = status
                return True
        return False

    def get_next_feature(self) -> dict[str, Any] | None:
        """获取下一个待处理功能（一次一个功能原则）。"""
        for feature in self.feature_list:
            if feature.get("status") == "in_progress":
                return feature
        for feature in self.feature_list:
            if feature.get("status") == "pending":
                return feature
        return None

    def get_progress_summary(self) -> str:
        """生成进度摘要文本。"""
        if not self.feature_list:
            return "无功能列表。"
        total = len(self.feature_list)
        completed = sum(1 for f in self.feature_list if f.get("status") == "completed")
        in_progress = sum(1 for f in self.feature_list if f.get("status") == "in_progress")
        pending = total - completed - in_progress
        return (
            f"功能进度: {completed}/{total} 完成, "
            f"{in_progress} 进行中, {pending} 待处理"
        )
