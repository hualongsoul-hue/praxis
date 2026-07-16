"""验证引擎数据模型——S10 跨组件共享类型。"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class VerificationStatus(StrEnum):
    """验证结果状态。"""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class VerificationType(StrEnum):
    """验证类型。"""

    COMPUTATIONAL = "computational"
    INFERENTIAL = "inferential"
    VISUAL = "visual"


class QualityPhase(StrEnum):
    """质量左移阶段。"""

    PRE_INTEGRATION = "pre_integration"
    POST_INTEGRATION = "post_integration"
    CONTINUOUS_MONITORING = "continuous_monitoring"
    RUNTIME_FEEDBACK = "runtime_feedback"


class FailureDetail(BaseModel):
    """验证失败详情——文件/行号/错误消息。"""

    file: str = ""
    line: int | None = None
    column: int | None = None
    message: str = ""
    severity: str = "error"
    rule: str = ""


class VerificationResult(BaseModel):
    """通用验证结果。"""

    status: VerificationStatus
    verification_type: VerificationType
    verifier_name: str = ""
    score: float | None = None
    failures: list[FailureDetail] = Field(default_factory=lambda: [])
    feedback: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def passed(self) -> bool:
        """是否通过验证。"""
        return self.status == VerificationStatus.PASS
