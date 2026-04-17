"""验证引擎数据模型——S10 跨子系统共享类型。"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    """验证结果状态。"""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class VerificationType(str, Enum):
    """验证类型。"""

    COMPUTATIONAL = "computational"
    INFERENTIAL = "inferential"
    VISUAL = "visual"


class QualityPhase(str, Enum):
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
    failures: list[FailureDetail] = Field(default_factory=list)
    feedback: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed(self) -> bool:
        """是否通过验证。"""
        return self.status == VerificationStatus.PASS
