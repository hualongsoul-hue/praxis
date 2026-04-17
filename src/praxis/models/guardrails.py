"""护栏系统数据模型——S8 内部及跨组件共享。"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VerdictType(str, Enum):
    """护栏裁决类型。"""

    PASS = "pass"
    BLOCK = "block"
    AUTO_APPROVE = "auto_approve"
    CONFIRM = "confirm"
    DENY = "deny"


class GuardrailVerdict(BaseModel):
    """护栏裁决结果。"""

    verdict: VerdictType
    reason: str = ""
    tripwire: bool = False
    rule_name: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
