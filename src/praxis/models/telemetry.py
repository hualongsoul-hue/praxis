"""遥测相关数据模型——S2 审计日志事件。"""

from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator

from praxis.models.base import SafeBaseModel


class AuditEvent(SafeBaseModel):
    """不可篡改的审计事件记录。

    事件类型：
    - ``tool_call``：工具调用（时间、工具名、参数摘要、结果、权限判定）
    - ``llm_call``：LLM 调用（模型、Token 用量、是否触发护栏）
    - ``permission_decision``：权限决策（允许/拒绝/需确认、触发规则）
    - ``guardrail_verdict``：护栏裁决（输入/工具/输出检测结果）
    - ``recovery_event``：错误恢复事件（熔断器状态转换、降级触发）
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    event_type: Literal[
        "tool_call",
        "llm_call",
        "permission_decision",
        "guardrail_verdict",
        "recovery_event",
    ]
    component: str
    action: str = ""
    runtime_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("details", mode="before")
    @classmethod
    def redact_details(cls, value: object) -> dict[str, Any]:
        from praxis.telemetry.redaction import redact_observability_value

        redacted = redact_observability_value(value)
        return cast(dict[str, Any], redacted) if isinstance(redacted, dict) else {}
