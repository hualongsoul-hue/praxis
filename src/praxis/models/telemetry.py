"""遥测相关数据模型——S2 审计日志事件。"""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    """不可篡改的审计事件记录。

    事件类型：
    - ``tool_call``：工具调用（时间、工具名、参数摘要、结果、权限判定）
    - ``llm_call``：LLM 调用（模型、Token 用量、是否触发护栏）
    - ``permission_decision``：权限决策（允许/拒绝/需确认、触发规则）
    - ``guardrail_verdict``：护栏裁决（输入/工具/输出检测结果）
    - ``recovery_event``：错误恢复事件（熔断器状态转换、降级触发）
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    event_type: Literal[
        "tool_call",
        "llm_call",
        "permission_decision",
        "guardrail_verdict",
        "recovery_event",
    ]
    subsystem: str
    action: str = ""
    session_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
