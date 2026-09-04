"""会话管理数据模型——S12 跨组件共享类型。"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from praxis.models.tools import ToolExecutionRecord


class SessionStatus(StrEnum):
    """会话状态。"""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    TERMINATED = "terminated"


class ContinuationPhase(StrEnum):
    """跨窗口续接阶段。"""

    INITIALIZATION = "initialization"
    WARMUP = "warmup"
    WORKING = "working"


class SessionMetadata(BaseModel):
    """会话元数据。"""

    session_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: SessionStatus = SessionStatus.INITIALIZING
    total_turns: int = 0
    total_tokens: int = 0
    checkpoint_count: int = 0
    last_checkpoint_id: str | None = None
    continuation_phase: ContinuationPhase = ContinuationPhase.INITIALIZATION


class CheckpointInfo(BaseModel):
    """检查点摘要信息（用于列表展示）。"""

    checkpoint_id: str
    session_id: str
    created_at: datetime
    turn_number: int = 0
    description: str = ""


class SessionSnapshot(BaseModel):
    """完整会话状态快照（用于检查点保存/恢复）。"""

    metadata: SessionMetadata
    context_state: dict[str, Any] = Field(default_factory=dict)
    memory_state: dict[str, Any] = Field(default_factory=dict)
    loop_state: dict[str, Any] = Field(default_factory=dict)
    strategy_state: dict[str, Any] = Field(default_factory=dict)
    recovery_state: dict[str, Any] = Field(default_factory=dict)
    approval_state: dict[str, Any] = Field(default_factory=dict)
    skill_state: dict[str, Any] = Field(default_factory=dict)
    tool_execution_ledger: dict[str, ToolExecutionRecord] = Field(default_factory=dict)
    file_refs: list[str] = Field(default_factory=list)
