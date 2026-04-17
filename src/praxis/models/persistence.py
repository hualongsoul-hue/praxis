"""持久化相关数据模型——S3 检查点。"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Checkpoint(BaseModel):
    """Agent 状态快照检查点。

    包含完整 Agent 状态（消息历史、记忆状态、轮次计数器、工具状态）。
    """

    checkpoint_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    session_id: str
    state: dict[str, Any]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    description: str | None = None
