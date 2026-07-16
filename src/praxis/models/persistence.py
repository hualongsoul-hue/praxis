"""持久化相关数据模型——S3 检查点。"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, model_validator

from praxis.exceptions import CheckpointCorruptionError, CheckpointVersionError

CHECKPOINT_SCHEMA_VERSION = 1


def checkpoint_checksum(session_id: str, state: dict[str, Any]) -> str:
    payload = json.dumps(
        {"schema_version": CHECKPOINT_SCHEMA_VERSION, "session_id": session_id, "state": state},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Checkpoint(BaseModel):
    """Agent 状态快照检查点。

    包含完整 Agent 状态（消息历史、记忆状态、轮次计数器、工具状态）。
    """

    checkpoint_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    session_id: str
    state: dict[str, Any]
    checksum: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    description: str | None = None

    @model_validator(mode="after")
    def validate_checksum(self) -> "Checkpoint":
        expected = checkpoint_checksum(self.session_id, self.state)
        if self.checksum and self.checksum != expected:
            raise ValueError("checkpoint checksum mismatch")
        if not self.checksum:
            self.checksum = expected
        return self

    @classmethod
    def from_storage(cls, data: object) -> "Checkpoint":
        if not isinstance(data, dict):
            raise CheckpointCorruptionError("检查点根节点不是对象")
        payload = cast(dict[str, object], data)
        version = payload.get("schema_version")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointVersionError(
                f"不支持的检查点 schema 版本: {version!r}",
                details={"supported": CHECKPOINT_SCHEMA_VERSION, "actual": version},
            )
        checksum = payload.get("checksum")
        if not isinstance(checksum, str) or not checksum:
            raise CheckpointCorruptionError("检查点缺少校验和")
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise CheckpointCorruptionError("检查点结构或校验和无效") from exc
