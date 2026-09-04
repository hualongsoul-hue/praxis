"""Runtime 生命周期与健康检查公共模型。"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RuntimeState(StrEnum):
    NEW = "new"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    CLOSED = "closed"
    FAILED = "failed"


class HealthStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


class ComponentHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: HealthStatus
    detail: str = ""
    reason: str = ""
    required: bool = True
    last_probe_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float = Field(default=0.0, ge=0.0)


class RuntimeHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: HealthStatus
    runtime_state: RuntimeState
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
