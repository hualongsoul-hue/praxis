"""各组件配置 Schema 定义。

每个组件拥有独立的配置模型，所有字段带默认值，确保零配置可启动。
配置 Schema 由此模块集中定义，通过 PraxisConfig 组合为顶层配置树。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TelemetryConfig(BaseModel):
    """S2 遥测系统配置。"""

    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"
    log_levels: dict[str, str] = Field(
        default_factory=dict,
        description="按组件独立设置日志级别，如 {'gateway': 'DEBUG'}",
    )
    metrics_enabled: bool = True
    metrics_export: Literal["prometheus", "file"] = "file"
    metrics_file: str | None = None
    tracing_enabled: bool = True
    tracing_export: Literal["otlp", "console"] = "console"
    audit_enabled: bool = True


class PersistenceConfig(BaseModel):
    """S3 持久化引擎配置。"""

    backend: Literal["sqlite", "redis", "filesystem"] = "sqlite"
    sqlite_path: str = "data/praxis.db"
    redis_url: str | None = None
    filesystem_path: str = "data/storage"


class GatewayConfig(BaseModel):
    """S4 模型网关配置。

    model_list 格式与 LiteLLM Router model_list 对齐，直接透传。
    """

    model_list: list[dict[str, Any]] = Field(default_factory=list)
    default_model: str = "default"
    routing_strategy: str = "simple-shuffle"
    num_retries: int = 3
    timeout: float = 60.0
    max_budget: float | None = Field(
        default=None,
        description="成本预算上限（USD），None 表示无限制",
    )


class ToolsConfig(BaseModel):
    """S5 工具系统配置。"""

    allowed_paths: list[str] = Field(
        default_factory=list,
        description="沙箱文件系统白名单路径",
    )
    default_timeout: float = 30.0
    max_concurrent_readonly: int = 5
    shell_timeout: float = 120.0
    network_allowed: bool = True


class MemoryConfig(BaseModel):
    """S6 记忆系统配置。"""

    vector_dimensions: int = 1536
    background_batch_threshold: int = 5
    background_interval_seconds: float = 10.0
    dream_min_hours: float = 24.0
    dream_min_sessions: int = 5
    max_memories: int = 10000
    decay_enabled: bool = True


class ContextConfig(BaseModel):
    """S7 上下文引擎配置。"""

    compaction_threshold: float = Field(
        default=0.8,
        description="Token 占比超过此阈值触发上下文压缩",
    )
    masking_turn_distance: int = 10
    masking_token_threshold: int = 2000
    recent_file_refs_keep: int = 5


class GuardrailsConfig(BaseModel):
    """S8 护栏系统配置。"""

    default_permission: Literal["auto_approve", "confirm", "deny"] = "confirm"
    permissions_file: str | None = None
    input_guardrails_enabled: bool = True
    output_guardrails_enabled: bool = True


class RecoveryConfig(BaseModel):
    """S9 错误恢复配置。"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown: float = 60.0


class VerificationConfig(BaseModel):
    """S10 验证引擎配置。"""

    computational_enabled: bool = True
    inferential_enabled: bool = True
    visual_enabled: bool = False


class OrchestratorConfig(BaseModel):
    """S11 编排循环配置。"""

    max_turns: int = 100
    default_strategy: Literal["react", "plan-and-execute"] = "react"
    stream_events: bool = True


class SessionConfig(BaseModel):
    """S12 会话管理配置。"""

    auto_checkpoint: bool = True
    max_checkpoints_per_session: int = 50


class SubagentConfig(BaseModel):
    """S13 子代理协调配置。"""

    max_concurrent: int = 5
    default_max_turns: int = 50
    default_timeout: float = 300.0


class SkillsConfig(BaseModel):
    """S14 技能系统配置。"""

    skill_paths: list[str] = Field(
        default_factory=lambda: [".praxis/skills", "~/.praxis/skills"],
    )
    auto_discover: bool = True
    max_skills_in_context: int = 10
