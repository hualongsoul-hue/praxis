"""各组件配置 Schema 定义。

每个组件拥有独立的配置模型，所有字段带默认值，确保零配置可启动。
配置 Schema 由此模块集中定义，通过 PraxisConfig 组合为顶层配置树。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]


class StrictConfigModel(BaseModel):
    """拒绝未知字段的配置基类，避免拼写错误被静默忽略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelCapabilities(StrictConfigModel):
    """Attachment modalities accepted by a model deployment."""

    image: bool = False
    audio: bool = False
    video: bool = False
    file: bool = False


class InputConfig(StrictConfigModel):
    """Fail-closed attachment resolution policy."""

    allowed_paths: list[str] = Field(default_factory=list)
    remote_enabled: bool = False
    allow_private_networks: bool = False
    max_attachment_bytes: int = Field(default=20_000_000, ge=1, le=100_000_000)
    max_total_bytes: int = Field(default=50_000_000, ge=1, le=500_000_000)
    max_redirects: int = Field(default=5, ge=0, le=20)
    remote_timeout: float = Field(default=30.0, gt=0, le=300.0)
    image_media_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/gif", "image/webp"]
    )
    audio_media_types: list[str] = Field(default_factory=lambda: ["audio/mpeg", "audio/wav"])
    video_media_types: list[str] = Field(
        default_factory=lambda: ["video/mp4", "video/webm", "video/quicktime"]
    )
    file_media_types: list[str] = Field(
        default_factory=lambda: [
            "application/pdf",
            "application/json",
            "application/yaml",
            "text/csv",
            "text/markdown",
            "text/plain",
        ]
    )


class TelemetryConfig(StrictConfigModel):
    """S2 遥测系统配置。"""

    log_level: LogLevel = "INFO"
    log_format: Literal["json", "text"] = "text"
    log_file: str | None = Field(
        default=None,
        description="日志输出文件路径；None 表示输出到 stderr",
    )
    log_levels: dict[str, LogLevel] = Field(
        default_factory=dict,
        description="按组件独立设置日志级别，如 {'gateway': 'DEBUG'}",
    )
    metrics_enabled: bool = True
    metrics_export: Literal["prometheus", "file"] = "file"
    metrics_file: str | None = None
    metrics_port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="metrics_export=prometheus 时 /metrics 抓取端点的监听端口",
    )
    tracing_enabled: bool = True
    tracing_export: Literal["console", "otlp", "none"] = "console"
    otlp_endpoint: str | None = Field(
        default=None,
        description="tracing_export=otlp 时的 OTLP Collector 端点；None 用 OTEL_EXPORTER_OTLP_ENDPOINT 环境变量",
    )
    audit_enabled: bool = True


class PersistenceConfig(StrictConfigModel):
    """S3 持久化引擎配置。"""

    backend: Literal["sqlite", "redis", "filesystem"] = "sqlite"
    sqlite_path: str = Field(default="data/praxis.db", min_length=1)
    redis_url: str | None = None
    filesystem_path: str = Field(default="data/storage", min_length=1)


class ModelDeployment(StrictConfigModel):
    """类型化模型部署；凭据仅保存环境变量名，不保存密钥值。"""

    model_name: str = Field(default="default", min_length=1)
    model: str = Field(default="openai/glm-5.1-openai", min_length=1)
    api_base: str = Field(default="http://172.24.23.192:3000/v1", min_length=1)
    api_key_env: Literal["PRAXIS_MODEL_API_KEY"] = "PRAXIS_MODEL_API_KEY"
    input_cost_per_token: float | None = Field(default=None, ge=0)
    output_cost_per_token: float | None = Field(default=None, ge=0)
    default_max_output_tokens: int = Field(default=4096, ge=1)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    supports_vision: bool = False


class GatewayConfig(StrictConfigModel):
    """S4 模型网关配置。

    deployments 会在 LiteLLM 适配器边界转换，配置树中永不包含 API Key。
    """

    deployments: list[ModelDeployment] = Field(
        default_factory=lambda: [ModelDeployment()],
        min_length=1,
    )
    default_model: str = "default"
    routing_strategy: Literal[
        "simple-shuffle",
        "least-busy",
        "usage-based-routing",
        "latency-based-routing",
        "cost-based-routing",
        "usage-based-routing-v2",
    ] = "simple-shuffle"
    num_retries: int = Field(default=3, ge=0)
    timeout: float = Field(default=60.0, gt=0)
    max_concurrent_requests: int = Field(default=8, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_budget: float | None = Field(
        default=None,
        ge=0,
        description="成本预算上限（USD），None 表示无限制",
    )

    @model_validator(mode="after")
    def validate_default_model(self) -> "GatewayConfig":
        if self.default_model not in {item.model_name for item in self.deployments}:
            raise ValueError("default_model 必须引用 deployments 中的 model_name")
        return self


class ToolsConfig(StrictConfigModel):
    """S5 工具系统配置。"""

    allowed_paths: list[str] = Field(
        default_factory=list,
        description="沙箱文件系统白名单路径",
    )
    default_timeout: float = Field(default=30.0, gt=0)
    max_concurrent_readonly: int = Field(default=5, ge=1)
    shell_timeout: float = Field(default=120.0, gt=0)
    shell_enabled: bool = False
    shell_environment_allowlist: list[str] = Field(default_factory=list)
    network_allowed: bool = False
    allow_private_networks: bool = False
    network_max_response_bytes: int = Field(default=1_000_000, ge=1, le=100_000_000)
    approval_timeout: float = Field(default=60.0, gt=0)
    fallback_mappings: dict[str, str] = Field(
        default_factory=dict,
        description="工具降级映射（首选工具名 → 降级替代工具名），S9 优雅降级使用",
    )


class MemoryConfig(StrictConfigModel):
    """S6 记忆系统配置。"""

    embedding_api_base: str | None = None
    embedding_model: str | None = None
    embedding_api_key_env: str | None = None
    embedding_timeout: float = Field(default=30.0, gt=0)
    embedding_dimensions: int = Field(default=2560, ge=1)
    extraction_prompts: dict[str, str] = Field(default_factory=dict)
    consolidation_similarity_threshold: float = Field(default=0.75, ge=0, le=1)
    background_enabled: bool = True
    background_batch_threshold: int = Field(default=3, ge=1)
    background_interval_seconds: float = Field(default=10.0, gt=0)
    dream_enabled: bool = True
    dream_min_hours: float = Field(default=24.0, ge=0)
    dream_min_sessions: int = Field(default=5, ge=1)
    dream_check_interval_seconds: float = Field(default=3600.0, gt=0)
    dream_scopes: list[str] = Field(default_factory=lambda: ["global"])
    max_memories: int = Field(default=10000, ge=1)
    decay_enabled: bool = True
    decay_half_life_days: float = Field(default=30.0, gt=0)
    inactivity_threshold_days: float = Field(default=90.0, ge=0)
    project_root: str | None = None
    project_name: str | None = None
    load_project_praxis_md: bool = True


class ContextConfig(StrictConfigModel):
    """S7 上下文引擎配置。"""

    compaction_threshold: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Token 占比超过此阈值触发上下文压缩",
    )
    masking_turn_distance: int = Field(default=10, ge=0)
    masking_token_threshold: int = Field(default=2000, ge=1)
    recent_file_refs_keep: int = Field(default=5, ge=0)
    compaction_min_history: int = Field(
        default=8,
        ge=1,
        description="对话历史长度达到此值才触发观察遮蔽与上下文压缩检查",
    )


class GuardrailsConfig(StrictConfigModel):
    """S8 护栏系统配置。"""

    default_permission: Literal["auto_approve", "confirm", "deny"] = "confirm"
    permissions_file: str | None = None
    input_guardrails_enabled: bool = True
    output_guardrails_enabled: bool = True


class RecoveryConfig(StrictConfigModel):
    """S9 错误恢复配置。"""

    max_retries: int = Field(default=3, ge=0)
    base_delay: float = Field(default=1.0, ge=0)
    max_delay: float = Field(default=30.0, ge=0)
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_cooldown: float = Field(default=60.0, ge=0)


class VerificationConfig(StrictConfigModel):
    """S10 验证引擎配置。"""

    computational_enabled: bool = True
    inferential_enabled: bool = True
    visual_enabled: bool = False


class OrchestratorConfig(StrictConfigModel):
    """S11 编排循环配置。"""

    max_turns: int = Field(default=100, ge=1)
    default_strategy: Literal["react", "plan-and-execute"] = "react"


class SessionConfig(StrictConfigModel):
    """S12 会话管理配置。"""

    auto_checkpoint: bool = True
    max_checkpoints_per_session: int = Field(default=50, ge=0)


class SubagentConfig(StrictConfigModel):
    """S13 子代理协调配置。"""

    max_concurrent: int = Field(default=5, ge=1)
    default_max_turns: int = Field(default=50, ge=1)
    default_timeout: float = Field(default=300.0, gt=0)


class SkillsConfig(StrictConfigModel):
    """S14 技能系统配置。"""

    skill_paths: list[str] = Field(
        default_factory=lambda: [".praxis/skills", "~/.praxis/skills"],
    )
    auto_discover: bool = True
    max_skills_in_context: int = Field(default=10, ge=1)


class MCPConfig(StrictConfigModel):
    """MCP 可选能力配置。未启用时不加载 MCP 依赖或创建连接。"""

    enabled: bool = False
    connect_timeout: float = Field(default=30.0, gt=0)
    sampling_enabled: bool = True
