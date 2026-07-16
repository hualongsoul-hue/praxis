"""不可变的 Praxis 顶层配置模型。"""

from pydantic import ConfigDict, Field

from praxis.config.schemas import (
    ContextConfig,
    GatewayConfig,
    GuardrailsConfig,
    InputConfig,
    MCPConfig,
    MemoryConfig,
    OrchestratorConfig,
    PersistenceConfig,
    RecoveryConfig,
    SessionConfig,
    SkillsConfig,
    StrictConfigModel,
    SubagentConfig,
    TelemetryConfig,
    ToolsConfig,
    VerificationConfig,
)


class PraxisConfig(StrictConfigModel):
    """应用配置快照；加载后不可变，可供多个 Runtime 安全共享。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    inputs: InputConfig = Field(default_factory=InputConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    subagent: SubagentConfig = Field(default_factory=SubagentConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
