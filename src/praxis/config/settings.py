"""PraxisConfig 顶层配置模型。

配置来源优先级（从低到高）：默认值 → YAML 文件 → 环境变量 → 初始化参数。
YAML 数据通过 set_yaml_data() 注入，由 loader 在 load_config 时调用。
"""

from typing import Any

from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from praxis.config.subsystems import (
    ContextConfig,
    GatewayConfig,
    GuardrailsConfig,
    LifecycleConfig,
    MemoryConfig,
    OrchestratorConfig,
    PersistenceConfig,
    RecoveryConfig,
    SkillsConfig,
    SubagentConfig,
    TelemetryConfig,
    ToolsConfig,
    VerificationConfig,
)

yaml_data: dict[str, Any] = {}


def set_yaml_data(data: dict[str, Any]) -> None:
    """设置 YAML 配置数据，供 YamlFileSource 读取。"""
    global yaml_data
    yaml_data = data


class YamlFileSource(PydanticBaseSettingsSource):
    """YAML 配置源。优先级介于环境变量和默认值之间。"""

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        value = yaml_data.get(field_name)
        return value, field_name, False

    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        return value

    def __call__(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for field_name, field_info in self.settings_cls.model_fields.items():
            value, key, is_complex = self.get_field_value(field_info, field_name)
            value = self.prepare_field_value(field_name, field_info, value, is_complex)
            if value is not None:
                d[key] = value
        return d


class PraxisConfig(BaseSettings):
    """Praxis 顶层配置，包含所有子系统配置切片。

    环境变量前缀：``PRAXIS_``，嵌套分隔符：``__``。
    示例：``PRAXIS_TELEMETRY__LOG_LEVEL=DEBUG``。
    """

    model_config = SettingsConfigDict(
        env_prefix="PRAXIS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    subagent: SubagentConfig = Field(default_factory=SubagentConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlFileSource(settings_cls),
        )
