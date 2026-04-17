"""praxis.config — 配置系统（S1）：全局配置加载、验证、热更新。"""

from praxis.config.loader import get_subsystem_config, load_config, reload_config
from praxis.config.settings import PraxisConfig
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
from praxis.config.validation import on_config_change, register_validator

__all__ = [
    "ContextConfig",
    "GatewayConfig",
    "GuardrailsConfig",
    "LifecycleConfig",
    "MemoryConfig",
    "OrchestratorConfig",
    "PersistenceConfig",
    "PraxisConfig",
    "RecoveryConfig",
    "SkillsConfig",
    "SubagentConfig",
    "TelemetryConfig",
    "ToolsConfig",
    "VerificationConfig",
    "get_subsystem_config",
    "load_config",
    "on_config_change",
    "register_validator",
    "reload_config",
]
