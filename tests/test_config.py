"""S1 配置系统验证测试。"""

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from praxis.config import (
    PraxisConfig,
    get_component_config,
    load_config,
)
from praxis.config.schemas import GatewayConfig, TelemetryConfig, ToolsConfig
from praxis.exceptions import ConfigError


class TestPraxisConfig:
    """Task 2.1: PraxisConfig 顶层配置模型验证。"""

    def test_default_instantiation(self) -> None:
        config = PraxisConfig()
        assert config.gateway.default_model == "default"
        assert config.gateway.deployments[0].model == "openai/glm-5.1-openai"
        assert config.gateway.deployments[0].api_base == "http://172.24.23.192:3000/v1"
        assert config.telemetry.log_level == "INFO"
        assert config.persistence.backend == "sqlite"
        assert config.orchestrator.max_turns == 100
        assert config.guardrails.default_permission == "confirm"
        assert config.recovery.max_retries == 3
        assert config.subagent.max_concurrent == 5

    def test_all_component_slices_accessible(self) -> None:
        config = PraxisConfig()
        component_names = [
            "telemetry", "persistence", "gateway", "tools", "memory",
            "context", "guardrails", "recovery", "verification",
            "orchestrator", "session", "subagent", "skills",
        ]
        for name in component_names:
            assert hasattr(config, name), f"缺少组件配置: {name}"

    @pytest.mark.parametrize(
        ("config_type", "values"),
        [
            (GatewayConfig, {"defualt_model": "typo"}),
            (ToolsConfig, {"unknown_timeout": 1}),
            (TelemetryConfig, {"unknown_exporter": "stdout"}),
        ],
    )
    def test_component_configs_reject_unknown_fields(
        self,
        config_type: type,
        values: dict[str, object],
    ) -> None:
        with pytest.raises(ValidationError):
            config_type(**values)

    @pytest.mark.parametrize(
        ("config_type", "values"),
        [
            (ToolsConfig, {"default_timeout": 0}),
            (ToolsConfig, {"shell_timeout": -1}),
            (GatewayConfig, {"timeout": 0}),
        ],
    )
    def test_timeouts_must_be_positive(
        self,
        config_type: type,
        values: dict[str, object],
    ) -> None:
        with pytest.raises(ValidationError):
            config_type(**values)

    def test_log_level_must_be_known(self) -> None:
        with pytest.raises(ValidationError):
            TelemetryConfig(log_level="NOT_A_LEVEL")

    def test_root_config_rejects_unknown_sections(self) -> None:
        with pytest.raises(ValidationError):
            PraxisConfig(unknown_section={})


class TestConfigLoading:
    """Task 2.2: 分层配置加载验证。"""

    def test_load_without_yaml(self) -> None:
        config = load_config()
        assert config.telemetry.log_level == "INFO"

    def test_load_with_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            telemetry:
              log_level: WARNING
              log_format: json
            gateway:
              timeout: 120.0
        """))
        config = load_config(yaml_file)
        assert config.telemetry.log_level == "WARNING"
        assert config.telemetry.log_format == "json"
        assert config.gateway.timeout == 120.0

    def test_env_overrides_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("telemetry:\n  log_level: WARNING\n")
        config = load_config(
            yaml_file,
            environ={"PRAXIS_TELEMETRY__LOG_LEVEL": "ERROR"},
        )
        assert config.telemetry.log_level == "ERROR"

    def test_overrides_highest_priority(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("gateway:\n  timeout: 120.0\n")
        config = load_config(yaml_file, gateway={"timeout": 30.0})
        assert config.gateway.timeout == 30.0

    def test_missing_yaml_raises(self) -> None:
        with pytest.raises(ConfigError, match="配置文件不存在"):
            load_config("/nonexistent/path.yaml")


class TestComponentIsolation:
    """Task 2.3: 组件配置隔离验证。"""

    def test_get_component_config(self) -> None:
        config = load_config()
        gw = get_component_config(config, "gateway")
        assert gw.timeout == 60.0

    def test_unknown_component_raises(self) -> None:
        config = load_config()
        with pytest.raises(ConfigError, match="未知的组件配置"):
            get_component_config(config, "nonexistent")

    def test_loads_are_independent(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("telemetry:\n  log_level: DEBUG\n")
        first = load_config(yaml_file)
        second = load_config(environ={})
        assert first.telemetry.log_level == "DEBUG"
        assert second.telemetry.log_level == "INFO"

    def test_api_key_is_never_part_of_config(self) -> None:
        config = load_config(
            environ={"PRAXIS_MODEL_API_KEY": "secret-value-that-must-not-leak"},
        )
        serialized = config.model_dump_json()
        assert "secret-value-that-must-not-leak" not in serialized
        assert "api_key" not in config.gateway.deployments[0].model_fields_set


class TestValidation:
    """配置内容由 Pydantic 在加载边界严格验证。"""

    def test_invalid_config_rejected(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("persistence:\n  backend: invalid_backend\n")
        with pytest.raises(ValidationError):
            load_config(yaml_file)
