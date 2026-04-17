"""S1 配置系统验证测试。"""

import os
import textwrap
from pathlib import Path

import pytest

from praxis.config import (
    PraxisConfig,
    get_component_config,
    load_config,
    on_config_change,
    register_validator,
    reload_config,
)
from praxis.exceptions import ConfigError


class TestPraxisConfig:
    """Task 2.1: PraxisConfig 顶层配置模型验证。"""

    def test_default_instantiation(self) -> None:
        config = PraxisConfig()
        assert config.gateway.default_model == "default"
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
        os.environ["PRAXIS_TELEMETRY__LOG_LEVEL"] = "ERROR"
        try:
            config = load_config(yaml_file)
            assert config.telemetry.log_level == "ERROR"
        finally:
            del os.environ["PRAXIS_TELEMETRY__LOG_LEVEL"]

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
        load_config()
        gw = get_component_config("gateway")
        assert gw.timeout == 60.0

    def test_unknown_component_raises(self) -> None:
        load_config()
        with pytest.raises(ConfigError, match="未知的组件配置"):
            get_component_config("nonexistent")

    def test_config_not_loaded_raises(self) -> None:
        import praxis.config.loader as loader_mod
        saved = loader_mod.current_config
        loader_mod.current_config = None
        try:
            with pytest.raises(ConfigError, match="配置未加载"):
                get_component_config("gateway")
        finally:
            loader_mod.current_config = saved


class TestValidationAndReload:
    """Task 2.4: 配置验证与热更新验证。"""

    def test_custom_validator(self) -> None:
        def require_model_list(config: PraxisConfig) -> None:
            if not config.gateway.model_list:
                raise ConfigError("gateway.model_list 不能为空")

        register_validator(require_model_list)
        try:
            with pytest.raises(ConfigError, match="model_list 不能为空"):
                load_config()
        finally:
            import praxis.config.validation as val_mod
            val_mod.validators.remove(require_model_list)

    def test_reload_detects_changes(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("telemetry:\n  log_level: INFO\n")

        load_config(yaml_file)

        changes_received: list[dict] = []
        on_config_change(lambda c: changes_received.append(c))

        yaml_file.write_text("telemetry:\n  log_level: DEBUG\n")
        reload_config()

        assert len(changes_received) == 1
        assert "telemetry.log_level" in changes_received[0]
        old_val, new_val = changes_received[0]["telemetry.log_level"]
        assert old_val == "INFO"
        assert new_val == "DEBUG"

        import praxis.config.validation as val_mod
        val_mod.change_listeners.clear()

    def test_invalid_config_rejected(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("persistence:\n  backend: invalid_backend\n")
        with pytest.raises(Exception):
            load_config(yaml_file)
