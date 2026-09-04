"""S1 配置系统验证测试。"""

import textwrap
from pathlib import Path

import pytest

from praxis.config import (
    PraxisConfig,
    get_component_config,
    load_config,
)
from praxis.config.schemas import (
    GatewayConfig,
    MCPConfig,
    MemoryConfig,
    ModelDeployment,
    PersistenceConfig,
    TelemetryConfig,
    ToolsConfig,
)
from praxis.exceptions import ConfigError, ModelValidationError
from praxis.models.mcp import MCPServerConfig, MCPTransportType


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
        with pytest.raises(ModelValidationError):
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
        with pytest.raises(ModelValidationError):
            config_type(**values)

    def test_log_level_must_be_known(self) -> None:
        with pytest.raises(ModelValidationError):
            TelemetryConfig(log_level="NOT_A_LEVEL")

    def test_root_config_rejects_unknown_sections(self) -> None:
        with pytest.raises(ModelValidationError):
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
        with pytest.raises(ModelValidationError):
            load_config(yaml_file)

    def test_enabled_mcp_requires_server_configuration(self) -> None:
        with pytest.raises(ModelValidationError):
            MCPConfig(enabled=True)

    def test_mcp_server_names_must_be_unique(self) -> None:
        server = MCPServerConfig(name="duplicate", command="python")
        with pytest.raises(ModelValidationError):
            MCPConfig(enabled=True, servers=[server, server])

    @pytest.mark.parametrize(
        "server",
        [
            MCPServerConfig.model_construct(name="stdio", transport=MCPTransportType.STDIO),
            MCPServerConfig.model_construct(name="http", transport=MCPTransportType.HTTP),
        ],
    )
    def test_mcp_transport_requires_its_endpoint(self, server: MCPServerConfig) -> None:
        with pytest.raises(ModelValidationError):
            MCPServerConfig(**server.model_dump())

    def test_nested_configuration_collections_are_immutable(self) -> None:
        config = PraxisConfig(
            telemetry=TelemetryConfig(log_levels={"gateway": "DEBUG"}),
            tools=ToolsConfig(fallback_mappings={"primary": "fallback"}),
            mcp=MCPConfig(
                enabled=True,
                servers=[MCPServerConfig(name="local", command="python")],
            ),
        )
        with pytest.raises(AttributeError):
            config.gateway.deployments.append(ModelDeployment())
        with pytest.raises(TypeError):
            config.telemetry.log_levels["gateway"] = "INFO"  # type: ignore[index]
        with pytest.raises(TypeError):
            config.tools.fallback_mappings["primary"] = "other"  # type: ignore[index]
        with pytest.raises(TypeError):
            config.mcp.servers[0].headers["Authorization"] = "value"  # type: ignore[index]
        assert "gateway" in config.model_dump_json()

    def test_duplicate_model_aliases_are_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            GatewayConfig(deployments=[ModelDeployment(), ModelDeployment()])

    def test_incomplete_backend_and_export_configs_are_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            PersistenceConfig(backend="redis")
        with pytest.raises(ModelValidationError):
            PersistenceConfig(backend="redis", redis_url="https://cache.example.com")
        with pytest.raises(ModelValidationError):
            TelemetryConfig(metrics_enabled=True, metrics_export="file", metrics_file=None)

    def test_dependent_network_and_embedding_fields_are_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            ToolsConfig(allow_private_networks=True)
        with pytest.raises(ModelValidationError):
            MemoryConfig(embedding_model="orphaned-model")
        with pytest.raises(ModelValidationError):
            MemoryConfig(embedding_api_base="file:///tmp/embed")

    def test_paths_are_anchored_to_the_configuration_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_directory = tmp_path / "config"
        other_directory = tmp_path / "other"
        config_directory.mkdir()
        other_directory.mkdir()
        config_file = config_directory / "praxis.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                persistence:
                  sqlite_path: ./state/praxis.db
                tools:
                  allowed_paths: [./workspace]
                telemetry:
                  metrics_file: ./metrics/praxis.prom
            """),
            encoding="utf-8",
        )
        monkeypatch.chdir(other_directory)

        config = load_config(config_file)

        assert Path(config.persistence.sqlite_path) == (
            config_directory / "state" / "praxis.db"
        ).resolve()
        assert Path(config.tools.allowed_paths[0]) == (
            config_directory / "workspace"
        ).resolve()
        assert Path(config.telemetry.metrics_file or "") == (
            config_directory / "metrics" / "praxis.prom"
        ).resolve()
