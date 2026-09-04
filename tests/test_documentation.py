"""Keep published Markdown and configuration examples aligned with the SDK."""

import ast
import re
from pathlib import Path

import yaml

from examples import console, streaming_console
from praxis import load_config
from praxis.config.settings import PraxisConfig

ROOT = Path(__file__).parents[1]
MARKDOWN_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
FENCED_EXAMPLE = re.compile(r"```(?P<language>python|yaml)\n(?P<body>.*?)```", re.DOTALL)


def test_markdown_python_and_yaml_examples_are_valid() -> None:
    found = 0
    for path in MARKDOWN_FILES:
        text = path.read_text(encoding="utf-8")
        for match in FENCED_EXAMPLE.finditer(text):
            found += 1
            body = match.group("body")
            if match.group("language") == "python":
                compile(
                    body,
                    f"{path}:python-example",
                    "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
            else:
                assert yaml.safe_load(body) is not None
    assert found >= 8


def test_example_config_uses_the_production_model_boundary() -> None:
    config = load_config(ROOT / "config.example.yaml", environ={})
    deployment = config.gateway.deployments[0]
    assert deployment.model == "openai/glm-5.1-openai"
    assert deployment.api_base == "http://172.24.23.192:3000/v1"
    assert deployment.api_key_env == "PRAXIS_MODEL_API_KEY"


def assert_complete_mapping(raw: object, complete: object, path: str = "") -> None:
    """Assert that the example explicitly documents every configured model field."""

    if isinstance(complete, dict):
        assert isinstance(raw, dict), path
        if not complete:
            return
        assert set(raw) == set(complete), path
        for key, value in complete.items():
            assert_complete_mapping(raw[key], value, f"{path}.{key}".lstrip("."))
    elif isinstance(complete, list) and complete:
        assert isinstance(raw, list) and raw, path
        assert_complete_mapping(raw[0], complete[0], f"{path}[0]")


def test_example_config_explicitly_lists_every_schema_field() -> None:
    raw = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
    complete = PraxisConfig().model_dump(mode="json")
    complete["mcp"]["servers"] = [
        load_config(ROOT / "config.example.yaml", environ={})
        .mcp.servers[0]
        .model_dump(mode="json")
    ]

    assert_complete_mapping(raw, complete)


def test_interactive_console_config_matches_current_template() -> None:
    template = yaml.safe_load(
        (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    )
    console_config = yaml.safe_load(
        (ROOT / "examples" / "config.yaml").read_text(encoding="utf-8")
    )

    assert console_config == template
    assert load_config(ROOT / "examples" / "config.yaml", environ={})


def test_documentation_links_reference_current_files() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+]\(([^)]+\.md)\)", readme):
        assert (ROOT / target).is_file(), target


def test_sdk_examples_compile_against_public_runtime_api() -> None:
    examples = sorted((ROOT / "examples").glob("*.py"))
    assert examples
    for path in examples:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_repository_examples_execute_current_configuration_boundary() -> None:
    expected = (ROOT / "examples" / "config.yaml").resolve()

    assert console.build_parser().parse_args([]).config == expected
    assert streaming_console.build_parser().parse_args([]).config == expected
    assert load_config(expected, environ={}).gateway.default_model == "default"


def test_documentation_describes_final_health_event_and_transport_contracts() -> None:
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in MARKDOWN_FILES)

    assert "health_probe_ttl" in corpus
    assert "schema_version" in corpus
    assert "HTTPS/mTLS" in corpus
    assert "YAML 文件所在目录" in corpus


def test_ci_has_an_explicit_credentialed_live_release_gate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "runs-on: [self-hosted, linux]" in workflow
    assert "test_gateway_live.py" in workflow
    assert "test_gateway_live_multimodal.py" in workflow
    assert "PRAXIS_MODEL_API_KEY: ${{secrets.PRAXIS_MODEL_API_KEY}}" in workflow
