"""命令行入口测试。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxis.__main__ import main


def test_version(capsys) -> None:
    rc = main(["version"])
    assert rc == 0
    assert "praxis" in capsys.readouterr().out


def test_validate_defaults_ok() -> None:
    assert main(["config", "validate"]) == 0


def test_validate_bad_config_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("gateway:\n  timeout: not-a-number\n", encoding="utf-8")
    assert main(["config", "validate", str(bad)]) == 1


def test_show_config_outputs_json(capsys) -> None:
    rc = main(["config", "show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"telemetry"' in out


def test_show_config_never_prints_api_key(monkeypatch, capsys) -> None:
    monkeypatch.setenv("PRAXIS_MODEL_API_KEY", "unit-secret-value")
    assert main(["config", "show"]) == 0
    assert "unit-secret-value" not in capsys.readouterr().out


def test_doctor_reports_missing_key_without_leaking_value(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("PRAXIS_MODEL_API_KEY", raising=False)
    config = tmp_path / "doctor.yaml"
    config.write_text(
        f"persistence:\n  backend: filesystem\n  filesystem_path: '{tmp_path / 'store'}'\n",
        encoding="utf-8",
    )
    assert main(["doctor", str(config)]) == 1
    assert "PRAXIS_MODEL_API_KEY" in capsys.readouterr().out


def test_show_and_doctor_report_configuration_load_failures(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.yaml"
    assert main(["config", "show", str(missing)]) == 1
    assert "配置加载失败" in capsys.readouterr().err
    assert main(["doctor", str(missing)]) == 1
    assert "config" in capsys.readouterr().out


def test_doctor_success_with_remote_embedding(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("PRAXIS_MODEL_API_KEY", "unit-test-key")
    config = tmp_path / "doctor.yaml"
    config.write_text(
        (
            "persistence:\n"
            "  backend: filesystem\n"
            f"  filesystem_path: '{tmp_path / 'store'}'\n"
            "memory:\n"
            "  embedding_api_base: 'https://embedding.example.com'\n"
        ),
        encoding="utf-8",
    )
    assert main(["doctor", str(config)]) == 0
    output = capsys.readouterr().out
    assert "model_credentials" in output
    assert "远程 Provider" in output
    assert "unit-test-key" not in output


class FakeChatSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def run(self, message: str) -> SimpleNamespace:
        return SimpleNamespace(content=f"echo:{message}")


class FakeChatRuntime:
    def __init__(self, gateway_config: object) -> None:
        self.chat_session = FakeChatSession()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def session(self) -> FakeChatSession:
        return self.chat_session


def test_chat_handles_messages_empty_input_and_exit(capsys) -> None:
    inputs = iter(["", "hello", "quit"])
    with (
        patch("praxis.__main__.PraxisRuntime", FakeChatRuntime),
        patch("builtins.input", side_effect=lambda _prompt: next(inputs)),
    ):
        assert main(["chat"]) == 0
    assert "echo:hello" in capsys.readouterr().out


def test_chat_reports_runtime_failure(capsys) -> None:
    with patch("praxis.__main__.chat_command", side_effect=RuntimeError("startup")):
        assert main(["chat"]) == 1
    assert "RuntimeError" in capsys.readouterr().err
