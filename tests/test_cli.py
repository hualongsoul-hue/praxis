"""命令行入口测试。"""

from pathlib import Path

from praxis.__main__ import main


def test_version(capsys) -> None:
    rc = main(["version"])
    assert rc == 0
    assert "praxis" in capsys.readouterr().out


def test_validate_defaults_ok() -> None:
    assert main(["validate"]) == 0


def test_validate_bad_config_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("gateway:\n  timeout: not-a-number\n", encoding="utf-8")
    assert main(["validate", str(bad)]) == 1


def test_show_config_outputs_json(capsys) -> None:
    rc = main(["show-config"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"telemetry"' in out
