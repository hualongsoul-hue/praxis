"""Install the built wheel into a clean uv environment and run package smoke checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run(
    *command: str,
    working_directory: Path,
    environment: dict[str, str] | None = None,
    expected_return_code: int = 0,
    timeout: float = 300,
) -> subprocess.CompletedProcess[str]:
    print(f"[wheel smoke] {' '.join(command)}", flush=True)
    try:
        result = subprocess.run(
            command,
            cwd=working_directory,
            env=environment,
            check=False,
            capture_output=True,
            encoding="utf-8",
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"command {command!r} exceeded {timeout:.0f} seconds"
        ) from error
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != expected_return_code:
        raise RuntimeError(
            f"command {command!r} returned {result.returncode}, "
            f"expected {expected_return_code}"
        )
    return result


def clean_environment() -> dict[str, str]:
    allowed_names = (
        "APPDATA",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "PATH",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    )
    environment = {
        name: os.environ[name]
        for name in allowed_names
        if name in os.environ
    }
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    return environment


def main() -> int:
    wheels = sorted((ROOT / "dist").glob("praxis-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one Praxis wheel, found {len(wheels)}")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable not found")

    with tempfile.TemporaryDirectory(prefix="praxis-wheel-", dir=ROOT.parent) as directory:
        working_directory = Path(directory).resolve()
        if working_directory.is_relative_to(ROOT):
            raise RuntimeError("wheel smoke environment must be outside the repository")
        environment = Path(directory) / "venv"
        run(
            uv,
            "venv",
            str(environment),
            "--python",
            sys.executable,
            working_directory=working_directory,
        )
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        cli = environment / ("Scripts/praxis.exe" if sys.platform == "win32" else "bin/praxis")
        run(
            uv,
            "pip",
            "install",
            "--link-mode",
            "copy",
            "--python",
            str(python),
            str(wheels[0]),
            working_directory=working_directory,
        )
        smoke_config = working_directory / "wheel-smoke.yaml"
        smoke_config.write_text(
            "persistence:\n"
            "  backend: filesystem\n"
            "  filesystem_path: ./store\n",
            encoding="utf-8",
        )
        isolated_environment = clean_environment()
        version = run(
            str(cli),
            "version",
            working_directory=working_directory,
            environment=isolated_environment,
        )
        if version.stdout.strip() != "praxis 1.0.0":
            raise RuntimeError(f"unexpected version output: {version.stdout!r}")
        run(
            str(cli),
            "config",
            "validate",
            str(smoke_config),
            working_directory=working_directory,
            environment=isolated_environment,
        )
        doctor = run(
            str(cli),
            "doctor",
            str(smoke_config),
            working_directory=working_directory,
            environment=isolated_environment,
            expected_return_code=1,
        )
        expected_doctor_output = (
            "[READY] config",
            "[FAILED] model_credentials",
            "[READY] storage",
            "[DEGRADED] embedding",
        )
        if not all(marker in doctor.stdout for marker in expected_doctor_output):
            raise RuntimeError(f"unexpected doctor output: {doctor.stdout!r}")
        run(
            str(python),
            "-c",
            (
                "from importlib.resources import files; "
                "import praxis; "
                "from praxis import AudioInput, FileInput, ImageInput, UserInput, VideoInput; "
                "assert praxis.__version__ == '1.0.0'; "
                "assert UserInput(text='ok').text == 'ok'; "
                "assert all(value is not None for value in "
                "(AudioInput, FileInput, ImageInput, VideoInput)); "
                "assert files('praxis').joinpath('py.typed').is_file(); "
                "assert files('praxis.skills.builtins').joinpath("
                "'task-planning', 'SKILL.md').is_file(); "
                "from praxis.verification import VerifierRegistry; "
                "assert VerifierRegistry is not None"
            ),
            working_directory=working_directory,
            environment=isolated_environment,
        )
        run(
            str(python),
            "-c",
            (
                "import asyncio\n"
                "from pathlib import Path\n"
                "from typing import cast\n"
                "from praxis.config import PersistenceConfig\n"
                "from praxis.exceptions import PersistenceError\n"
                "from praxis.gateway.router import GatewayRouter\n"
                "from praxis.persistence import create_store\n"
                "from praxis.verification.visual import VisualVerifier\n"
                "async def check():\n"
                "    try:\n"
                "        await create_store(PersistenceConfig(backend='redis'))\n"
                "    except PersistenceError as exc:\n"
                "        assert 'praxis[redis]' in str(exc)\n"
                "    else:\n"
                "        raise AssertionError('Redis extra unexpectedly installed')\n"
                "    verifier = VisualVerifier(cast(GatewayRouter, object()))\n"
                "    try:\n"
                "        await verifier.capture_screenshot('about:blank', Path('unused.png'))\n"
                "    except RuntimeError as exc:\n"
                "        assert 'praxis[visual]' in str(exc)\n"
                "    else:\n"
                "        raise AssertionError('visual extra unexpectedly installed')\n"
                "asyncio.run(check())\n"
                "try:\n"
                "    import praxis.tools.mcp\n"
                "except RuntimeError as exc:\n"
                "    assert 'praxis[mcp]' in str(exc)\n"
                "else:\n"
                "    raise AssertionError('MCP extra unexpectedly installed')"
            ),
            working_directory=working_directory,
            environment=isolated_environment,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
