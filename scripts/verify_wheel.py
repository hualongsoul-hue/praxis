"""Install the built wheel into a clean uv environment and run package smoke checks."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    wheels = sorted((ROOT / "dist").glob("praxis-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one Praxis wheel, found {len(wheels)}")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable not found")

    with tempfile.TemporaryDirectory(prefix="praxis-wheel-") as directory:
        environment = Path(directory) / "venv"
        run(uv, "venv", str(environment), "--python", sys.executable)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        run(uv, "pip", "install", "--python", str(python), str(wheels[0]))
        run(str(python), "-m", "praxis", "version")
        run(str(python), "-m", "praxis", "config", "validate", "config.example.yaml")
        run(
            str(python),
            "-c",
            (
                "from importlib.resources import files; "
                "import praxis; "
                "assert praxis.__version__ == '1.0.0'; "
                "assert files('praxis').joinpath('py.typed').is_file(); "
                "assert files('praxis.skills.builtins').joinpath("
                "'task-planning', 'SKILL.md').is_file(); "
                "from praxis.verification import VerifierRegistry; "
                "assert VerifierRegistry is not None"
            ),
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
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
