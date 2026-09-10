"""Exercise the same public SDK workflow used against an isolated wheel install."""

import sys

import pytest

from praxis.config import PraxisConfig
from praxis.config.schemas import PersistenceConfig
from scripts.verify_wheel import run
from scripts.wheel_runtime_smoke import smoke


async def test_packaged_runtime_smoke_uses_offline_public_contract(tmp_path) -> None:
    result = await smoke(PraxisConfig(persistence=PersistenceConfig(
        backend="filesystem", filesystem_path=str(tmp_path / "store"),
    )))
    assert result == ("offline response", "offline response", True)


def test_wheel_output_decoding_failure_cannot_silently_pass(tmp_path) -> None:
    with pytest.raises(UnicodeDecodeError):
        run(
            sys.executable, "-I", "-c", "import sys; sys.stdout.buffer.write(bytes([255]))",
            working_directory=tmp_path,
        )
