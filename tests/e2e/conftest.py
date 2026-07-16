"""Lifecycle fixtures for real-boundary end-to-end tests."""

from collections.abc import AsyncIterator

import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


@pytest.fixture(autouse=True)
async def drain_litellm_logging_worker() -> AsyncIterator[None]:
    """Drain LiteLLM's public logging queue before pytest closes the test loop."""
    try:
        yield
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await GLOBAL_LOGGING_WORKER.stop()
