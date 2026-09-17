"""Native output caps: one effective value for transport and budget reservation."""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from praxis.config.schemas import GatewayConfig, ModelDeployment
from praxis.exceptions import GatewayError
from praxis.gateway.metering import get_token_count
from praxis.gateway.router import GatewayRouter
from praxis.session.core import Session
from tests import test_output_limits as support
from tests.test_gateway import make_raw_response, make_raw_stream_chunk

output_store = support.output_store
output_session = support.output_session
written_values = support.written_values

MESSAGES = [{"role": "user", "content": "A synthetic request"}]


@pytest.fixture
def gateway() -> GatewayRouter:
    return GatewayRouter(GatewayConfig(
        deployments=[ModelDeployment(
            model_name=name, model="openai/gpt-4o", default_max_output_tokens=limit,
            input_cost_per_token=0.001, output_cost_per_token=0.002,
        ) for name, limit in [("default", 13), ("alternate", 29)]],
        max_total_tokens=100_000, max_budget=1000, max_concurrent_requests=1,
    ), environ={"PRAXIS_MODEL_API_KEY": "unit-test-model-key"})


async def consume(gateway: GatewayRouter, streaming: bool, **kwargs: Any) -> None:
    if streaming:
        async for chunk in gateway.stream(MESSAGES, **kwargs):  # noqa: B007
            pass
    else:
        await gateway.complete(MESSAGES, **kwargs)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("model,default", [("default", 13), ("alternate", 29)])
@pytest.mark.parametrize("options,field,override", [
    ({}, "max_tokens", None),
    ({"max_tokens": 7}, "max_tokens", 7),
    ({"max_tokens": 47}, "max_tokens", 47),
    ({"max_completion_tokens": 19}, "max_completion_tokens", 19),
    ({"max_tokens": None}, "max_tokens", None),
    ({"max_completion_tokens": None}, "max_tokens", None),
    ({"max_tokens": None, "max_completion_tokens": None}, "max_tokens", None),
    ({"max_tokens": None, "max_completion_tokens": 19}, "max_completion_tokens", 19),
    ({"max_tokens": 7, "max_completion_tokens": None}, "max_tokens", 7),
])
async def test_effective_cap_reaches_transport_and_reservation(
    gateway: GatewayRouter, streaming: bool, model: str, default: int,
    options: dict[str, Any], field: str, override: int | None,
) -> None:
    effective = default if override is None else override
    prompt_tokens = get_token_count(MESSAGES, "openai/gpt-4o")
    observed: list[dict[str, Any]] = []

    async def raw_stream() -> AsyncIterator[SimpleNamespace]:
        chunk = make_raw_stream_chunk(content="not cropped", finish_reason="stop")
        chunk.usage = make_raw_response(prompt_tokens=3, completion_tokens=2).usage
        yield chunk

    async def completion(**kwargs: Any) -> Any:
        observed.append(kwargs)
        assert gateway.reserved_tokens == prompt_tokens + effective
        assert gateway.reserved_spend_usd == pytest.approx(
            prompt_tokens * 0.001 + effective * 0.002,
        )
        return raw_stream() if streaming else make_raw_response(
            "not cropped", prompt_tokens=3, completion_tokens=2,
        )

    gateway.router.acompletion = AsyncMock(side_effect=completion)
    await consume(gateway, streaming, model=model, temperature=0.2, **options)
    assert len(observed) == 1
    assert observed[0]["model"] == model
    assert observed[0]["temperature"] == 0.2
    assert {key: observed[0][key] for key in ("max_tokens", "max_completion_tokens")
            if key in observed[0]} == {field: effective}
    assert gateway.reservations == {}
    assert gateway.reserved_tokens == 0
    assert gateway.reserved_spend_usd == 0
    assert gateway.total_tokens == 5
    assert gateway.total_spend == pytest.approx(0.007)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("options", [
    {field: value} for field in ("max_tokens", "max_completion_tokens")
    for value in (0, -1, True, False, 3.5, "7")
] + [{"max_tokens": 7, "max_completion_tokens": 9},
     {"max_tokens": 7, "max_completion_tokens": 7}])
async def test_invalid_or_ambiguous_caps_fail_before_transport_and_reservation(
    gateway: GatewayRouter, streaming: bool, options: dict[str, Any],
) -> None:
    transport = AsyncMock(side_effect=AssertionError("must not dispatch"))
    gateway.router.acompletion = transport
    with pytest.raises(ValueError, match="max_.*tokens"):
        await consume(gateway, streaming, **options)
    transport.assert_not_awaited()
    assert gateway.reservations == {}
    assert gateway.next_reservation_id == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_default_capped_provider_failure_stays_observable_and_releases_budget(
    gateway: GatewayRouter, streaming: bool,
) -> None:
    transport = AsyncMock(side_effect=RuntimeError("synthetic provider failure"))
    gateway.router.acompletion = transport
    with pytest.raises(GatewayError, match="synthetic provider failure"):
        await consume(gateway, streaming)
    assert transport.call_args.kwargs["max_tokens"] == 13
    assert gateway.reservations == {}
    assert gateway.reserved_tokens == 0
    assert gateway.total_tokens == 0


@pytest.mark.parametrize("streaming", [False, True])
async def test_default_capped_cancel_settles_unknown_usage_and_releases_slot(
    gateway: GatewayRouter, streaming: bool,
) -> None:
    started = asyncio.Event()

    async def completion(**kwargs: Any) -> Any:
        started.set()
        await asyncio.Event().wait()

    transport = AsyncMock(side_effect=completion)
    gateway.router.acompletion = transport
    task = asyncio.create_task(consume(gateway, streaming))
    async with asyncio.timeout(3):
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert gateway.reservations == {}
    assert gateway.total_tokens == get_token_count(MESSAGES, "openai/gpt-4o") + 13
    assert transport.call_args.kwargs["max_tokens"] == 13
    async with asyncio.timeout(0.2):
        async with gateway.request_slot():
            pass


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("finish,refusal,reason", [
    ("length", None, "token_exhausted"),
    ("stop", "Provider refuses", "safety_refusal"),
    ("stop", None, "natural"),
])
async def test_native_session_propagates_default_without_cropping_or_repair(
    output_session: Session, gateway: GatewayRouter, written_values: list[str],
    streaming: bool, finish: str, refusal: str | None, reason: str,
) -> None:
    content = "Synthetic provider text longer than the configured numeric cap."
    observed: list[dict[str, Any]] = []

    async def raw_stream() -> AsyncIterator[SimpleNamespace]:
        yield make_raw_stream_chunk(content=content, refusal=refusal)
        yield make_raw_stream_chunk(content=None, finish_reason=finish)

    async def completion(**kwargs: Any) -> Any:
        observed.append(kwargs)
        raw = make_raw_response(content, refusal=refusal)
        raw.choices[0].finish_reason = finish
        return raw_stream() if streaming else raw

    gateway.router.acompletion = AsyncMock(side_effect=completion)
    events = await support.run_output(output_session, streaming)
    assert events[-1].data["reason"] == reason
    assert len(observed) == 1
    assert observed[0]["max_tokens"] == 13
    assert "max_completion_tokens" not in observed[0]
    if refusal is None:
        assert events[-1].data["content"] == content
    assert written_values == []
    assert gateway.reservations == {}


@pytest.mark.parametrize("streaming", [False, True])
async def test_cancel_before_dispatch_releases_without_charging(
    gateway: GatewayRouter, streaming: bool,
) -> None:
    transport = AsyncMock(side_effect=AssertionError("must not dispatch"))
    gateway.router.acompletion = transport
    entering = asyncio.Event()

    async def queued_call() -> None:
        entering.set()
        await consume(gateway, streaming)

    async with gateway.request_slot():
        task = asyncio.create_task(queued_call())
        async with asyncio.timeout(3):
            await entering.wait()
            assert gateway.reservations
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    transport.assert_not_awaited()
    assert gateway.reservations == {}
    assert gateway.reserved_tokens == 0
    assert gateway.total_tokens == 0
    assert gateway.total_spend == 0


@pytest.mark.parametrize("streaming", [False, True])
async def test_default_is_sent_even_without_budget_constraints(streaming: bool) -> None:
    gateway = GatewayRouter(GatewayConfig(deployments=[ModelDeployment(
        model_name="default", model="openai/gpt-4o", default_max_output_tokens=32,
    )]), environ={"PRAXIS_MODEL_API_KEY": "unit-test-model-key"})

    async def raw_stream() -> AsyncIterator[SimpleNamespace]:
        yield make_raw_stream_chunk(content="unchanged", finish_reason="stop")

    transport = AsyncMock(return_value=raw_stream() if streaming else make_raw_response())
    gateway.router.acompletion = transport
    await consume(gateway, streaming)
    assert transport.call_args.kwargs["max_tokens"] == 32
    assert "max_completion_tokens" not in transport.call_args.kwargs
    assert gateway.reservations == {}
