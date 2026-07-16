"""Live evidence for the configured OpenAI-compatible multimodal endpoint."""

from __future__ import annotations

import base64
import io
import os
import struct
import wave
import zlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from praxis.config import GatewayConfig, ModelDeployment
from praxis.exceptions import GatewayError
from praxis.gateway.chat import chat
from praxis.gateway.router import GatewayRouter

MODEL = "openai/glm-5.1-openai"
API_BASE = "http://172.24.23.192:3000/v1"
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8A"
    "AAAASUVORK5CYII="
)
MP4_BASE64 = (
    "AAAAJGZ0eXBpc29tAAACAGlzb21pc282aXNvMmF2YzFtcDQxAAAC721vb3YAAABsbXZoZAAAAAAA"
    "AAAAAAAAAAAAA+gAAAAAAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAA"
    "AAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAHxdHJhawAAAFx0a2hkAAAAAwAAAAAA"
    "AAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAA"
    "AAAAQAAAAAACAAAAAgAAAAABjW1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAMgAAAAAAVcQAAAAAAC1o"
    "ZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAAThtaW5mAAAAFHZtaGQAAAAB"
    "AAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAD4c3RibAAAAKxzdHNk"
    "AAAAAAAAAAEAAACcYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAACAAIASAAAAEgAAAAAAAAAARVM"
    "YXZjNjIuMjkuMTAwIGxpYngyNjQAAAAAAAAAAAAAABj//wAAADZhdmNDAWQACv/hABlnZAAKrNlfiIjA"
    "RAAAAwAEAAADAMg8SJZYAQAGaOvjyyLA/fj4AAAAABBwYXNwAAAAAQAAAAEAAAAQc3R0cwAAAAAAAAAA"
    "AAAAEHN0c2MAAAAAAAAAAAAAABRzdHN6AAAAAAAAAAAAAAAAAAAAEHN0Y28AAAAAAAAAAAAAAChtdmV4"
    "AAAAIHRyZXgAAAAAAAAAAQAAAAEAAAAAAAAAAAAAAAAAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxy"
    "AAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0b28AAAAdZGF0YQAAAAEAAAAATGF2"
    "ZjYyLjEzLjEwMAAAAHBtb29mAAAAEG1maGQAAAAAAAAAAQAAAFh0cmFmAAAAJHRmaGQAAAA5AAAAAQAA"
    "AAAAAAMTAAACAAAAArcBAQAAAAAAFHRmZHQBAAAAAAAAAAAAAAAAAAAYdHJ1bgAAAAUAAAABAAAAeAIA"
    "AAAAAAK/bWRhdAAAAqAGBf//nNxF6b3m2Ui3lizYINkj7u94MjY0IC0gY29yZSAxNjUgLSBILjI2NC9N"
    "UEVHLTQgQVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDI1IC0gaHR0cDovL3d3dy52aWRlb2xhbi5v"
    "cmcveDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9"
    "MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVmPTEg"
    "bWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6b25lPTIx"
    "LDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MSBsb29rYWhlYWRfdGhy"
    "ZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVyYXlf"
    "Y29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJhbWlkPTIgYl9hZGFwdD0x"
    "IGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdlaWdodHA9MiBrZXlpbnQ9MjUw"
    "IGtleWludF9taW49MjUgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCBy"
    "Yz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00"
    "IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAPZYiEACv//vZzfAprbbGBAAAAQ21mcmEAAAArdGZy"
    "YQEAAAAAAAABAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAMTAQEBAAAAEG1mcm8AAAAAAAAAQw=="
)


def build_wav_base64() -> str:
    """Build 100 ms of valid mono PCM silence without touching disk."""

    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(bytes(1_600))
    return base64.b64encode(output.getvalue()).decode("ascii")


def modality_cases() -> list[tuple[str, dict[str, Any], type[GatewayError] | None]]:
    """Return minimal provider blocks and the endpoint contract under test."""

    return [
        (
            "image",
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{PNG_BASE64}"},
            },
            GatewayError,
        ),
        (
            "audio",
            {
                "type": "input_audio",
                "input_audio": {"data": build_wav_base64(), "format": "wav"},
            },
            GatewayError,
        ),
        (
            "video",
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{MP4_BASE64}"},
            },
            GatewayError,
        ),
        (
            "file",
            {
                "type": "file",
                "file": {
                    "filename": "evidence.txt",
                    "file_data": "data:text/plain;base64,"
                    + base64.b64encode("Praxis 端点能力证据".encode()).decode("ascii"),
                },
            },
            GatewayError,
        ),
    ]


def parse_png_size(data: bytes) -> tuple[int, int, tuple[bytes, ...]]:
    """Validate PNG chunk bounds and CRCs, returning IHDR size and chunk names."""

    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("invalid PNG signature")
    offset = 8
    width = 0
    height = 0
    names: list[bytes] = []
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length = int.from_bytes(data[offset : offset + 4], "big")
        name = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("PNG chunk exceeds payload")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(data[offset + 8 + length : end], "big")
        if zlib.crc32(name + payload) != expected_crc:
            raise ValueError("invalid PNG chunk CRC")
        names.append(name)
        if name == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        offset = end
        if name == b"IEND":
            break
    if offset != len(data):
        raise ValueError("trailing PNG bytes")
    return width, height, tuple(names)


def parse_mp4_boxes(data: bytes) -> tuple[bytes, ...]:
    """Validate top-level ISO BMFF box bounds without decoding the video."""

    offset = 0
    names: list[bytes] = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated MP4 box")
        size = int.from_bytes(data[offset : offset + 4], "big")
        header_size = 8
        if size == 1:
            if offset + 16 > len(data):
                raise ValueError("truncated extended MP4 box")
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header_size = 16
        elif size == 0:
            size = len(data) - offset
        if size < header_size or offset + size > len(data):
            raise ValueError("MP4 box exceeds payload")
        names.append(data[offset + 4 : offset + 8])
        offset += size
    return tuple(names)


@pytest.fixture
async def multimodal_gateway() -> AsyncIterator[GatewayRouter]:
    api_key = os.environ.get("PRAXIS_MODEL_API_KEY")
    if not api_key:
        pytest.skip("PRAXIS_MODEL_API_KEY is required for live endpoint evidence")
    gateway = GatewayRouter(
        GatewayConfig(
            deployments=[
                ModelDeployment(
                    model_name="default",
                    model=MODEL,
                    api_base=API_BASE,
                    api_key_env="PRAXIS_MODEL_API_KEY",
                    default_max_output_tokens=128,
                ),
            ],
            default_model="default",
            timeout=30.0,
            num_retries=0,
            max_concurrent_requests=1,
            max_total_tokens=20_000,
        ),
        environ={"PRAXIS_MODEL_API_KEY": api_key},
    )
    try:
        yield gateway
    finally:
        await gateway.close()


def test_live_samples_are_valid_and_minimal() -> None:
    png = base64.b64decode(PNG_BASE64, validate=True)
    mp4 = base64.b64decode(MP4_BASE64, validate=True)
    wav_bytes = base64.b64decode(build_wav_base64(), validate=True)

    width, height, png_chunks = parse_png_size(png)
    assert (width, height) == (1, 1)
    assert png_chunks == (b"IHDR", b"IDAT", b"IEND")
    assert parse_mp4_boxes(mp4) == (b"ftyp", b"moov", b"moof", b"mdat", b"mfra")
    with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getnframes() == 800


async def assert_healthy_text_control(
    gateway: GatewayRouter,
    stage: str,
) -> None:
    """Require a healthy text request before classifying a modality result."""

    response = None
    error_category: str | None = None
    try:
        response = await chat(
            gateway,
            [{"role": "user", "content": "Reply with OK."}],
            timeout=30.0,
            max_tokens=16,
        )
    except GatewayError as exc:
        error_category = type(exc).__name__
    if error_category is not None:
        pytest.fail(f"text control {stage} returned {error_category}")
    if response is None or not isinstance(response.content, str) or not response.content.strip():
        pytest.fail(f"text control {stage} returned an empty response")


@pytest.mark.live_model
@pytest.mark.parametrize(("modality", "content_block", "expected_error"), modality_cases())
async def test_live_multimodal_capability(
    multimodal_gateway: GatewayRouter,
    modality: str,
    content_block: dict[str, Any],
    expected_error: type[GatewayError] | None,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Acknowledge the attached {modality} briefly."},
                content_block,
            ],
        },
    ]

    await assert_healthy_text_control(multimodal_gateway, "before")
    response = None
    observed_error: GatewayError | None = None
    try:
        response = await chat(
            multimodal_gateway,
            messages,
            timeout=30.0,
            max_tokens=64,
        )
    except GatewayError as exc:
        observed_error = exc
    await assert_healthy_text_control(multimodal_gateway, "after")

    if observed_error is not None:
        if expected_error is None:
            pytest.fail(f"{modality} returned {type(observed_error).__name__}")
        assert type(observed_error) is expected_error
        assert observed_error.details.get("original_type") == "BadRequestError"
        return

    if expected_error is not None:
        pytest.fail(f"{modality} unexpectedly succeeded")
    if response is None or not isinstance(response.content, str) or not response.content.strip():
        pytest.fail(f"{modality} returned an empty response")
