"""Live evidence for the configured OpenAI-compatible multimodal endpoint."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import struct
import wave
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from praxis.config import GatewayConfig, ModelDeployment
from praxis.exceptions import GatewayError
from praxis.gateway.chat import chat
from praxis.gateway.router import GatewayRouter
from tests.integration.live_model_evidence import (
    CapabilityClassification,
    HealthWindowEvidence,
    ModalityObservation,
    classify_capability_windows,
    format_safe_evidence,
    response_has_model_output,
)

MODEL = "openai/glm-5.1-openai"
API_BASE = "http://172.24.23.192:3000/v1"
MP4_SOURCE = (
    "ffmpeg -v error -f lavfi -i color=c=black:s=2x2:d=0.04 -an -c:v libx264 "
    "-pix_fmt yuv420p -movflags frag_keyframe+empty_moov -f mp4 pipe:1"
)
MP4_SHA256 = "16c91607b6f2760bd470bcdf149ff0845c9d0b272862d37cb46a29cc838288ec"
REQUIRED_HEALTH_WINDOWS = 3
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


def modality_cases() -> list[tuple[str, dict[str, Any], CapabilityClassification]]:
    """Return minimal provider blocks and the endpoint contract under test."""

    return [
        (
            "image",
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{PNG_BASE64}"},
            },
            CapabilityClassification.UNSUPPORTED,
        ),
        (
            "audio",
            {
                "type": "input_audio",
                "input_audio": {"data": build_wav_base64(), "format": "wav"},
            },
            CapabilityClassification.UNSUPPORTED,
        ),
        (
            "video",
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{MP4_BASE64}"},
            },
            CapabilityClassification.UNSUPPORTED,
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
            CapabilityClassification.UNSUPPORTED,
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

    return tuple(box.name for box in parse_mp4_box_records(data))


@dataclass(frozen=True, slots=True)
class Mp4Box:
    """One structurally bounded ISO BMFF box."""

    name: bytes
    payload: bytes


@dataclass(frozen=True, slots=True)
class Mp4VideoTrack:
    """Minimal codec and fragment facts required by the live fixture."""

    codec: bytes
    width: int
    height: int
    handler_type: bytes
    has_avc_configuration: bool
    fragment_sample_count: int
    media_data_bytes: int


def parse_mp4_box_records(data: bytes) -> tuple[Mp4Box, ...]:
    """Parse one ISO BMFF box sequence while validating every bound."""

    offset = 0
    boxes: list[Mp4Box] = []
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
        boxes.append(
            Mp4Box(
                name=data[offset + 4 : offset + 8],
                payload=data[offset + header_size : offset + size],
            )
        )
        offset += size
    return tuple(boxes)


def require_mp4_box(boxes: tuple[Mp4Box, ...], name: bytes) -> Mp4Box:
    """Return exactly one named box or reject an ambiguous fixture."""

    matches = [box for box in boxes if box.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r} MP4 box")
    return matches[0]


def inspect_mp4_video_track(data: bytes) -> Mp4VideoTrack:
    """Validate the H.264 video track and one-sample fragment structure."""

    top_level = parse_mp4_box_records(data)
    moov = require_mp4_box(top_level, b"moov")
    trak = require_mp4_box(parse_mp4_box_records(moov.payload), b"trak")
    mdia = require_mp4_box(parse_mp4_box_records(trak.payload), b"mdia")
    media_boxes = parse_mp4_box_records(mdia.payload)
    handler = require_mp4_box(media_boxes, b"hdlr")
    if len(handler.payload) < 12:
        raise ValueError("truncated MP4 handler")

    minf = require_mp4_box(media_boxes, b"minf")
    stbl = require_mp4_box(parse_mp4_box_records(minf.payload), b"stbl")
    stsd = require_mp4_box(parse_mp4_box_records(stbl.payload), b"stsd")
    if len(stsd.payload) < 8:
        raise ValueError("truncated MP4 sample description")
    entry_count = int.from_bytes(stsd.payload[4:8], "big")
    sample_entries = parse_mp4_box_records(stsd.payload[8:])
    if entry_count != 1 or len(sample_entries) != 1:
        raise ValueError("fixture must contain one MP4 sample entry")
    sample_entry = sample_entries[0]
    if sample_entry.name != b"avc1" or len(sample_entry.payload) < 78:
        raise ValueError("fixture must contain an AVC visual sample entry")
    width = int.from_bytes(sample_entry.payload[24:26], "big")
    height = int.from_bytes(sample_entry.payload[26:28], "big")
    codec_boxes = parse_mp4_box_records(sample_entry.payload[78:])
    has_avc_configuration = any(box.name == b"avcC" for box in codec_boxes)

    moof = require_mp4_box(top_level, b"moof")
    traf = require_mp4_box(parse_mp4_box_records(moof.payload), b"traf")
    trun = require_mp4_box(parse_mp4_box_records(traf.payload), b"trun")
    if len(trun.payload) < 8:
        raise ValueError("truncated MP4 track run")
    fragment_sample_count = int.from_bytes(trun.payload[4:8], "big")
    media_data = require_mp4_box(top_level, b"mdat")
    return Mp4VideoTrack(
        codec=sample_entry.name,
        width=width,
        height=height,
        handler_type=handler.payload[8:12],
        has_avc_configuration=has_avc_configuration,
        fragment_sample_count=fragment_sample_count,
        media_data_bytes=len(media_data.payload),
    )


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
    assert hashlib.sha256(mp4).hexdigest() == MP4_SHA256
    assert "color=c=black:s=2x2:d=0.04" in MP4_SOURCE
    video_track = inspect_mp4_video_track(mp4)
    assert video_track.codec == b"avc1"
    assert (video_track.width, video_track.height) == (2, 2)
    assert video_track.handler_type == b"vide"
    assert video_track.has_avc_configuration is True
    assert video_track.fragment_sample_count == 1
    assert video_track.media_data_bytes > 0
    with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getnframes() == 800


async def probe_text_control(
    gateway: GatewayRouter,
) -> tuple[bool, str | None]:
    """Return only text health and a safe mapped category."""

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
        return False, error_category
    if response is None or not response_has_model_output(response):
        return False, "EmptyModelResponse"
    return True, None


async def probe_modality(
    gateway: GatewayRouter,
    messages: list[dict[str, Any]],
) -> ModalityObservation:
    """Probe one modality while retaining no request or response payload."""

    response = None
    observed_error: GatewayError | None = None
    try:
        response = await chat(
            gateway,
            messages,
            timeout=30.0,
            max_tokens=64,
        )
    except GatewayError as exc:
        observed_error = exc
    if observed_error is not None:
        original_type = observed_error.details.get("original_type")
        return ModalityObservation.failed(
            type(observed_error).__name__,
            original_type if isinstance(original_type, str) else "UnknownProviderError",
        )
    if response is None or not response_has_model_output(response):
        return ModalityObservation.failed("EmptyModelResponse", "EmptyModelResponse")
    return ModalityObservation.succeeded()


async def probe_health_window(
    gateway: GatewayRouter,
    messages: list[dict[str, Any]],
) -> HealthWindowEvidence:
    """Run one independent before/modality/after endpoint window."""

    before_healthy, before_category = await probe_text_control(gateway)
    modality = await probe_modality(gateway, messages) if before_healthy else None
    after_healthy, after_category = await probe_text_control(gateway)
    control_categories = tuple(
        category
        for category in (before_category, after_category)
        if category is not None
    )
    return HealthWindowEvidence(
        before_healthy=before_healthy,
        modality=modality,
        after_healthy=after_healthy,
        control_category=(
            ",".join(dict.fromkeys(control_categories))
            if control_categories
            else None
        ),
    )


@pytest.mark.live_model
@pytest.mark.parametrize(
    ("modality", "content_block", "expected_classification"),
    modality_cases(),
)
async def test_live_multimodal_capability(
    multimodal_gateway: GatewayRouter,
    modality: str,
    content_block: dict[str, Any],
    expected_classification: CapabilityClassification,
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

    windows: list[HealthWindowEvidence] = []
    while len(windows) < REQUIRED_HEALTH_WINDOWS:
        windows.append(await probe_health_window(multimodal_gateway, messages))
    evidence = classify_capability_windows(windows)
    safe_projection = format_safe_evidence(modality, evidence)
    if expected_classification is CapabilityClassification.UNCLASSIFIED:
        pytest.fail(f"endpoint capability has no published oracle; {safe_projection}")
    assert evidence.classification is expected_classification, safe_projection
