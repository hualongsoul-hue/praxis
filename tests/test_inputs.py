"""Typed multimodal input and secure resolution tests."""

import asyncio
import base64
from pathlib import Path

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from praxis import (
    AudioContent,
    AudioInput,
    FileContent,
    FileInput,
    ImageContent,
    ImageInput,
    InputMediaTypeError,
    InputNetworkError,
    InputPathError,
    InputResolver,
    InputSizeLimitError,
    InputSourceKind,
    InvalidInputSourceError,
    UnsupportedInputModalityError,
    UserInput,
    VideoContent,
    VideoInput,
)
from praxis.config import InputConfig, ModelCapabilities
from praxis.models import TextContent

SAFE_FILENAMES = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cc", "Cs"),
        blacklist_characters="/\\:",
    ),
    min_size=1,
    max_size=40,
).filter(lambda value: value.strip() not in {"", ".", ".."})


def test_user_input_requires_text_or_attachment() -> None:
    with pytest.raises(ValueError):
        UserInput()


def test_input_config_fails_closed() -> None:
    config = InputConfig()
    assert config.allowed_paths == []
    assert config.remote_enabled is False
    assert config.allow_private_networks is False
    assert config.max_attachment_bytes == 20_000_000
    assert config.max_total_bytes == 50_000_000


def test_string_and_text_envelope_resolve_equivalently() -> None:
    resolver = InputResolver(InputConfig())
    first = asyncio.run(resolver.resolve("hello", ModelCapabilities()))
    second = asyncio.run(resolver.resolve(UserInput(text="hello"), ModelCapabilities()))
    assert first == second


def test_path_beside_allowed_root_is_rejected(tmp_path: Path) -> None:
    allowed = tmp_path / "safe"
    allowed.mkdir()
    outside = tmp_path / "safe-neighbor"
    outside.mkdir()
    source = outside / "image.png"
    source.write_bytes(b"image")
    resolver = InputResolver(InputConfig(allowed_paths=[str(allowed)]))

    with pytest.raises(InputPathError):
        asyncio.run(
            resolver.resolve(
                UserInput(parts=(ImageInput.from_path(source, media_type="image/png"),)),
                ModelCapabilities(image=True),
            )
        )


def test_symlink_resolving_outside_allowed_root_is_rejected(tmp_path: Path) -> None:
    allowed = tmp_path / "safe"
    allowed.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    link = allowed / "link.png"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this platform: {exc}")
    resolver = InputResolver(InputConfig(allowed_paths=[str(allowed)]))

    with pytest.raises(InputPathError):
        asyncio.run(
            resolver.resolve(
                UserInput(parts=(ImageInput.from_path(link, media_type="image/png"),)),
                ModelCapabilities(image=True),
            )
        )


def test_allowed_path_is_read_and_media_type_is_inferred(tmp_path: Path) -> None:
    allowed = tmp_path / "safe"
    allowed.mkdir()
    source = allowed / "image.png"
    source.write_bytes(b"image")
    resolver = InputResolver(InputConfig(allowed_paths=[str(allowed)]))

    resolved = asyncio.run(
        resolver.resolve(
            UserInput(parts=(ImageInput.from_path(source),)),
            ModelCapabilities(image=True),
        )
    )

    assert resolved.attachments[0].filename == "image.png"
    assert resolved.attachments[0].media_type == "image/png"


def test_url_credentials_are_rejected_without_leaking_secrets() -> None:
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=b"never")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        resolver = InputResolver(InputConfig(remote_enabled=True), client=client)
        with pytest.raises(InputNetworkError) as captured:
            asyncio.run(
                resolver.resolve(
                    UserInput(
                        parts=(
                            FileInput.from_url(
                                "https://user:password@93.184.216.34/document",
                                media_type="text/plain",
                            ),
                        )
                    ),
                    ModelCapabilities(file=True),
                )
            )
    finally:
        asyncio.run(client.aclose())

    error_text = f"{captured.value} {captured.value.details}"
    assert "user" not in error_text
    assert "password" not in error_text
    assert "93.184.216.34" not in error_text
    assert requests == []


def test_public_redirect_to_loopback_is_rejected_and_responses_are_closed() -> None:
    requests: list[str] = []
    responses: list[httpx.Response] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        response = httpx.Response(
            302,
            headers={"location": "http://127.0.0.1/private"},
        )
        responses.append(response)
        return response

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        resolver = InputResolver(InputConfig(remote_enabled=True), client=client)
        with pytest.raises(InputNetworkError):
            asyncio.run(
                resolver.resolve(
                    UserInput(
                        parts=(
                            FileInput.from_url(
                                "http://93.184.216.34/public",
                                media_type="text/plain",
                            ),
                        )
                    ),
                    ModelCapabilities(file=True),
                )
            )
    finally:
        asyncio.run(client.aclose())

    assert requests == ["http://93.184.216.34/public"]
    assert responses[0].is_closed is True


def test_remote_response_media_type_parameters_are_stripped() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png; charset=binary"},
            content=b"image",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        resolver = InputResolver(
            InputConfig(remote_enabled=True, allow_private_networks=True),
            client=client,
        )
        resolved = asyncio.run(
            resolver.resolve(
                UserInput(
                    parts=(
                        ImageInput.from_url(
                            "http://127.0.0.1/image",
                            filename="image.png",
                        ),
                    )
                ),
                ModelCapabilities(image=True),
            )
        )
    finally:
        asyncio.run(client.aclose())

    assert resolved.attachments[0].media_type == "image/png"


def test_attachment_byte_limit_is_enforced() -> None:
    resolver = InputResolver(
        InputConfig(max_attachment_bytes=4, max_total_bytes=10),
    )

    with pytest.raises(InputSizeLimitError):
        asyncio.run(
            resolver.resolve(
                UserInput(
                    parts=(ImageInput.from_bytes(b"12345", media_type="image/png"),)
                ),
                ModelCapabilities(image=True),
            )
        )


def test_cumulative_byte_limit_is_enforced() -> None:
    resolver = InputResolver(
        InputConfig(max_attachment_bytes=10, max_total_bytes=15),
    )
    parts = (
        FileInput.from_bytes(b"12345678", media_type="text/plain", filename="first.txt"),
        FileInput.from_bytes(b"12345678", media_type="text/plain", filename="second.txt"),
    )

    with pytest.raises(InputSizeLimitError):
        asyncio.run(
            resolver.resolve(
                UserInput(parts=parts),
                ModelCapabilities(file=True),
            )
        )


def test_media_type_must_match_attachment_kind() -> None:
    resolver = InputResolver(InputConfig())

    with pytest.raises(InputMediaTypeError):
        asyncio.run(
            resolver.resolve(
                UserInput(parts=(ImageInput.from_bytes(b"wave", media_type="audio/wav"),)),
                ModelCapabilities(image=True),
            )
        )


def test_unsupported_modality_is_rejected_before_http_request() -> None:
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=b"image")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        resolver = InputResolver(InputConfig(remote_enabled=True), client=client)
        with pytest.raises(UnsupportedInputModalityError):
            asyncio.run(
                resolver.resolve(
                    UserInput(
                        parts=(
                            ImageInput.from_url(
                                "http://93.184.216.34/image",
                                media_type="image/png",
                            ),
                        )
                    ),
                    ModelCapabilities(image=False),
                )
            )
    finally:
        asyncio.run(client.aclose())

    assert requests == []


def test_all_provider_content_blocks_are_built() -> None:
    resolver = InputResolver(InputConfig())
    parts = (
        ImageInput.from_bytes(
            b"image",
            media_type="image/png",
            filename="folder\\image.png",
        ),
        AudioInput.from_bytes(b"wave", media_type="audio/wav", filename="sound.wav"),
        VideoInput.from_bytes(b"video", media_type="video/mp4", filename="video.mp4"),
        FileInput.from_bytes(b"file", media_type="text/plain", filename="file.txt"),
    )

    resolved = asyncio.run(
        resolver.resolve(
            UserInput(text="describe", parts=parts),
            ModelCapabilities(image=True, audio=True, video=True, file=True),
        )
    )

    assert isinstance(resolved.content, list)
    assert isinstance(resolved.content[0], TextContent)
    assert isinstance(resolved.content[1], ImageContent)
    assert isinstance(resolved.content[2], AudioContent)
    assert isinstance(resolved.content[3], VideoContent)
    assert isinstance(resolved.content[4], FileContent)
    assert resolved.attachments[0].filename == "image.png"
    assert resolved.modalities == {item.kind for item in parts}


@settings(max_examples=60)
@given(payload=st.binary(min_size=1, max_size=4096), filename=SAFE_FILENAMES)
def test_text_projection_never_contains_encoded_payload_or_data_url(
    payload: bytes,
    filename: str,
) -> None:
    resolver = InputResolver(InputConfig())
    resolved = asyncio.run(
        resolver.resolve(
            UserInput(
                parts=(
                    FileInput.from_bytes(
                        payload,
                        media_type="text/plain",
                        filename=filename,
                    ),
                )
            ),
            ModelCapabilities(file=True),
        )
    )

    encoded = base64.b64encode(payload).decode()
    assert encoded not in resolved.text_projection
    assert "data:" not in resolved.text_projection


def test_text_projection_redacts_payload_encoded_as_filename() -> None:
    payload = b"projection-secret"
    encoded = base64.b64encode(payload).decode()
    resolver = InputResolver(InputConfig())

    resolved = asyncio.run(
        resolver.resolve(
            UserInput(
                parts=(
                    FileInput.from_bytes(
                        payload,
                        media_type="text/plain",
                        filename=f"{encoded}.txt",
                    ),
                )
            ),
            ModelCapabilities(file=True),
        )
    )

    assert encoded not in resolved.text_projection


def test_text_projection_redacts_api_key_filename() -> None:
    api_key = "sk-example0123456789abcdef"
    resolver = InputResolver(InputConfig())

    resolved = asyncio.run(
        resolver.resolve(
            UserInput(
                parts=(
                    FileInput.from_bytes(
                        b"file",
                        media_type="text/plain",
                        filename=f"api_key={api_key}.txt",
                    ),
                )
            ),
            ModelCapabilities(file=True),
        )
    )

    assert api_key not in resolved.text_projection


def test_text_projection_redacts_data_url_marker_in_filename() -> None:
    resolver = InputResolver(InputConfig())

    resolved = asyncio.run(
        resolver.resolve(
            UserInput(
                parts=(
                    FileInput.from_bytes(
                        b"file",
                        media_type="text/plain",
                        filename="data:payload.txt",
                    ),
                )
            ),
            ModelCapabilities(file=True),
        )
    )

    assert "data:" not in resolved.text_projection


def test_declared_source_kind_must_match_source_value() -> None:
    resolver = InputResolver(InputConfig(remote_enabled=True))
    invalid = ImageInput(
        source_kind=InputSourceKind.URL,
        source=b"not-a-url",
        media_type="image/png",
    )

    with pytest.raises(InvalidInputSourceError, match="source"):
        asyncio.run(
            resolver.resolve(
                UserInput(parts=(invalid,)),
                ModelCapabilities(image=True),
            )
        )
