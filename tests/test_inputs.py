"""Typed multimodal input and secure resolution tests."""

import asyncio
import base64
import io
import socket
import traceback
from pathlib import Path

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

import praxis
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
from praxis.models import ContentPart, TextContent

SAFE_FILENAMES = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cc", "Cs"),
        blacklist_characters="/\\:",
    ),
    min_size=1,
    max_size=40,
).filter(lambda value: value.strip() not in {"", ".", ".."})

ATTACK_FILENAMES = st.sampled_from(
    [
        "data:payload.txt",
        "api_key=sk-example0123456789abcdef.txt",
        "VGhpcyBpcyBhbiB1bnJlbGF0ZWQgQmFzZTY0IHRva2Vu.txt",
    ]
)

ATTACK_TEXTS = st.sampled_from(
    [
        "",
        "api_key=sk-example0123456789abcdef",
        "data:text/plain;base64,c2VjcmV0",
        "VGhpcyBpcyBhbiB1bnJlbGF0ZWQgQmFzZTY0IHRva2Vu",
    ]
)


class OpenedPathRecord:
    """Test double for an already-opened path handle."""

    def __init__(
        self,
        stream: io.BytesIO,
        final_path: Path,
        size_bytes: int,
        *,
        regular: bool = True,
        reparse: bool = False,
    ) -> None:
        self.stream = stream
        self.final_path = final_path
        self.size_bytes = size_bytes
        self.regular = regular
        self.reparse = reparse


class FailingResponseStream(httpx.AsyncByteStream):
    """Stream that exposes whether read failures are safely mapped."""

    async def __aiter__(self):
        yield b"safe"
        raise httpx.ReadError("credential_query_marker")

    async def aclose(self) -> None:
        return None


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

    resolver = InputResolver(
        InputConfig(remote_enabled=True),
        transport=httpx.MockTransport(handle),
    )
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

    resolver = InputResolver(
        InputConfig(remote_enabled=True),
        transport=httpx.MockTransport(handle),
    )
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

    assert requests == ["http://93.184.216.34/public"]
    assert responses[0].is_closed is True


def test_remote_response_media_type_parameters_are_stripped() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png; charset=binary"},
            content=b"image",
        )

    resolver = InputResolver(
        InputConfig(remote_enabled=True, allow_private_networks=True),
        transport=httpx.MockTransport(handle),
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

    resolver = InputResolver(
        InputConfig(remote_enabled=True),
        transport=httpx.MockTransport(handle),
    )
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
@given(
    payload=st.binary(min_size=1, max_size=4096),
    filename=st.one_of(SAFE_FILENAMES, ATTACK_FILENAMES),
    text=ATTACK_TEXTS,
)
def test_text_projection_never_contains_encoded_payload_or_data_url(
    payload: bytes,
    filename: str,
    text: str,
) -> None:
    resolver = InputResolver(InputConfig())
    resolved = asyncio.run(
        resolver.resolve(
            UserInput(
                text=text,
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
    assert "sk-example0123456789abcdef" not in resolved.text_projection
    assert "VGhpcyBpcyBhbiB1bnJlbGF0ZWQgQmFzZTY0IHRva2Vu" not in resolved.text_projection


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


def test_dns_target_is_pinned_and_preserves_host_and_sni(monkeypatch) -> None:
    dns_results = ["93.184.216.34", "127.0.0.1"]
    dns_calls: list[str] = []
    requests: list[httpx.Request] = []

    def resolve_host(host: str, port: int, **options):
        dns_calls.append(host)
        address = dns_results[min(len(dns_calls) - 1, 1)]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=b"safe",
        )

    monkeypatch.setattr(socket, "getaddrinfo", resolve_host)
    resolver = InputResolver(
        InputConfig(remote_enabled=True),
        transport=httpx.MockTransport(handle),
    )

    resolved = asyncio.run(
        resolver.resolve(
            UserInput(
                parts=(FileInput.from_url("https://example.com/file", filename="file.txt"),)
            ),
            ModelCapabilities(file=True),
        )
    )

    assert resolved.attachments[0].size_bytes == 4
    assert dns_calls == ["example.com"]
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == "example.com"
    assert requests[0].extensions["sni_hostname"] == "example.com"


def test_redirect_never_auto_follows_to_private_target(monkeypatch) -> None:
    requested_hosts: list[str] = []

    def resolve_host(host: str, port: int, **options):
        address = "93.184.216.34" if host == "example.com" else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    def handle(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    monkeypatch.setattr(socket, "getaddrinfo", resolve_host)
    resolver = InputResolver(
        InputConfig(remote_enabled=True),
        transport=httpx.MockTransport(handle),
    )

    with pytest.raises(InputNetworkError):
        asyncio.run(
            resolver.resolve(
                UserInput(
                    parts=(
                        FileInput.from_url(
                            "http://example.com/public",
                            media_type="text/plain",
                        ),
                    )
                ),
                ModelCapabilities(file=True),
            )
        )

    assert requested_hosts == ["93.184.216.34"]


def test_projection_sanitizes_text_and_unrelated_base64_filename() -> None:
    original_text = (
        "api_key=sk-example0123456789abcdef "
        "data:text/plain;base64,c2VjcmV0 "
        "VGhpcyBpcyBhbiB1bnJlbGF0ZWQgQmFzZTY0IHRva2Vu"
    )
    unrelated_encoding = "QW5vdGhlciB1bnJlbGF0ZWQgQmFzZTY0IHZhbHVl"
    resolver = InputResolver(InputConfig())

    resolved = asyncio.run(
        resolver.resolve(
            UserInput(
                text=original_text,
                parts=(
                    FileInput.from_bytes(
                        b"file",
                        media_type="text/plain",
                        filename=f"{unrelated_encoding}.txt",
                    ),
                ),
            ),
            ModelCapabilities(file=True),
        )
    )

    assert isinstance(resolved.content, list)
    assert isinstance(resolved.content[0], TextContent)
    assert resolved.content[0].text == original_text
    assert "sk-example0123456789abcdef" not in resolved.text_projection
    assert "data:" not in resolved.text_projection
    assert "VGhpcyBpcyBhbiB1bnJlbGF0ZWQgQmFzZTY0IHRva2Vu" not in resolved.text_projection
    assert unrelated_encoding not in resolved.text_projection


def test_network_error_traceback_does_not_retain_url_query(monkeypatch) -> None:
    query_marker = "credential_query_marker"

    def resolve_host(host: str, port: int, **options):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed request {request.url}", request=request)

    monkeypatch.setattr(socket, "getaddrinfo", resolve_host)
    resolver = InputResolver(
        InputConfig(remote_enabled=True),
        transport=httpx.MockTransport(fail),
    )
    with pytest.raises(InputNetworkError) as captured:
        asyncio.run(
            resolver.resolve(
                UserInput(
                    parts=(
                        FileInput.from_url(
                            f"http://example.com/file?api_key={query_marker}",
                            media_type="text/plain",
                        ),
                    )
                ),
                ModelCapabilities(file=True),
            )
        )

    rendered = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert query_marker not in rendered
    assert "api_key=" not in rendered
    assert "example.com/file" not in rendered


def test_stream_read_error_is_typed_and_does_not_leak_traceback(monkeypatch) -> None:
    def resolve_host(host: str, port: int, **options):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    def stream_failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=FailingResponseStream(),
        )

    monkeypatch.setattr(socket, "getaddrinfo", resolve_host)
    resolver = InputResolver(
        InputConfig(remote_enabled=True),
        transport=httpx.MockTransport(stream_failure),
    )
    with pytest.raises(InputNetworkError) as captured:
        asyncio.run(
            resolver.resolve(
                UserInput(
                    parts=(
                        FileInput.from_url(
                            "http://example.com/file?api_key=credential_query_marker",
                            media_type="text/plain",
                        ),
                    )
                ),
                ModelCapabilities(file=True),
            )
        )

    rendered = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert "credential_query_marker" not in rendered
    assert "httpx.ReadError" not in rendered


def test_path_error_traceback_does_not_retain_source_path(tmp_path: Path) -> None:
    allowed = tmp_path / "safe"
    allowed.mkdir()
    source = allowed / "sensitive-source-name.txt"
    resolver = InputResolver(InputConfig(allowed_paths=[str(allowed)]))

    with pytest.raises(InputPathError) as captured:
        asyncio.run(
            resolver.resolve(
                UserInput(parts=(FileInput.from_path(source, media_type="text/plain"),)),
                ModelCapabilities(file=True),
            )
        )

    rendered = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert str(source) not in rendered
    assert "sensitive-source-name.txt" not in rendered


def test_all_capabilities_are_preflighted_before_any_source_access(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"file")

    resolver = InputResolver(
        InputConfig(allowed_paths=[str(tmp_path)], remote_enabled=True),
        transport=httpx.MockTransport(handle),
    )
    parts = (
        FileInput.from_path(missing, media_type="text/plain"),
        FileInput.from_url("http://93.184.216.34/file", media_type="text/plain"),
        ImageInput.from_bytes(b"image", media_type="image/png"),
    )

    with pytest.raises(UnsupportedInputModalityError):
        asyncio.run(
            resolver.resolve(
                UserInput(parts=parts),
                ModelCapabilities(file=True, image=False),
            )
        )

    assert requests == []


def test_remaining_total_budget_is_checked_before_second_encoding(monkeypatch) -> None:
    resolver = InputResolver(InputConfig(max_attachment_bytes=10, max_total_bytes=15))
    encoded_sizes: list[int] = []
    original_builder = resolver.build_content_part

    def record_builder(kind, data, media_type, filename):
        encoded_sizes.append(len(data))
        return original_builder(kind, data, media_type, filename)

    monkeypatch.setattr(resolver, "build_content_part", record_builder)
    parts = (
        FileInput.from_bytes(b"12345678", media_type="text/plain", filename="first.txt"),
        FileInput.from_bytes(b"12345678", media_type="text/plain", filename="second.txt"),
    )

    with pytest.raises(InputSizeLimitError):
        asyncio.run(
            resolver.resolve(UserInput(parts=parts), ModelCapabilities(file=True))
        )

    assert encoded_sizes == [8]


def test_oversized_chunk_is_rejected_before_buffer_extension() -> None:
    resolver = InputResolver(InputConfig(max_attachment_bytes=5, max_total_bytes=5))
    body = bytearray(b"1234")

    with pytest.raises(InputSizeLimitError):
        resolver.append_bounded(body, b"56789", 5)

    assert body == b"1234"


def test_opened_handle_outside_root_is_rejected_and_closed(tmp_path: Path) -> None:
    allowed = tmp_path / "safe"
    allowed.mkdir()
    source = allowed / "source.txt"
    source.write_bytes(b"inside")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    stream = io.BytesIO(b"outside")

    def swap_path(path: Path) -> OpenedPathRecord:
        return OpenedPathRecord(stream, outside, 7)

    resolver = InputResolver(
        InputConfig(allowed_paths=[str(allowed)]),
        path_opener=swap_path,
    )

    with pytest.raises(InputPathError):
        resolver.read_path(source)

    assert stream.closed is True


def test_opened_reparse_handle_is_rejected_and_closed(tmp_path: Path) -> None:
    allowed = tmp_path / "safe"
    allowed.mkdir()
    source = allowed / "source.txt"
    source.write_bytes(b"inside")
    stream = io.BytesIO(b"inside")

    def open_reparse(path: Path) -> OpenedPathRecord:
        return OpenedPathRecord(stream, source, 6, reparse=True)

    resolver = InputResolver(
        InputConfig(allowed_paths=[str(allowed)]),
        path_opener=open_reparse,
    )

    with pytest.raises(InputPathError):
        resolver.read_path(source)

    assert stream.closed is True


def test_mime_allowlists_are_immutable() -> None:
    config = InputConfig()
    assert isinstance(config.image_media_types, tuple)
    with pytest.raises(AttributeError):
        config.image_media_types.append("image/bmp")  # type: ignore[attr-defined]


def test_content_part_is_top_level_strict_discriminated_public_type() -> None:
    assert praxis.ContentPart is ContentPart
    adapter = TypeAdapter(ContentPart)
    data = {
        "type": "file",
        "file": {
            "filename": "file.txt",
            "file_data": "data:text/plain;base64,ZmlsZQ==",
        },
    }
    parsed = adapter.validate_python(data)
    assert isinstance(parsed, FileContent)
    assert parsed.model_dump() == data

    with pytest.raises(ValidationError):
        adapter.validate_python({**data, "unexpected": True})
    with pytest.raises(ValidationError):
        UserInput(text="hello", unexpected=True)
