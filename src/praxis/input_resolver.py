"""Securely resolve typed user inputs into provider content blocks."""

import asyncio
import base64
import mimetypes
import os
import re
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import urljoin

import httpx

from praxis.config.schemas import InputConfig, ModelCapabilities
from praxis.exceptions import (
    InputMediaTypeError,
    InputNetworkError,
    InputPathError,
    InputSizeLimitError,
    InvalidInputSourceError,
    UnsupportedInputModalityError,
)
from praxis.models.inputs import (
    InputAttachment,
    InputKind,
    InputSourceKind,
    InputValue,
    UserInput,
)
from praxis.models.messages import (
    AttachmentMetadata,
    AudioContent,
    AudioData,
    ContentPart,
    FileContent,
    FileData,
    ImageContent,
    ImageUrl,
    ResolvedUserInput,
    TextContent,
    VideoContent,
    VideoUrl,
)
from praxis.network import (
    HTTP_REDIRECT_STATUS_CODES,
    TransportFactory,
    close_http_resources,
    send_pinned_http_request,
    validate_http_url,
)

PROJECTION_SECRET_PATTERNS = (
    re.compile(r"(?i)data:[^\s]+"),
    re.compile(r"(?i)https?://[^\s/@]+:[^\s/@]+@[^\s]+"),
    re.compile(r"(?i)sk-[a-z0-9_-]{12,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*[^\s]+"),
    re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])"),
)


class OpenedPathLike(Protocol):
    """Structural result required from a secure path opener."""

    stream: BinaryIO
    final_path: Path
    size_bytes: int
    regular: bool
    reparse: bool


@dataclass
class OpenedPath:
    """An opened file handle plus facts obtained from that same handle."""

    stream: BinaryIO
    final_path: Path
    size_bytes: int
    regular: bool
    reparse: bool


PathOpener = Callable[[Path], OpenedPathLike]


def sanitize_projection_text(
    value: str,
    payload_encodings: tuple[str, ...] = (),
) -> str:
    """Redact attachment encodings, credentials, data URLs, and Base64-like tokens."""

    sanitized = value
    for encoding in payload_encodings:
        if encoding:
            sanitized = sanitized.replace(encoding, "[redacted]")
    for pattern in PROJECTION_SECRET_PATTERNS:
        sanitized = pattern.sub("[redacted]", sanitized)
    return sanitized


class InputResolver:
    """Resolve user input according to a fail-closed input policy."""

    def __init__(
        self,
        config: InputConfig,
        *,
        transport_factory: TransportFactory | None = None,
        path_opener: PathOpener | None = None,
    ) -> None:
        self.config = config
        self.transport_factory = transport_factory
        self.path_opener = path_opener or self.open_path_handle
        self.allowed_paths = tuple(
            Path(item).expanduser().resolve() for item in config.allowed_paths
        )
        self.media_types = {
            InputKind.IMAGE: frozenset(config.image_media_types),
            InputKind.AUDIO: frozenset(config.audio_media_types),
            InputKind.VIDEO: frozenset(config.video_media_types),
            InputKind.FILE: frozenset(config.file_media_types),
        }

    async def resolve(
        self,
        value: InputValue,
        capabilities: ModelCapabilities,
    ) -> ResolvedUserInput:
        """Preflight all modalities, then resolve within one cumulative byte budget."""

        normalized = UserInput(text=value) if isinstance(value, str) else value
        for attachment in normalized.parts:
            self.ensure_capability(attachment, capabilities)
        if not normalized.parts:
            return ResolvedUserInput(
                content=normalized.text,
                text_projection=sanitize_projection_text(normalized.text),
            )

        content: list[ContentPart] = []
        if normalized.text:
            content.append(TextContent(text=normalized.text))
        metadata: list[AttachmentMetadata] = []
        payload_encodings: list[str] = []
        total_bytes = 0
        for attachment in normalized.parts:
            remaining_bytes = self.config.max_total_bytes - total_bytes
            content_part, attachment_metadata = await self.resolve_attachment(
                attachment,
                capabilities,
                remaining_bytes=remaining_bytes,
            )
            total_bytes += attachment_metadata.size_bytes
            content.append(content_part)
            metadata.append(attachment_metadata)
            payload_encodings.append(self.content_payload_encoding(content_part))

        attachments = tuple(metadata)
        return ResolvedUserInput(
            content=content,
            text_projection=self.build_text_projection(
                normalized.text,
                attachments,
                tuple(payload_encodings),
            ),
            modalities=frozenset(item.kind for item in attachments),
            attachments=attachments,
        )

    def ensure_capability(
        self,
        attachment: InputAttachment,
        capabilities: ModelCapabilities,
    ) -> None:
        """Reject unsupported modalities before any attachment source access."""

        capability_enabled = {
            InputKind.IMAGE: capabilities.image,
            InputKind.AUDIO: capabilities.audio,
            InputKind.VIDEO: capabilities.video,
            InputKind.FILE: capabilities.file,
        }[attachment.kind]
        if not capability_enabled:
            raise UnsupportedInputModalityError(
                "模型不支持此输入模态",
                details={"kind": attachment.kind.value},
            )

    async def resolve_attachment(
        self,
        attachment: InputAttachment,
        capabilities: ModelCapabilities,
        *,
        remaining_bytes: int | None = None,
    ) -> tuple[ContentPart, AttachmentMetadata]:
        """Resolve one attachment without exceeding attachment or remaining total budget."""

        self.ensure_capability(attachment, capabilities)
        effective_limit = min(
            self.config.max_attachment_bytes,
            self.config.max_total_bytes if remaining_bytes is None else remaining_bytes,
        )
        if effective_limit < 0:
            raise InputSizeLimitError("附件累计大小超过配置上限")

        data: bytes
        media_type = attachment.media_type
        filename = attachment.filename or f"{attachment.kind.value}.bin"
        if attachment.source_kind == InputSourceKind.BYTES:
            if not isinstance(attachment.source, bytes):
                raise InvalidInputSourceError(
                    "source 与声明的来源类型不匹配",
                    details={"kind": attachment.kind.value},
                )
            self.enforce_attachment_size(
                len(attachment.source),
                attachment.kind,
                effective_limit,
            )
            data = attachment.source
        elif attachment.source_kind == InputSourceKind.PATH:
            if not isinstance(attachment.source, (str, Path)):
                raise InvalidInputSourceError(
                    "source 与声明的来源类型不匹配",
                    details={"kind": attachment.kind.value},
                )
            source_path = Path(attachment.source)
            data = await asyncio.to_thread(self.read_path, source_path, effective_limit)
            filename = source_path.name
            media_type = media_type or mimetypes.guess_type(source_path.name)[0]
        elif attachment.source_kind == InputSourceKind.URL:
            if not isinstance(attachment.source, str):
                raise InvalidInputSourceError(
                    "source 与声明的来源类型不匹配",
                    details={"kind": attachment.kind.value},
                )
            data, response_media_type = await self.download_url(
                attachment.source,
                effective_limit,
            )
            media_type = response_media_type or media_type
            filename = attachment.filename or f"remote-{attachment.kind.value}.bin"
        else:
            raise InvalidInputSourceError(
                "source 来源类型无效",
                details={"kind": attachment.kind.value},
            )

        normalized_media_type = self.validate_media_type(attachment.kind, media_type)
        safe_filename = self.sanitize_filename(filename, attachment.kind)
        content_part = self.build_content_part(
            attachment.kind,
            data,
            normalized_media_type,
            safe_filename,
        )
        payload_encoding = self.content_payload_encoding(content_part)
        final_filename = self.sanitize_filename(
            safe_filename,
            attachment.kind,
            (payload_encoding,),
        )
        if isinstance(content_part, FileContent) and final_filename != safe_filename:
            content_part = FileContent(
                file=FileData(
                    filename=final_filename,
                    file_data=content_part.file.file_data,
                )
            )
        return content_part, AttachmentMetadata(
            kind=attachment.kind,
            filename=final_filename,
            media_type=normalized_media_type,
            size_bytes=len(data),
        )

    def read_path(self, path: str | Path, byte_limit: int | None = None) -> bytes:
        """Read only from a verified open handle whose final target remains authorized."""

        if not self.allowed_paths:
            raise InputPathError("未配置附件路径授权根目录")
        candidate = Path(path).expanduser()
        try:
            preliminary = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            raise InputPathError("附件路径不可访问") from None
        if not self.path_is_allowed(preliminary):
            raise InputPathError("附件路径不在授权根目录内")
        try:
            opened = self.path_opener(candidate)
        except (OSError, RuntimeError, ValueError):
            raise InputPathError("附件路径不可访问") from None

        try:
            final_path = Path(os.path.abspath(opened.final_path))
            if opened.reparse or not opened.regular:
                raise InputPathError("附件路径必须是非重解析普通文件")
            if not self.path_is_allowed(final_path):
                raise InputPathError("附件句柄的最终目标不在授权根目录内")
            limit = self.config.max_attachment_bytes if byte_limit is None else byte_limit
            self.enforce_attachment_size(opened.size_bytes, byte_limit=limit)
            body = bytearray()
            while chunk := opened.stream.read(65_536):
                self.append_bounded(body, chunk, limit)
            return bytes(body)
        except InputSizeLimitError:
            raise
        except InputPathError:
            raise
        except (OSError, RuntimeError, ValueError):
            raise InputPathError("附件文件读取失败") from None
        finally:
            opened.stream.close()

    def path_is_allowed(self, path: Path) -> bool:
        """Check a normalized final path against snapshotted resolved roots."""

        return any(path == allowed or allowed in path.parents for allowed in self.allowed_paths)

    def open_path_handle(self, path: Path) -> OpenedPath:
        """Open one path without following its final link and inspect that same handle."""

        if os.name == "nt":
            return self.open_windows_path_handle(path)
        return self.open_linux_path_handle(path)

    def open_linux_path_handle(self, path: Path) -> OpenedPath:
        """Open and identify a Linux file through O_NOFOLLOW and /proc/self/fd."""

        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise OSError("O_NOFOLLOW unavailable")
        flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        transferred = False
        try:
            information = os.fstat(descriptor)
            final_path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            stream = os.fdopen(descriptor, "rb", closefd=True)
            transferred = True
            return OpenedPath(
                stream=stream,
                final_path=final_path,
                size_bytes=information.st_size,
                regular=stat.S_ISREG(information.st_mode),
                reparse=False,
            )
        finally:
            if not transferred:
                os.close(descriptor)

    def open_windows_path_handle(self, path: Path) -> OpenedPath:
        """Open and identify a Windows file through a non-following Win32 handle."""

        if sys.platform != "win32":
            raise OSError("Windows secure path handles are unavailable")

        import ctypes
        import ctypes.wintypes as wintypes
        import msvcrt

        GENERIC_READ = 0x80000000
        FILE_SHARE_ALL = 0x00000007
        OPEN_EXISTING = 3
        FILE_ATTRIBUTE_DIRECTORY = 0x00000010
        FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
        FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
        FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        get_information = kernel32.GetFileInformationByHandle
        get_information.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
        get_information.restype = wintypes.BOOL
        get_final_path = kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        get_final_path.restype = wintypes.DWORD
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        handle = create_file(
            str(path),
            GENERIC_READ,
            FILE_SHARE_ALL,
            None,
            OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle is None or handle == invalid_handle:
            raise OSError(ctypes.get_last_error(), "secure path open failed")

        transferred = False
        try:
            information_buffer = ctypes.create_string_buffer(52)
            if not get_information(handle, ctypes.byref(information_buffer)):
                raise OSError(ctypes.get_last_error(), "secure handle inspection failed")
            attributes = int.from_bytes(information_buffer.raw[0:4], "little")
            size_high = int.from_bytes(information_buffer.raw[32:36], "little")
            size_low = int.from_bytes(information_buffer.raw[36:40], "little")
            path_buffer = ctypes.create_unicode_buffer(32_768)
            path_length = get_final_path(handle, path_buffer, len(path_buffer), 0)
            if path_length == 0 or path_length >= len(path_buffer):
                raise OSError(ctypes.get_last_error(), "final handle path unavailable")
            final_name = path_buffer.value
            if final_name.startswith("\\\\?\\UNC\\"):
                final_name = "\\\\" + final_name[8:]
            elif final_name.startswith("\\\\?\\"):
                final_name = final_name[4:]

            descriptor = msvcrt.open_osfhandle(
                int(handle),
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
            try:
                stream = os.fdopen(descriptor, "rb", closefd=True)
            except (OSError, ValueError):
                os.close(descriptor)
                transferred = True
                raise
            transferred = True
            return OpenedPath(
                stream=stream,
                final_path=Path(final_name),
                size_bytes=(size_high << 32) | size_low,
                regular=not bool(attributes & FILE_ATTRIBUTE_DIRECTORY),
                reparse=bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT),
            )
        finally:
            if not transferred:
                close_handle(handle)

    async def download_url(
        self,
        url: str,
        byte_limit: int | None = None,
    ) -> tuple[bytes, str | None]:
        """Download through DNS-pinned requests with per-hop validation and bounded streaming."""

        if not self.config.remote_enabled:
            raise InputNetworkError("远程附件输入未启用")
        try:
            target = await validate_http_url(url, self.config.allow_private_networks)
        except ValueError as exc:
            raise InputNetworkError(str(exc)) from None

        limit = self.config.max_attachment_bytes if byte_limit is None else byte_limit
        redirect_count = 0
        while True:
            try:
                response, client = await send_pinned_http_request(
                    target,
                    self.transport_factory,
                    timeout=self.config.remote_timeout,
                )
            except ValueError as exc:
                raise InputNetworkError(str(exc)) from None
            try:
                if response.status_code in HTTP_REDIRECT_STATUS_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise InputNetworkError("重定向响应缺少 Location")
                    candidate = urljoin(target.original_url, location)
                    try:
                        validated_candidate = await validate_http_url(
                            candidate,
                            self.config.allow_private_networks,
                        )
                    except ValueError as exc:
                        raise InputNetworkError(str(exc)) from None
                    if redirect_count >= self.config.max_redirects:
                        raise InputNetworkError("重定向次数超过配置上限")
                    redirect_count += 1
                    target = validated_candidate
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    raise InputNetworkError("远程附件返回错误状态") from None
                body = bytearray()
                try:
                    async for chunk in response.aiter_bytes():
                        self.append_bounded(body, chunk, limit)
                except (httpx.HTTPError, OSError):
                    raise InputNetworkError("远程附件响应读取失败") from None
                return bytes(body), response.headers.get("content-type")
            finally:
                active_error = sys.exc_info()[0] is not None
                close_failed = await close_http_resources(response, client)
                if close_failed and not active_error:
                    raise InputNetworkError("远程附件响应关闭失败") from None

    def validate_media_type(self, kind: InputKind, media_type: str | None) -> str:
        """Normalize MIME parameters and enforce a snapshotted modality allowlist."""

        if media_type is None:
            raise InputMediaTypeError(
                "附件缺少可验证的媒体类型",
                details={"kind": kind.value},
            )
        normalized = media_type.split(";", maxsplit=1)[0].strip().lower()
        if normalized not in self.media_types[kind]:
            raise InputMediaTypeError(
                "媒体类型不允许用于此附件模态",
                details={"kind": kind.value},
            )
        return normalized

    def build_content_part(
        self,
        kind: InputKind,
        data: bytes,
        media_type: str,
        filename: str,
    ) -> ContentPart:
        """Encode resolved bytes in the provider block required by the modality."""

        encoded = base64.b64encode(data).decode("ascii")
        if kind == InputKind.AUDIO:
            audio_format = "wav" if media_type == "audio/wav" else "mp3"
            return AudioContent(input_audio=AudioData(data=encoded, format=audio_format))
        data_url = f"data:{media_type};base64,{encoded}"
        if kind == InputKind.IMAGE:
            return ImageContent(image_url=ImageUrl(url=data_url))
        if kind == InputKind.VIDEO:
            return VideoContent(video_url=VideoUrl(url=data_url))
        return FileContent(file=FileData(filename=filename, file_data=data_url))

    def content_payload_encoding(self, content_part: ContentPart) -> str:
        """Return the existing provider encoding without encoding attachment bytes again."""

        if isinstance(content_part, AudioContent):
            return content_part.input_audio.data
        if isinstance(content_part, ImageContent):
            return content_part.image_url.url.partition(",")[2]
        if isinstance(content_part, VideoContent):
            return content_part.video_url.url.partition(",")[2]
        if isinstance(content_part, FileContent):
            return content_part.file.file_data.partition(",")[2]
        return ""

    def build_text_projection(
        self,
        text: str,
        attachments: tuple[AttachmentMetadata, ...],
        payload_encodings: tuple[str, ...] = (),
    ) -> str:
        """Build a byte-free, URL-free, secret-redacted projection."""

        lines = [sanitize_projection_text(text, payload_encodings)] if text else []
        lines.extend(
            sanitize_projection_text(
                (
                    f"[attachment kind={item.kind.value} filename={item.filename} "
                    f"media_type={item.media_type} size_bytes={item.size_bytes}]"
                ),
                payload_encodings,
            )
            for item in attachments
        )
        return "\n".join(lines)

    def enforce_attachment_size(
        self,
        observed_bytes: int,
        kind: InputKind | None = None,
        byte_limit: int | None = None,
    ) -> None:
        """Reject an attachment before it exceeds its effective byte bound."""

        limit = self.config.max_attachment_bytes if byte_limit is None else byte_limit
        if observed_bytes > limit:
            details: dict[str, int | str] = {
                "limit_bytes": limit,
                "observed_bytes": observed_bytes,
            }
            if kind is not None:
                details["kind"] = kind.value
            raise InputSizeLimitError("附件大小超过配置上限", details=details)

    def append_bounded(
        self,
        body: bytearray,
        chunk: bytes,
        byte_limit: int,
    ) -> None:
        """Check the resulting size before extending an attachment buffer."""

        observed_bytes = len(body) + len(chunk)
        self.enforce_attachment_size(observed_bytes, byte_limit=byte_limit)
        body.extend(chunk)

    def sanitize_filename(
        self,
        filename: str,
        kind: InputKind,
        payload_encodings: tuple[str, ...] = (),
    ) -> str:
        """Return a printable basename processed by the shared projection sanitizer."""

        basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        printable = "".join(character for character in basename if character.isprintable())
        sanitized = printable.strip()
        if sanitized in {"", ".", ".."}:
            sanitized = f"{kind.value}.bin"
        return sanitize_projection_text(sanitized, payload_encodings)
