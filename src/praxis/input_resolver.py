"""Securely resolve typed user inputs into provider content blocks."""

import asyncio
import base64
import mimetypes
import re
from pathlib import Path
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
from praxis.network import HTTP_REDIRECT_STATUS_CODES, validate_http_url

SECRET_FILENAME_PATTERNS = (
    re.compile(r"(?i)data:"),
    re.compile(r"(?i)sk-[a-z0-9_-]{12,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)[=: -]*[^.\s]{4,}"),
    re.compile(r"[^@\s]+@[^@\s]+"),
)


class InputResolver:
    """Resolve user input according to a fail-closed input policy."""

    def __init__(
        self,
        config: InputConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.allowed_paths = tuple(
            Path(item).expanduser().resolve() for item in config.allowed_paths
        )

    async def resolve(
        self,
        value: InputValue,
        capabilities: ModelCapabilities,
    ) -> ResolvedUserInput:
        """Normalize and resolve all attachments without retaining their raw bytes."""

        normalized = UserInput(text=value) if isinstance(value, str) else value
        if not normalized.parts:
            return ResolvedUserInput(
                content=normalized.text,
                text_projection=normalized.text,
            )

        content: list[ContentPart] = []
        if normalized.text:
            content.append(TextContent(text=normalized.text))
        metadata: list[AttachmentMetadata] = []
        total_bytes = 0
        for attachment in normalized.parts:
            content_part, attachment_metadata = await self.resolve_attachment(
                attachment,
                capabilities,
            )
            total_bytes += attachment_metadata.size_bytes
            if total_bytes > self.config.max_total_bytes:
                raise InputSizeLimitError(
                    "附件累计大小超过配置上限",
                    details={
                        "limit_bytes": self.config.max_total_bytes,
                        "observed_bytes": total_bytes,
                    },
                )
            content.append(content_part)
            metadata.append(attachment_metadata)

        attachments = tuple(metadata)
        return ResolvedUserInput(
            content=content,
            text_projection=self.build_text_projection(normalized.text, attachments),
            modalities=frozenset(item.kind for item in attachments),
            attachments=attachments,
        )

    async def resolve_attachment(
        self,
        attachment: InputAttachment,
        capabilities: ModelCapabilities,
    ) -> tuple[ContentPart, AttachmentMetadata]:
        """Resolve one attachment after checking deployment support."""

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

        data: bytes
        media_type = attachment.media_type
        filename = attachment.filename or f"{attachment.kind.value}.bin"
        if attachment.source_kind == InputSourceKind.BYTES:
            if not isinstance(attachment.source, bytes):
                raise InvalidInputSourceError(
                    "source 与声明的来源类型不匹配",
                    details={"kind": attachment.kind.value},
                )
            data = bytes(attachment.source)
            self.enforce_attachment_size(len(data), attachment.kind)
        elif attachment.source_kind == InputSourceKind.PATH:
            if not isinstance(attachment.source, (str, Path)):
                raise InvalidInputSourceError(
                    "source 与声明的来源类型不匹配",
                    details={"kind": attachment.kind.value},
                )
            source_path = Path(attachment.source)
            data = await asyncio.to_thread(self.read_path, source_path)
            filename = source_path.name
            media_type = media_type or mimetypes.guess_type(source_path.name)[0]
        elif attachment.source_kind == InputSourceKind.URL:
            if not isinstance(attachment.source, str):
                raise InvalidInputSourceError(
                    "source 与声明的来源类型不匹配",
                    details={"kind": attachment.kind.value},
                )
            data, response_media_type = await self.download_url(attachment.source)
            media_type = response_media_type or media_type
            filename = attachment.filename or f"remote-{attachment.kind.value}.bin"
        else:
            raise InvalidInputSourceError(
                "source 来源类型无效",
                details={"kind": attachment.kind.value},
            )

        normalized_media_type = self.validate_media_type(attachment.kind, media_type)
        safe_filename = self.sanitize_filename(filename, attachment.kind)
        payload_encoding = base64.b64encode(data).decode("ascii")
        if payload_encoding and payload_encoding in safe_filename:
            safe_filename = safe_filename.replace(payload_encoding, "[redacted]")
        content_part = self.build_content_part(
            attachment.kind,
            data,
            normalized_media_type,
            safe_filename,
        )
        return content_part, AttachmentMetadata(
            kind=attachment.kind,
            filename=safe_filename,
            media_type=normalized_media_type,
            size_bytes=len(data),
        )

    def read_path(self, path: str | Path) -> bytes:
        """Read an authorized regular file in bounded chunks."""

        try:
            resolved = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InputPathError("附件路径不可访问") from exc
        if not self.allowed_paths:
            raise InputPathError("未配置附件路径授权根目录")
        if not any(
            resolved == allowed or allowed in resolved.parents for allowed in self.allowed_paths
        ):
            raise InputPathError("附件路径不在授权根目录内")
        if not resolved.is_file():
            raise InputPathError("附件路径必须指向普通文件")
        try:
            known_size = resolved.stat().st_size
            self.enforce_attachment_size(known_size)
            body = bytearray()
            with resolved.open("rb") as stream:
                while chunk := stream.read(65_536):
                    body.extend(chunk)
                    self.enforce_attachment_size(len(body))
            return bytes(body)
        except InputSizeLimitError:
            raise
        except OSError as exc:
            raise InputPathError("附件文件读取失败") from exc

    async def download_url(self, url: str) -> tuple[bytes, str | None]:
        """Download a remote attachment with per-hop validation and bounded streaming."""

        if not self.config.remote_enabled:
            raise InputNetworkError("远程附件输入未启用")
        try:
            current_url = await validate_http_url(url, self.config.allow_private_networks)
        except ValueError as exc:
            raise InputNetworkError(str(exc)) from exc

        owns_client = self.client is None
        active_client = self.client or httpx.AsyncClient(
            timeout=self.config.remote_timeout,
            follow_redirects=False,
        )
        redirect_count = 0
        try:
            while True:
                request = active_client.build_request("GET", current_url)
                try:
                    response = await active_client.send(request, stream=True)
                except httpx.HTTPError as exc:
                    raise InputNetworkError("远程附件下载失败") from exc
                try:
                    if response.status_code in HTTP_REDIRECT_STATUS_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise InputNetworkError("重定向响应缺少 Location")
                        candidate = urljoin(current_url, location)
                        try:
                            validated_candidate = await validate_http_url(
                                candidate,
                                self.config.allow_private_networks,
                            )
                        except ValueError as exc:
                            raise InputNetworkError(str(exc)) from exc
                        if redirect_count >= self.config.max_redirects:
                            raise InputNetworkError("重定向次数超过配置上限")
                        redirect_count += 1
                        current_url = validated_candidate
                        continue

                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise InputNetworkError("远程附件返回错误状态") from exc
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        self.enforce_attachment_size(len(body))
                    return bytes(body), response.headers.get("content-type")
                finally:
                    await response.aclose()
        finally:
            if owns_client:
                await active_client.aclose()

    def validate_media_type(self, kind: InputKind, media_type: str | None) -> str:
        """Normalize MIME parameters and enforce the modality allowlist."""

        if media_type is None:
            raise InputMediaTypeError(
                "附件缺少可验证的媒体类型",
                details={"kind": kind.value},
            )
        normalized = media_type.split(";", maxsplit=1)[0].strip().lower()
        allowed = {
            InputKind.IMAGE: self.config.image_media_types,
            InputKind.AUDIO: self.config.audio_media_types,
            InputKind.VIDEO: self.config.video_media_types,
            InputKind.FILE: self.config.file_media_types,
        }[kind]
        if normalized not in allowed:
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

    def build_text_projection(
        self,
        text: str,
        attachments: tuple[AttachmentMetadata, ...],
    ) -> str:
        """Build a byte-free, URL-free projection for memory and guardrails."""

        lines = [text] if text else []
        lines.extend(
            (
                f"[attachment kind={item.kind.value} filename={item.filename} "
                f"media_type={item.media_type} size_bytes={item.size_bytes}]"
            )
            for item in attachments
        )
        return "\n".join(lines)

    def enforce_attachment_size(
        self,
        observed_bytes: int,
        kind: InputKind | None = None,
    ) -> None:
        """Reject an attachment as soon as it exceeds its configured bound."""

        if observed_bytes > self.config.max_attachment_bytes:
            details: dict[str, int | str] = {
                "limit_bytes": self.config.max_attachment_bytes,
                "observed_bytes": observed_bytes,
            }
            if kind is not None:
                details["kind"] = kind.value
            raise InputSizeLimitError("附件大小超过配置上限", details=details)

    def sanitize_filename(self, filename: str, kind: InputKind) -> str:
        """Return a printable basename with credential-like values redacted."""

        basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        printable = "".join(character for character in basename if character.isprintable())
        sanitized = printable.strip()
        if sanitized in {"", ".", ".."}:
            sanitized = f"{kind.value}.bin"
        for pattern in SECRET_FILENAME_PATTERNS:
            sanitized = pattern.sub("[redacted]", sanitized)
        return sanitized
