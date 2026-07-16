"""Typed user input sources for text and multimodal attachments."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputKind(StrEnum):
    """Supported attachment modalities."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


class InputSourceKind(StrEnum):
    """Ways attachment bytes can enter Praxis."""

    BYTES = "bytes"
    PATH = "path"
    URL = "url"


class AttachmentInput(BaseModel):
    """Immutable attachment source envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: InputSourceKind
    source: bytes | Path | str
    media_type: str | None = None
    filename: str | None = None

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: str,
        filename: str | None = None,
    ) -> Self:
        return cls(
            source_kind=InputSourceKind.BYTES,
            source=bytes(data),
            media_type=media_type,
            filename=filename,
        )

    @classmethod
    def from_path(cls, path: str | Path, *, media_type: str | None = None) -> Self:
        return cls(
            source_kind=InputSourceKind.PATH,
            source=Path(path),
            media_type=media_type,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        media_type: str | None = None,
        filename: str | None = None,
    ) -> Self:
        return cls(
            source_kind=InputSourceKind.URL,
            source=url,
            media_type=media_type,
            filename=filename,
        )


class ImageInput(AttachmentInput):
    """Image attachment source."""

    kind: Literal[InputKind.IMAGE] = InputKind.IMAGE


class AudioInput(AttachmentInput):
    """Audio attachment source."""

    kind: Literal[InputKind.AUDIO] = InputKind.AUDIO


class VideoInput(AttachmentInput):
    """Video attachment source."""

    kind: Literal[InputKind.VIDEO] = InputKind.VIDEO


class FileInput(AttachmentInput):
    """File attachment source."""

    kind: Literal[InputKind.FILE] = InputKind.FILE


InputAttachment = Annotated[
    ImageInput | AudioInput | VideoInput | FileInput,
    Field(discriminator="kind"),
]


class UserInput(BaseModel):
    """A user turn containing text, attachments, or both."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = ""
    parts: tuple[InputAttachment, ...] = ()

    @model_validator(mode="after")
    def validate_content(self) -> "UserInput":
        if not self.text.strip() and not self.parts:
            raise ValueError("用户输入必须包含文本或至少一个附件")
        return self


InputValue = str | UserInput
