"""消息类型与安全内容解析入口。

`ContentPart` 仅是类型/Schema 别名。未受信任的数据必须通过
`validate_content_part` 或 `validate_content_part_json` 解析；第三方直接创建的
Pydantic `TypeAdapter` 不属于 Praxis SDK 的安全验证边界。
"""

from enum import StrEnum
from typing import Annotated, Literal, cast

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from praxis.exceptions import ModelValidationError
from praxis.models.base import SafeBaseModel
from praxis.models.inputs import InputKind
from praxis.models.tools import ToolCall


class Role(StrEnum):
    """消息角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StrictContentModel(SafeBaseModel):
    """Provider content model that rejects misspelled or unexpected fields."""

    model_config = ConfigDict(extra="forbid")


class TextContent(StrictContentModel):
    """文本内容块。"""

    type: Literal["text"] = "text"
    text: str


class ImageUrl(StrictContentModel):
    """图片 URL 及细节级别。"""

    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImageContent(StrictContentModel):
    """图片内容块。"""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


class AudioData(StrictContentModel):
    """Provider audio payload and encoding."""

    data: str
    format: Literal["wav", "mp3"]


class AudioContent(StrictContentModel):
    """Provider audio content block."""

    type: Literal["input_audio"] = "input_audio"
    input_audio: AudioData


class VideoUrl(StrictContentModel):
    """Provider video data URL."""

    url: str


class VideoContent(StrictContentModel):
    """Provider video content block."""

    type: Literal["video_url"] = "video_url"
    video_url: VideoUrl


class FileData(StrictContentModel):
    """Provider file payload."""

    filename: str
    file_data: str


class FileContent(StrictContentModel):
    """Provider file content block."""

    type: Literal["file"] = "file"
    file: FileData


ContentPart = Annotated[
    TextContent | ImageContent | AudioContent | VideoContent | FileContent,
    Field(discriminator="type"),
]


def validate_content_part(value: object) -> ContentPart:
    """Parse one content value through the input-free SDK validation boundary."""

    validated: ContentPart | None = None
    validation_failed = False
    try:
        validated = cast(
            ContentPart,
            TypeAdapter(ContentPart).validate_python(value),
        )
    except ValidationError:
        validation_failed = True
    if validation_failed or validated is None:
        raise ModelValidationError("内容块输入验证失败")
    return validated


def validate_content_part_json(data: str | bytes | bytearray) -> ContentPart:
    """Parse one JSON content value through the input-free SDK validation boundary."""

    validated: ContentPart | None = None
    validation_failed = False
    try:
        validated = cast(
            ContentPart,
            TypeAdapter(ContentPart).validate_json(data),
        )
    except ValidationError:
        validation_failed = True
    if validation_failed or validated is None:
        raise ModelValidationError("内容块输入验证失败")
    return validated


class AttachmentMetadata(SafeBaseModel):
    """Non-sensitive facts retained after attachment resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: InputKind
    filename: str
    media_type: str
    size_bytes: int = Field(ge=0)


class ResolvedUserInput(SafeBaseModel):
    """Provider content plus a safe text-only projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | list[ContentPart]
    text_projection: str
    modalities: frozenset[InputKind] = frozenset()
    attachments: tuple[AttachmentMetadata, ...] = ()


class Message(SafeBaseModel):
    """统一消息模型，兼容 OpenAI 消息格式。"""

    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
