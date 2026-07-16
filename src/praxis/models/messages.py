"""消息类型定义——跨 S4、S6、S7、S11 共享。"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from praxis.models.inputs import InputKind
from praxis.models.tools import ToolCall


class Role(StrEnum):
    """消息角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TextContent(BaseModel):
    """文本内容块。"""

    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    """图片 URL 及细节级别。"""

    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImageContent(BaseModel):
    """图片内容块。"""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


class AudioData(BaseModel):
    """Provider audio payload and encoding."""

    data: str
    format: Literal["wav", "mp3"]


class AudioContent(BaseModel):
    """Provider audio content block."""

    type: Literal["input_audio"] = "input_audio"
    input_audio: AudioData


class VideoUrl(BaseModel):
    """Provider video data URL."""

    url: str


class VideoContent(BaseModel):
    """Provider video content block."""

    type: Literal["video_url"] = "video_url"
    video_url: VideoUrl


class FileData(BaseModel):
    """Provider file payload."""

    filename: str
    file_data: str


class FileContent(BaseModel):
    """Provider file content block."""

    type: Literal["file"] = "file"
    file: FileData


ContentPart = Annotated[
    TextContent | ImageContent | AudioContent | VideoContent | FileContent,
    Field(discriminator="type"),
]


class AttachmentMetadata(BaseModel):
    """Non-sensitive facts retained after attachment resolution."""

    model_config = ConfigDict(frozen=True)

    kind: InputKind
    filename: str
    media_type: str
    size_bytes: int = Field(ge=0)


class ResolvedUserInput(BaseModel):
    """Provider content plus a safe text-only projection."""

    model_config = ConfigDict(frozen=True)

    content: str | list[ContentPart]
    text_projection: str
    modalities: frozenset[InputKind] = frozenset()
    attachments: tuple[AttachmentMetadata, ...] = ()


class Message(BaseModel):
    """统一消息模型，兼容 OpenAI 消息格式。"""

    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
