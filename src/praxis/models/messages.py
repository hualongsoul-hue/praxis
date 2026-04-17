"""消息类型定义——跨 S4、S6、S7、S11 共享。"""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from praxis.models.tools import ToolCall


class Role(str, Enum):
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


ContentPart = Annotated[Union[TextContent, ImageContent], Field(discriminator="type")]


class Message(BaseModel):
    """统一消息模型，兼容 OpenAI 消息格式。"""

    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
