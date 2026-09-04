"""Praxis 可嵌入 Agent SDK。

运行时实现按需导入，使配置检查、版本查询和仅类型/模型用途不会加载
体量较大的模型适配器。公共导入路径保持为 ``praxis.PraxisRuntime``。
"""

from typing import TYPE_CHECKING

from praxis.config import InputConfig, ModelCapabilities, PraxisConfig, load_config
from praxis.exceptions import (
    InputError,
    InputMediaTypeError,
    InputNetworkError,
    InputPathError,
    InputSizeLimitError,
    InvalidInputSourceError,
    ModelValidationError,
    UnsupportedInputModalityError,
)
from praxis.input_resolver import InputResolver
from praxis.models import (
    AgentEvent,
    AgentEventPayload,
    AgentResponse,
    AttachmentInput,
    AttachmentMetadata,
    AudioContent,
    AudioData,
    AudioInput,
    ContentPart,
    EventPayload,
    EventType,
    FileContent,
    FileData,
    FileInput,
    ImageContent,
    ImageInput,
    ImageUrl,
    InputAttachment,
    InputKind,
    InputSourceKind,
    InputValue,
    ResolvedUserInput,
    TextContent,
    UserInput,
    VideoContent,
    VideoInput,
    VideoUrl,
    validate_content_part,
    validate_content_part_json,
)
from praxis.models.runtime import HealthStatus, RuntimeHealth
from praxis.resources import ResourceController

if TYPE_CHECKING:
    from praxis.runtime import AgentSession, PraxisRuntime

__version__ = "1.0.0"


def __getattr__(name: str) -> object:
    """Resolve heavyweight public runtime types only when requested."""
    if name == "AgentSession":
        from praxis.runtime import AgentSession

        return AgentSession
    if name == "PraxisRuntime":
        from praxis.runtime import PraxisRuntime

        return PraxisRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AgentSession",
    "AgentEvent",
    "AgentEventPayload",
    "AgentResponse",
    "AttachmentInput",
    "AttachmentMetadata",
    "AudioContent",
    "AudioData",
    "AudioInput",
    "ContentPart",
    "FileContent",
    "FileData",
    "FileInput",
    "EventPayload",
    "EventType",
    "HealthStatus",
    "ImageContent",
    "ImageInput",
    "ImageUrl",
    "InputAttachment",
    "InputConfig",
    "InputError",
    "InputKind",
    "InputMediaTypeError",
    "InputNetworkError",
    "InputPathError",
    "InputResolver",
    "InputSizeLimitError",
    "InputSourceKind",
    "InputValue",
    "InvalidInputSourceError",
    "ModelCapabilities",
    "ModelValidationError",
    "PraxisConfig",
    "PraxisRuntime",
    "ResolvedUserInput",
    "ResourceController",
    "RuntimeHealth",
    "TextContent",
    "UnsupportedInputModalityError",
    "UserInput",
    "VideoContent",
    "VideoInput",
    "VideoUrl",
    "load_config",
    "validate_content_part",
    "validate_content_part_json",
]
