"""MCP 集成数据模型——S5 MCP 组件共享类型。"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, field_serializer, model_validator

from praxis.config.immutable import FrozenMapping, freeze_mapping, thaw_value
from praxis.models.base import SafeBaseModel


class MCPTransportType(StrEnum):
    """MCP 传输类型。"""

    STDIO = "stdio"
    HTTP = "http"


class MCPServerStatus(StrEnum):
    """MCP 服务器连接状态。"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class MCPServerConfig(SafeBaseModel):
    """MCP 服务器连接配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    transport: MCPTransportType = MCPTransportType.STDIO
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = Field(default_factory=FrozenMapping)
    url: str = ""
    headers: Mapping[str, str] = Field(default_factory=FrozenMapping)
    timeout: float = Field(default=30.0, gt=0, le=300.0)
    reconnect_attempts: int = Field(default=3, ge=0, le=20)
    reconnect_delay: float = Field(default=1.0, ge=0, le=60.0)
    reconnect_max_delay: float = Field(default=30.0, gt=0, le=300.0)
    health_check_interval: float = Field(default=5.0, gt=0, le=300.0)

    @model_validator(mode="after")
    def validate_transport_endpoint(self) -> "MCPServerConfig":
        if self.transport is MCPTransportType.STDIO:
            if not self.command.strip():
                raise ValueError("stdio MCP Server 必须配置 command")
            if self.url:
                raise ValueError("stdio MCP Server 不允许配置 url")
            return self.freeze_config_mappings()

        if not self.url.strip():
            raise ValueError("HTTP MCP Server 必须配置 url")
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("HTTP MCP Server url 必须是有效的 HTTP/HTTPS 地址")
        if self.command or self.args or self.env:
            raise ValueError("HTTP MCP Server 不允许配置 command、args 或 env")
        return self.freeze_config_mappings()

    def freeze_config_mappings(self) -> "MCPServerConfig":
        object.__setattr__(self, "env", freeze_mapping(self.env))
        object.__setattr__(self, "headers", freeze_mapping(self.headers))
        return self

    @field_serializer("env", "headers", when_used="always")
    def serialize_config_mapping(self, value: Mapping[str, str]) -> object:
        return thaw_value(value)


class MCPResourceInfo(SafeBaseModel):
    """MCP 资源信息。"""

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = "text/plain"


class MCPResourceContent(SafeBaseModel):
    """MCP 资源内容。"""

    uri: str
    mime_type: str = "text/plain"
    text: str | None = None
    blob: str | None = None


class MCPPromptInfo(SafeBaseModel):
    """MCP 提示模板信息。"""

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=lambda: [])


class MCPPromptMessage(SafeBaseModel):
    """MCP 提示模板消息。"""

    role: str
    content: str


class MCPToolInfo(SafeBaseModel):
    """MCP 工具信息。"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    server_name: str = ""


class MCPToolResult(SafeBaseModel):
    """MCP 工具调用结果。"""

    content: list[dict[str, Any]] = Field(default_factory=lambda: [])
    is_error: bool = False


class MCPServerCapabilities(SafeBaseModel):
    """MCP 服务器能力。"""

    tools: bool = False
    resources: bool = False
    prompts: bool = False
    sampling: bool = False


class MCPElicitationRequest(SafeBaseModel):
    """MCP Elicitation 请求。"""

    server_name: str
    message: str
    request_schema: dict[str, Any] = Field(default_factory=dict)
    url: str | None = None


class MCPElicitationResponse(SafeBaseModel):
    """MCP Elicitation 响应。"""

    accepted: bool
    data: dict[str, Any] = Field(default_factory=dict)


class MCPSamplingRequest(SafeBaseModel):
    """MCP Sampling 请求。"""

    server_name: str
    messages: list[dict[str, Any]] = Field(default_factory=lambda: [])
    model_preferences: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = 4096
    tools: list[dict[str, Any]] = Field(default_factory=lambda: [])


class MCPTaskStatus(StrEnum):
    """MCP Task 状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MCPTaskInfo(SafeBaseModel):
    """MCP Task 信息。"""

    task_id: str
    server_name: str
    status: MCPTaskStatus = MCPTaskStatus.PENDING
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
