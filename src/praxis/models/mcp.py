"""MCP 集成数据模型——S5 MCP 组件共享类型。"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class MCPServerConfig(BaseModel):
    """MCP 服务器连接配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    transport: MCPTransportType = MCPTransportType.STDIO
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = Field(default=30.0, gt=0, le=300.0)
    reconnect_attempts: int = Field(default=3, ge=0, le=20)
    reconnect_delay: float = Field(default=1.0, ge=0, le=60.0)


class MCPResourceInfo(BaseModel):
    """MCP 资源信息。"""

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = "text/plain"


class MCPResourceContent(BaseModel):
    """MCP 资源内容。"""

    uri: str
    mime_type: str = "text/plain"
    text: str | None = None
    blob: str | None = None


class MCPPromptInfo(BaseModel):
    """MCP 提示模板信息。"""

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=lambda: [])


class MCPPromptMessage(BaseModel):
    """MCP 提示模板消息。"""

    role: str
    content: str


class MCPToolInfo(BaseModel):
    """MCP 工具信息。"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    server_name: str = ""


class MCPToolResult(BaseModel):
    """MCP 工具调用结果。"""

    content: list[dict[str, Any]] = Field(default_factory=lambda: [])
    is_error: bool = False


class MCPServerCapabilities(BaseModel):
    """MCP 服务器能力。"""

    tools: bool = False
    resources: bool = False
    prompts: bool = False
    sampling: bool = False


class MCPElicitationRequest(BaseModel):
    """MCP Elicitation 请求。"""

    server_name: str
    message: str
    request_schema: dict[str, Any] = Field(default_factory=dict)
    url: str | None = None


class MCPElicitationResponse(BaseModel):
    """MCP Elicitation 响应。"""

    accepted: bool
    data: dict[str, Any] = Field(default_factory=dict)


class MCPSamplingRequest(BaseModel):
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


class MCPTaskInfo(BaseModel):
    """MCP Task 信息。"""

    task_id: str
    server_name: str
    status: MCPTaskStatus = MCPTaskStatus.PENDING
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
