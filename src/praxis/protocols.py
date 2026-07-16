"""Runtime 依赖倒置边界的公共 Protocol。"""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from praxis.config.schemas import GatewayConfig, ModelCapabilities
from praxis.models.responses import ModelResponse, ModelResponseChunk
from praxis.models.telemetry import AuditEvent
from praxis.models.tools import ApprovalDecision, ApprovalRequest
from praxis.persistence.store import StorageBackend


@runtime_checkable
class ModelGateway(Protocol):
    @property
    def config(self) -> GatewayConfig: ...

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelResponseChunk]: ...

    async def health(self) -> bool: ...

    def capabilities(self, model_name: str | None = None) -> ModelCapabilities: ...

    async def close(self) -> None: ...


@runtime_checkable
class AuditSink(Protocol):
    async def record(self, event: AuditEvent) -> None: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class ApprovalHandler(Protocol):
    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...

    async def close(self) -> None: ...


__all__ = [
    "ApprovalHandler",
    "AuditSink",
    "EmbeddingProvider",
    "ModelGateway",
    "StorageBackend",
]
