"""Praxis 全局异常体系。

异常层次结构：
    PraxisError                         # 框架基础异常
    ├── ConfigError                     # S1 配置系统
    ├── TelemetryError                  # S2 遥测系统
    ├── PersistenceError                # S3 持久化引擎
    ├── GatewayError                    # S4 模型网关
    │   ├── AuthenticationError         #   Provider 认证失败
    │   ├── RateLimitError              #   Provider 速率限制
    │   ├── ModelNotFoundError          #   模型不存在
    │   ├── ContextWindowExceededError  #   上下文窗口溢出
    │   ├── BudgetExceededError         #   成本预算耗尽
    │   ├── ProviderUnavailableError    #   Provider 不可用
    │   └── GatewayTimeoutError         #   请求超时
    ├── ToolError                       # S5 工具系统
    │   ├── ToolNotFoundError           #   工具未注册
    │   ├── ToolExecutionError          #   工具执行失败
    │   ├── ToolTimeoutError            #   工具执行超时
    │   └── ToolPolicyViolationError    #   工具授权策略违反
    ├── CognitiveMemoryError            # S6 认知记忆系统
    ├── ContextError                    # S7 上下文引擎
    ├── GuardrailError                  # S8 护栏系统
    ├── RecoveryError                   # S9 错误恢复
    ├── VerificationError               # S10 验证引擎
    ├── OrchestrationError              # S11 编排循环
    ├── SessionError                    # S12 会话管理
    ├── SubagentError                   # S13 子代理协调
    └── SkillError                      # S14 技能系统
"""

import json as json_library
from typing import Any

# ── 基础异常 ─────────────────────────────────────────────────────────────────


class PraxisError(Exception):
    """Praxis 框架基础异常。所有 Praxis 异常的根类。"""

    component: str = "praxis"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ModelValidationError(PraxisError, ValueError):
    """Input-free replacement for Pydantic validation errors at SDK boundaries."""

    component = "validation"

    def errors(self, *args: Any, **options: Any) -> list[dict[str, str]]:
        """Return stable structured details without echoing rejected values."""

        return [{"type": "model_validation", "msg": "模型输入验证失败"}]

    def json(self, *args: Any, **options: Any) -> str:
        """Serialize only the stable input-free error representation."""

        return json_library.dumps(self.errors(), ensure_ascii=False)


# ── S1 配置系统 ──────────────────────────────────────────────────────────────


class ConfigError(PraxisError):
    """配置加载、验证或热更新错误。"""

    component = "config"


class InputError(PraxisError):
    """Typed input validation or resolution error."""

    component = "input"


class InvalidInputSourceError(InputError):
    """Attachment source does not match its declared source kind."""


class InputPathError(InputError):
    """Attachment path is unavailable or outside configured roots."""


class InputMediaTypeError(InputError):
    """Attachment media type is absent, invalid, or disallowed."""


class InputSizeLimitError(InputError):
    """Attachment bytes exceed a configured size limit."""


class InputNetworkError(InputError):
    """Remote attachment retrieval failed its security policy."""


class UnsupportedInputModalityError(InputError):
    """The selected deployment cannot accept an attachment modality."""


# ── S2 遥测系统 ──────────────────────────────────────────────────────────────


class TelemetryError(PraxisError):
    """日志、指标、追踪或审计错误。"""

    component = "telemetry"


# ── S3 持久化引擎 ────────────────────────────────────────────────────────────


class PersistenceError(PraxisError):
    """存储读写或检查点操作错误。"""

    component = "persistence"


class CheckpointCorruptionError(PersistenceError):
    """检查点校验和或结构损坏。"""


class CheckpointVersionError(PersistenceError):
    """检查点 schema 版本不受支持。"""


# ── S4 模型网关 ──────────────────────────────────────────────────────────────


class GatewayError(PraxisError):
    """模型网关通用错误。"""

    component = "gateway"


class AuthenticationError(GatewayError):
    """Provider API 认证失败（API Key 无效或过期）。"""


class RateLimitError(GatewayError):
    """Provider API 速率限制。"""


class ModelNotFoundError(GatewayError):
    """请求的模型不存在或不可用。"""


class ContextWindowExceededError(GatewayError):
    """输入 Token 超过模型上下文窗口限制。"""


class BudgetExceededError(GatewayError):
    """Token 或成本预算耗尽。"""


class ProviderUnavailableError(GatewayError):
    """Provider 服务不可用（所有部署均故障）。"""


class GatewayTimeoutError(GatewayError):
    """LLM 请求超时。"""


# ── S5 工具系统 ──────────────────────────────────────────────────────────────


class ToolError(PraxisError):
    """工具系统通用错误。"""

    component = "tools"


class ToolNotFoundError(ToolError):
    """请求的工具未在注册表中。"""


class ToolExecutionError(ToolError):
    """工具执行过程中发生错误。"""


class ToolTimeoutError(ToolError):
    """工具执行超过配置的超时时间。"""


class ToolPolicyViolationError(ToolError):
    """工具执行违反显式授权策略。"""


# ── S6 记忆系统 ──────────────────────────────────────────────────────────────


class CognitiveMemoryError(PraxisError):
    """认知记忆系统错误（避免遮蔽内建 MemoryError）。"""

    component = "memory"


# ── S7 上下文引擎 ────────────────────────────────────────────────────────────


class ContextError(PraxisError):
    """上下文组装或压缩错误。"""

    component = "context"


# ── S8 护栏系统 ──────────────────────────────────────────────────────────────


class GuardrailError(PraxisError):
    """护栏规则评估或权限检查错误。"""

    component = "guardrails"


# ── S9 错误恢复 ──────────────────────────────────────────────────────────────


class RecoveryError(PraxisError):
    """错误恢复策略执行错误。"""

    component = "recovery"


# ── S10 验证引擎 ─────────────────────────────────────────────────────────────


class VerificationError(PraxisError):
    """验证执行错误。"""

    component = "verification"


# ── S11 编排循环 ─────────────────────────────────────────────────────────────


class OrchestrationError(PraxisError):
    """编排循环执行错误。"""

    component = "orchestrator"


# ── S12 会话管理 ───────────────────────────────────────────────────────────


class SessionError(PraxisError):
    """会话管理错误。"""

    component = "session"


class ConcurrentSessionRunError(SessionError):
    """同一会话已在执行另一个轮次。"""


class RuntimeStateError(PraxisError):
    """Runtime 生命周期状态不允许当前操作。"""

    component = "runtime"


class RuntimeCloseError(PraxisError):
    """One or more resources failed to close after all cleanup was attempted."""

    component = "runtime"

    def __init__(self, failures: tuple[BaseException, ...]) -> None:
        self.failures = failures
        super().__init__(
            f"Runtime 关闭期间发生 {len(failures)} 个错误",
            details={"failure_types": [type(item).__name__ for item in failures]},
        )


# ── S13 子代理协调 ───────────────────────────────────────────────────────────


class SubagentError(PraxisError):
    """子代理创建或协调错误。"""

    component = "subagent"


# ── S14 技能系统 ─────────────────────────────────────────────────────────────


class SkillError(PraxisError):
    """技能发现、加载或执行错误。"""

    component = "skills"
