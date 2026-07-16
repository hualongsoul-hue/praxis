# 扩展 SDK

Praxis 的外部边界使用 Protocol。自定义实现应保证异步关闭幂等、取消安全，并且不修改宿主全局状态。

## 审批处理器

```python
from praxis.models.tools import ApprovalDecision, ApprovalRequest


class ServiceApprovalHandler:
    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        decision = await lookup_external_decision(request)
        return ApprovalDecision(approved=decision)
```

`lookup_external_decision` 由宿主实现。没有处理器、超时或异常时 Praxis 会拒绝操作。

## 自定义工具

工具需要类型化定义与异步 handler。只有只读或声明为幂等的工具可以自动重试；写工具必须由业务方
提供幂等键或保持不重试。

```python
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.registry import ToolRegistry


async def lookup(arguments: dict[str, object]) -> str:
    return str(arguments.get("query", ""))


registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="lookup",
        description="查询受控数据源",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        metadata=ToolMetadata(readonly=True, idempotent=True),
    ),
    lookup,
)
```

## 存储、审计和嵌入

实现 `StorageBackend`、`AuditSink` 或 `EmbeddingProvider` Protocol 后，通过 `PraxisRuntime` 构造参数
注入。存储键必须保持 namespace 隔离；审计事件只能创建不能覆盖；嵌入失败时应返回错误或由宿主
明确选择本地词法降级。

## MCP

安装 `praxis[mcp]` 后可使用 stdio 或 Streamable HTTP 传输。Sampling 始终路由到 Runtime 默认
模型，Elicitation 在无处理器时拒绝。测试应使用本地 Python stdio Server 覆盖工具、资源、Prompt、
Sampling、Elicitation、断开和关闭，不依赖公网服务。

## 服务框架集成

在 ASGI lifespan、队列 worker 启停钩子或自定义守护进程中创建一个 Runtime。公开接口只依赖
`start()`、`session()`、`health()` 和 `close()`，因此无需让 Praxis 依赖宿主 Web 框架。
