# 生产部署

Praxis 不启动 Web 框架。宿主服务负责 HTTP/gRPC/队列入口、认证、限流、进程信号和根日志；
Runtime 负责 Agent 资源生命周期。

## 长期运行服务骨架

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from praxis import PraxisRuntime, load_config


@asynccontextmanager
async def agent_runtime() -> AsyncIterator[PraxisRuntime]:
    runtime = PraxisRuntime(load_config("config.yaml"))
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.close()
```

每个外部业务会话应创建自己的 `AgentSession`。不要跨并发请求共享同一个 Session；需要共享的是
Runtime。进程收到停止信号后，先停止接收请求，再等待或取消请求，最后调用 `runtime.close()`。

## 健康端点映射

- readiness：`health.status != FAILED`，并根据业务决定是否接受 `DEGRADED`。
- liveness：事件循环与服务进程仍可响应；不要用计费模型调用做 liveness。
- diagnostics：返回组件状态，但不要返回配置全文、Prompt、工具完整参数或异常中的凭据。

## Windows

- 使用服务管理器（Windows Service、NSSM 或受管容器）保持进程存活。
- 工作路径使用绝对路径；服务账号仅授予授权目录权限。
- Shell 超时会终止整个进程树，但不构成安全隔离。

## Linux

- 可用 systemd、容器编排或进程管理器运行。
- 使用非 root 用户、只读根文件系统和单独的可写数据卷。
- 不可信 Shell 使用独立容器、seccomp/AppArmor/SELinux 和资源限制。

## 可观测与数据

SDK 不覆盖宿主的 OpenTelemetry Provider。生产审计应使用持久化后端并设置备份、保留和访问控制。
日志、审计和追踪默认不记录 Prompt、密钥或完整敏感参数。SQLite 适合单机服务；多实例部署应选
Redis 或自定义 `StorageBackend`，并在部署层验证一致性需求。

## 升级

1. 在隔离环境安装 wheel 并运行 CLI/SDK 冒烟。
2. 验证配置、依赖漏洞、检查点 schema 和可选依赖。
3. 先发布 canary，再逐步切流。
4. Praxis 1.0 不迁移旧公开 API 或旧检查点；不兼容数据必须显式处理。
