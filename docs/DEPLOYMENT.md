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

`runtime.health()` 的模型检查是真实请求，并受 `gateway.health_probe_timeout` 和
`gateway.health_probe_ttl` 控制。服务自身的 liveness 应使用本地事件循环检查；readiness 可复用
缓存后的 Runtime 健康报告。CLI 的 `praxis doctor` 也会触发该探针，可能产生一次小额调用。

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

CLI 与完整控制台通过同一个 `PraxisCliApplication` 拥有本次进程创建的 Runtime、指标导出器和
追踪 Provider，并按依赖顺序关闭。嵌入式 SDK 不配置根日志，也不启动永久指标 HTTP 线程。

指定模型地址使用明文 HTTP，只适用于端点与 Agent 位于同一受信隔离网络的部署。若流量经过
任何不受信链路，必须使用 HTTPS/mTLS 代理或等效的加密服务网格；否则密钥、Prompt、附件和模型
响应都可能被旁路观察或篡改。

## 升级

1. 在隔离环境安装 wheel 并运行 CLI/SDK 冒烟。
2. 验证配置、依赖漏洞、检查点 schema 和可选依赖。
3. 先发布 canary，再逐步切流。
4. Praxis 1.0 不迁移旧公开 API 或旧检查点；不兼容数据必须显式处理。
