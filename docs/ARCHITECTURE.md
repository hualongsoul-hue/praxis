# 架构与生命周期

## 公共边界

应用只需要持有一个 `PraxisRuntime`，并按请求或业务会话创建 `AgentSession`：

```text
host service
└── PraxisRuntime
    ├── ModelGateway / StorageBackend / AuditSink / telemetry
    ├── TaskSupervisor / MCP transports
    └── AgentSession (0..n)
        ├── orchestration state machine
        ├── isolated memory and context
        └── Runtime-owned subagent sessions (0..n)
```

Runtime 的配置是深拷贝、冻结的快照。多个 Runtime 可以在同一进程共存，不共享可变配置、
审计、指标或生命周期状态。模型网关和存储可由同一 Runtime 下的不同 Session 安全共享。

## 会话状态机

流式与非流式入口复用同一条编排路径：

```text
input guardrail -> prompt -> model -> parse -> approve/execute tools
                -> verify -> checkpoint -> output guardrail -> response
```

会话状态为 `INITIALIZING`、`ACTIVE`、`WAITING_APPROVAL` 或 `TERMINATED`。同一个
`AgentSession` 只能执行一个轮次，不同 Session 可以并行。取消会沿模型流、工具进程、子代理和
后台任务传播；关闭会先阻止新工作，再终止会话与子任务，最后冲刷审计并关闭外部资源。

## 资源所有权

- Runtime：网关、存储、审计、指标、任务监督器和注入的 Provider。
- AgentSession：编排器、上下文、记忆 Worker、检查点和会话级 MCP 退出栈。
- 子代理：由 Runtime 创建和登记，只继承冻结配置与显式工具子集；不能递归获得子代理工具。
- 宿主：根日志配置、全局 OpenTelemetry Provider、Web 服务器和进程信号策略。

`start()`、`close()` 和各组件关闭接口均幂等。不要绕过上下文管理器长期持有未关闭的会话。

## 持久化与恢复

文件、SQLite 和 Redis 后端实现同一存储协议。检查点带 schema version 与校验和；不支持的旧版本
或损坏数据会返回明确异常，不会静默恢复。文件后端对 namespace/key 编码、路径边界和原子替换
进行校验，SQLite 使用 WAL 并串行化关键事务。

## 健康状态

`runtime.health()` 返回 `READY`、`DEGRADED` 或 `FAILED`，并逐项报告模型、存储、嵌入、视觉和
后台任务。未配置远程 Embedding 时使用确定性本地词法检索并标记降级；禁用的可选能力不会伪装
成已连接的外部服务。
