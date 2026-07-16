# Changelog

本项目遵循 Semantic Versioning。

## [1.0.0] - 2026-07-16

首个生产级 SDK/CLI 版本。

### 核心

- 新增应用级 `PraxisRuntime`、并发安全 `AgentSession`、异步事件流和组件健康检查。
- 统一模型网关、Token/金额预算预留、并发限制、结构化错误和流式取消语义。
- 统一聊天、总结、判断、记忆、验证和 MCP Sampling 的默认模型别名。
- 新增隔离子代理会话、显式工具子集、并发限制、取消、超时和结果聚合。

### 安全与可靠性

- 严格、冻结且无进程全局状态的配置；模型密钥仅从环境变量读取。
- 文件路径边界、原子写入、检查点版本/校验和、append-only 审计和优雅关闭。
- Shell 默认禁用，网络默认阻止 SSRF，异步审批失败关闭，写工具不自动重试。
- Redis、MCP、视觉和 OTLP 拆分为懒加载可选依赖。

### 工程

- 支持 Python 3.12、3.13、3.14 以及 Windows/Linux。
- 提供 `praxis config validate/show`、`doctor`、`chat` 和 `version`。
- 提供 strict Pyright 类型、`py.typed`、跨平台 CI、安全扫描、wheel 干净环境冒烟和文档示例校验。
- 重写架构、配置、安全、部署、扩展和故障排查文档。

此版本不兼容旧公开 API 或旧检查点，不提供自动迁移层。
