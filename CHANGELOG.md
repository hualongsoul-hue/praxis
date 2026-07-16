# Changelog

本项目遵循 Semantic Versioning。

## [1.0.0] - 2026-07-16

首个生产级 SDK/CLI 版本。

### 核心

- 新增应用级 `PraxisRuntime`、并发安全 `AgentSession`、异步事件流和组件健康检查。
- 新增公开、强类型的 `UserInput`、`ImageInput`、`AudioInput`、`VideoInput` 和
  `FileInput`，流式与非流式 SDK 路径共享同一多模态内容契约。
- 统一模型网关、Token/金额预算预留、并发限制、结构化错误和流式取消语义。
- 统一聊天、总结、判断、记忆、验证和 MCP Sampling 的默认模型别名。
- 新增隔离子代理会话、显式工具子集、并发限制、取消、超时和结果聚合。

### 安全与可靠性

- 严格、冻结且无进程全局状态的配置；模型密钥仅从环境变量读取。
- 文件路径边界、原子写入、检查点版本/校验和、append-only 审计和优雅关闭。
- Shell 默认禁用，网络默认阻止 SSRF，异步审批失败关闭，写工具不自动重试。
- 本地附件默认无授权路径，远程附件默认禁用；路径穿越、符号链接逃逸、私网目标、
  MIME 与大小限制均在发送前失败关闭，异常和审计不保留附件正文或 Base64。
- Redis、MCP、视觉和 OTLP 拆分为懒加载可选依赖。

### 工程

- 支持 Python 3.12、3.13、3.14 以及 Windows/Linux。
- 提供 `praxis config validate/show`、`doctor`、`chat` 和 `version`。
- 新增仓库级公开符号 AST 策略，项目自有 Python 代码禁止单下划线私有定义与访问。
- 将内存替身场景归类到 `tests/scenarios`；`tests/e2e` 仅保留经过真实 HTTP/TCP
  边界的 Runtime → LiteLLM 契约测试，并独立校验集成与文档示例。
- 指定真实端点的多模态能力保持未分类且示例能力开关保持关闭：最近的健康窗口在纯文本
  控制请求阶段返回 HTTP 503 / `no_available_workers`，未把非健康窗口当作能力证据。
- 提供 strict Pyright 类型、`py.typed`、跨平台 CI、安全扫描、wheel 干净环境冒烟和文档示例校验。
- 发布门禁通过 837 项测试（12 项显式环境跳过）、91.14% 分支覆盖、Ruff、Pyright、
  pip-audit、wheel/sdist 构建，以及无密钥、无 Redis/MCP/视觉依赖的 wheel-only 冒烟。
- 重写架构、配置、安全、部署、扩展和故障排查文档。

此版本不兼容旧公开 API 或旧检查点，不提供自动迁移层。
