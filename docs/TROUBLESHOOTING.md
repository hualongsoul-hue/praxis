# 故障排查

## 缺少模型凭据

症状：`AuthenticationError` 或 `praxis doctor` 报告缺少 `PRAXIS_MODEL_API_KEY`。

确认环境变量设置在实际服务进程，而不只是当前交互终端。不要把密钥加入 `config.yaml`。

## 配置校验失败

运行：

```shell
praxis config validate config.yaml
```

未知字段不会被忽略。根据错误路径修正拼写、类型或范围；不要通过删除校验绕过问题。

## 金额预算下价格未知

启用 `max_budget` 且模型价格无法识别时，网关会失败关闭。为部署配置可信的
`input_cost_per_token` 和 `output_cost_per_token`，或只使用独立 `max_total_tokens` 上限。

## 文件或 Shell 被拒绝

空 `tools.allowed_paths` 会拒绝文件访问，`shell_enabled` 默认是 `false`。确认绝对路径位于授权根目录，
并提供异步审批处理器。不要为了排错全局开放文件系统或私网访问。

## 视觉验证不可用

需要同时满足：安装 `praxis[visual]`、执行 `playwright install chromium`、配置
`verification.visual_enabled: true`，并让默认模型部署声明 `supports_vision: true`。

## MCP 无法启动

确认安装 `praxis[mcp]`、stdio 命令使用绝对或可解析路径、子进程以 stdout 仅输出 MCP 协议消息。
诊断信息应写 stderr。关闭 Session/Runtime 后连接必须断开；出现残留子进程时检查宿主是否跳过了
异步上下文管理器。

## Runtime 为 DEGRADED

查看 `await runtime.health()` 的组件详情。默认本地词法检索会使 embedding 显示降级，这是明确的
能力状态，不代表核心模型或存储失败。根据服务 readiness 策略决定是否接流量。

## 检查点无法恢复

损坏、校验和不一致或旧 schema 会返回明确异常。Praxis 1.0 不自动迁移旧检查点；从新会话开始，
或在外部编写经过审计的迁移程序。
