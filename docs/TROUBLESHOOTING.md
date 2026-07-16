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

## 多模态输入在本地被拒绝

`UnsupportedInputModalityError` 表示所选部署的 `capabilities` 未声明该模态；这与 SDK 是否能构造
内容块无关。只有在“文本控制成功 → 模态调用成功 → 文本控制成功”的真实端点健康窗口中重复验证
后，才把对应字段设为 `true`。

`InputPathError` 通常表示本地附件不在 `inputs.allowed_paths` 内，或打开后发现符号链接/重解析点
逃逸。`InputSizeLimitError` 由单附件或整轮总字节上限触发。URL 输入还要求
`inputs.remote_enabled: true`；不要为绕过 SSRF 防护而开启私网访问。

若文本控制也返回 `ProviderUnavailableError`，当前窗口只能说明端点整体不可用，不能据此判断某种
模态不受支持。等待端点恢复并重新执行 live-model 套件，不要将测试改成“成功或任意错误都通过”。

## 视觉验证不可用

需要同时满足：安装 `praxis[visual]`、执行 `playwright install chromium`、配置
`verification.visual_enabled: true`，并让默认模型部署声明 `capabilities.image: true`。

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
