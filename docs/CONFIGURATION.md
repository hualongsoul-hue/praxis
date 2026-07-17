# 配置参考

配置加载顺序为 YAML、`PRAXIS_<SECTION>__<FIELD>` 环境变量、显式 Python 覆盖。所有模型都使用
`extra="forbid"`，未知字段、错误类型和越界数值会直接失败。加载结果是不可变快照。

## 模型部署

```yaml
gateway:
  deployments:
    - model_name: default
      model: openai/glm-5.1-openai
      api_base: http://172.24.23.192:3000/v1
      api_key_env: PRAXIS_MODEL_API_KEY
      default_max_output_tokens: 4096
      capabilities:
        image: true
        audio: false
        video: true
        file: false
  default_model: default
  max_concurrent_requests: 8
  max_total_tokens: 1000000
```

`api_key_env` 只能是 `PRAXIS_MODEL_API_KEY`。密钥本身不能出现在 YAML 中。所有聊天、总结、判断、
记忆处理、验证和 MCP Sampling 都使用 `default_model`；MCP Server 的模型 hint 不能覆盖它。

`capabilities` 描述当前端点已验证的能力，不是 SDK 支持列表。Praxis 的输入传输层支持图片、音频、
视频和文件；但只有在至少三个独立真实健康窗口中稳定成功的端点模态才能配置为 `true`。`false`
会在任何文件或网络 I/O 之前返回 `UnsupportedInputModalityError`，避免把未验证能力交给远端碰运气。

启用 `max_budget` 时，每个部署必须提供可信的显式价格，或由 LiteLLM 返回可识别价格；价格未知时
调用失败关闭。`max_total_tokens` 独立生效，不能通过零价格绕过。

## 用户输入

```yaml
inputs:
  allowed_paths:
    - ./attachments
  remote_enabled: false
  allow_private_networks: false
  max_attachment_bytes: 20000000
  max_total_bytes: 50000000
  max_redirects: 5
  remote_timeout: 30.0
```

`allowed_paths` 为空时拒绝所有本地附件路径。Windows 可填写例如
`D:\\AgentData\\attachments`，Linux 可填写 `/srv/praxis/attachments`。每个路径都按打开后的
文件句柄再次校验，符号链接或重解析点不能逃逸授权根目录。

URL 来源必须显式设置 `remote_enabled: true`。默认仍拒绝 URL 凭据、回环、私网、链路本地和
非全局地址，并逐跳校验重定向。`max_attachment_bytes` 限制单附件；`max_total_bytes` 在读取或编码
下一附件前限制整轮累计大小。

## 存储

```yaml
persistence:
  backend: sqlite
  sqlite_path: data/praxis.db
```

可选后端：

```yaml
persistence:
  backend: filesystem
  filesystem_path: data/storage
```

```yaml
persistence:
  backend: redis
  redis_url: redis://redis.example.internal:6379/0
```

Redis 需要安装 `praxis[redis]`。生产环境应使用专用数据库、认证、TLS 和备份策略。

## 工具边界

```yaml
tools:
  allowed_paths:
    - /srv/praxis/workspace
  shell_enabled: false
  network_allowed: false
  allow_private_networks: false
  network_max_response_bytes: 1000000
  approval_timeout: 60
```

Windows 可使用绝对路径（例如 `D:\\AgentData\\workspace`）。空 `allowed_paths` 拒绝所有文件访问。

## 可选能力

```yaml
verification:
  computational_enabled: true
  inferential_enabled: true
  visual_enabled: false

mcp:
  enabled: false
  connect_timeout: 30
  sampling_enabled: true
```

视觉验证必须同时安装 `praxis[visual]`、安装 Playwright 浏览器，并由默认部署声明
`capabilities.image: true`。未满足条件时返回明确不可用状态。

完整、可直接校验的示例见 [config.example.yaml](../config.example.yaml)。
