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
        image: false
        audio: false
        video: false
        file: false
  default_model: default
  max_concurrent_requests: 8
  max_total_tokens: 1000000
```

`api_key_env` 只能是 `PRAXIS_MODEL_API_KEY`。密钥本身不能出现在 YAML 中。所有聊天、总结、判断、
记忆处理、验证和 MCP Sampling 都使用 `default_model`；MCP Server 的模型 hint 不能覆盖它。

启用 `max_budget` 时，每个部署必须提供可信的显式价格，或由 LiteLLM 返回可识别价格；价格未知时
调用失败关闭。`max_total_tokens` 独立生效，不能通过零价格绕过。

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
