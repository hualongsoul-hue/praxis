# Praxis 配置参考

所有配置项均有默认值，支持零配置启动。配置通过 YAML 文件或环境变量提供。

---

## 配置文件格式

```yaml
# config.yaml
telemetry:
  log_level: "INFO"
  # ...

persistence:
  backend: "sqlite"
  # ...

gateway:
  default_model: "default"
  # ...

tools:
  default_timeout: 30.0
  # ...

memory:
  max_memories: 10000
  # ...

context:
  compaction_threshold: 0.8
  # ...

guardrails:
  default_permission: "confirm"
  # ...

recovery:
  max_retries: 3
  # ...

orchestrator:
  max_turns: 100
  # ...

session:
  auto_checkpoint: true
  # ...

subagent:
  max_concurrent: 5
  # ...

skills:
  auto_discover: true
  # ...
```

---

## S1 配置系统

配置加载优先级：环境变量 > 配置文件 > 默认值。

---

## S2 遥测配置 (`TelemetryConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `log_level` | `str` | `"INFO"` | 全局日志级别 |
| `log_format` | `"json" \| "text"` | `"text"` | 日志格式 |
| `log_levels` | `dict[str, str]` | `{}` | 按组件独立日志级别 |
| `metrics_enabled` | `bool` | `true` | 是否启用指标收集 |
| `metrics_export` | `"prometheus" \| "file"` | `"file"` | 指标导出方式 |
| `metrics_file` | `str \| null` | `null` | 指标文件路径 |
| `tracing_enabled` | `bool` | `true` | 是否启用链路追踪 |
| `tracing_export` | `"otlp" \| "console"` | `"console"` | 追踪导出方式 |
| `audit_enabled` | `bool` | `true` | 是否启用审计日志 |

---

## S3 持久化配置 (`PersistenceConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `backend` | `"sqlite" \| "redis" \| "filesystem"` | `"sqlite"` | 存储后端 |
| `sqlite_path` | `str` | `"data/praxis.db"` | SQLite 数据库路径 |
| `redis_url` | `str \| null` | `null` | Redis 连接 URL |
| `filesystem_path` | `str` | `"data/storage"` | 文件系统存储路径 |

---

## S4 模型网关配置 (`GatewayConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_list` | `list[dict]` | `[]` | LiteLLM Router 模型列表 |
| `default_model` | `str` | `"default"` | 默认模型名 |
| `routing_strategy` | `str` | `"simple-shuffle"` | 路由策略 |
| `num_retries` | `int` | `3` | LLM 调用重试次数 |
| `timeout` | `float` | `60.0` | LLM 调用超时（秒） |
| `max_budget` | `float \| null` | `null` | 成本预算上限（USD） |

### model_list 格式

```yaml
gateway:
  model_list:
    - model_name: "default"
      litellm_params:
        model: "openai/gpt-4o"
        api_key: "sk-..."
    - model_name: "fast"
      litellm_params:
        model: "anthropic/claude-3-5-haiku-20241022"
        api_key: "sk-ant-..."
```

---

## S5 工具系统配置 (`ToolsConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `allowed_paths` | `list[str]` | `[]` | 沙箱文件系统白名单（空=允许所有） |
| `default_timeout` | `float` | `30.0` | 默认工具执行超时（秒） |
| `max_concurrent_readonly` | `int` | `5` | 最大并发只读工具数 |
| `shell_timeout` | `float` | `120.0` | Shell 命令超时（秒） |
| `network_allowed` | `bool` | `true` | 是否允许网络出站 |

---

## S6 记忆系统配置 (`MemoryConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `vector_dimensions` | `int` | `1536` | 向量维度（匹配 embedding 模型） |
| `background_batch_threshold` | `int` | `5` | 后台批量处理阈值 |
| `background_interval_seconds` | `float` | `10.0` | 后台处理间隔（秒） |
| `dream_min_hours` | `float` | `24.0` | 离线整合最小间隔（小时） |
| `dream_min_sessions` | `int` | `5` | 离线整合最小会话数 |
| `max_memories` | `int` | `10000` | 最大记忆条目数 |
| `decay_enabled` | `bool` | `true` | 是否启用记忆衰减 |

---

## S7 上下文引擎配置 (`ContextConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `compaction_threshold` | `float` | `0.8` | Token 占比超过此值触发压缩 |
| `masking_turn_distance` | `int` | `10` | 注意力遮蔽的轮次距离 |
| `masking_token_threshold` | `int` | `2000` | 注意力遮蔽的 Token 阈值 |
| `recent_file_refs_keep` | `int` | `5` | 保留的近期文件引用数 |

> **注意**: 模型最大上下文窗口由 `litellm.get_max_tokens(model)` 动态获取，不在配置中硬编码。

---

## S8 护栏系统配置 (`GuardrailsConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `default_permission` | `"auto_approve" \| "confirm" \| "deny"` | `"confirm"` | 默认工具权限 |
| `permissions_file` | `str \| null` | `null` | 权限配置文件路径 |
| `input_guardrails_enabled` | `bool` | `true` | 是否启用输入护栏 |
| `output_guardrails_enabled` | `bool` | `true` | 是否启用输出护栏 |

---

## S9 错误恢复配置 (`RecoveryConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_retries` | `int` | `3` | 工具最大重试次数 |
| `base_delay` | `float` | `1.0` | 重试基础延迟（秒） |
| `max_delay` | `float` | `30.0` | 重试最大延迟（秒） |
| `circuit_breaker_threshold` | `int` | `5` | 熔断器失败阈值 |
| `circuit_breaker_cooldown` | `float` | `60.0` | 熔断器冷却时间（秒） |

---

## S10 验证引擎配置 (`VerificationConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `computational_enabled` | `bool` | `true` | 启用计算验证 |
| `inferential_enabled` | `bool` | `true` | 启用推理验证 |
| `visual_enabled` | `bool` | `false` | 启用视觉验证 |

---

## S11 编排循环配置 (`OrchestratorConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_turns` | `int` | `100` | 单轮对话最大轮次 |
| `default_strategy` | `"react" \| "plan-and-execute"` | `"react"` | 默认编排策略 |
| `stream_events` | `bool` | `true` | 是否流式推送编排事件 |

---

## S12 会话管理配置 (`SessionConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `auto_checkpoint` | `bool` | `true` | 自动检查点 |
| `max_checkpoints_per_session` | `int` | `50` | 每会话最大检查点数 |

---

## S13 子代理配置 (`SubagentConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_concurrent` | `int` | `5` | 最大并发子代理数 |
| `default_max_turns` | `int` | `50` | 子代理默认最大轮次 |
| `default_timeout` | `float` | `300.0` | 子代理默认超时（秒） |

---

## S14 技能系统配置 (`SkillsConfig`)

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `skill_paths` | `list[str]` | `[".praxis/skills", "~/.praxis/skills"]` | 技能搜索路径 |
| `auto_discover` | `bool` | `true` | 自动发现技能 |
| `max_skills_in_context` | `int` | `10` | 上下文中最大技能数 |
