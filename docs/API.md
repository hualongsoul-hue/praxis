# Praxis API 参考

## 目录

- [核心会话 API](#核心会话-api)
- [编排循环 API](#编排循环-api)
- [工具系统 API](#工具系统-api)
- [护栏系统 API](#护栏系统-api)
- [记忆系统 API](#记忆系统-api)
- [上下文引擎 API](#上下文引擎-api)
- [网关 API](#网关-api)
- [持久化 API](#持久化-api)
- [错误恢复 API](#错误恢复-api)
- [子代理 API](#子代理-api)
- [技能系统 API](#技能系统-api)
- [MCP 集成 API](#mcp-集成-api)
- [遥测 API](#遥测-api)
- [数据模型索引](#数据模型索引)

---

## 核心会话 API

### `SessionFactory`

**模块**: `praxis.session.core`

创建完整配线的会话实例。

```python
from praxis.session.core import SessionFactory
from praxis.config.schemas import (
    SessionConfig, OrchestratorConfig, ContextConfig,
)

factory = SessionFactory(
    store=persistence_store,
    session_config=SessionConfig(),
    orchestrator_config=OrchestratorConfig(),
    context_config=ContextConfig(),
)
session = factory.create_session(
    guardrails=guardrail_engine,
    registry=tool_registry,   # 可选，默认创建空注册表
    model="openai/gpt-4o",    # 可选，默认 "default"
)
```

### `Session`

**模块**: `praxis.session.core`

| 方法 | 说明 | 返回 |
|---|---|---|
| `run_turn(user_message)` | 执行一轮完整对话 | `AgentResponse` |
| `save_auto_checkpoint()` | 保存自动检查点 | `str \| None`（检查点 ID） |
| `terminate(reason)` | 终止会话 | `None` |

**属性**: `session_id`, `status`, `metadata`, `loop`, `assembler`, `registry`, `store`

### `CheckpointManager`

**模块**: `praxis.session.checkpoint`

| 方法 | 说明 |
|---|---|
| `save_checkpoint(session_id, state)` | 保存检查点 |
| `load_checkpoint(session_id, checkpoint_id)` | 加载指定检查点 |
| `list_checkpoints(session_id)` | 列出会话所有检查点 |

### `SessionResumer`

**模块**: `praxis.session.resume`

| 方法 | 说明 |
|---|---|
| `resume_session(session_id, guardrails)` | 从最新检查点恢复会话 |

---

## 编排循环 API

### `OrchestrationLoop`

**模块**: `praxis.orchestrator.loop`

编排 Agent 单轮生命周期：Prompt 组装 → LLM 调用 → 工具执行 → 上下文更新 → 终止检查。

| 方法 | 说明 | 返回 |
|---|---|---|
| `run(user_message)` | 执行编排循环 | `AgentResponse` |

**关联组件**: `ToolCoordinator`, `TerminationManager`, `OutputParser`, `EventEmitter`, `LoopStrategy`

### `TerminationManager`

**模块**: `praxis.orchestrator.termination`

| 方法 | 说明 |
|---|---|
| `should_terminate(state)` | 检查是否应终止循环 |

**终止原因枚举** (`TerminationReason`):
- `NATURAL` — LLM 自然停止
- `MAX_TURNS` — 达到最大轮次
- `GUARDRAIL_TRIPWIRE` — 护栏绊线触发
- `USER_CANCEL` — 用户取消
- `ERROR` — 不可恢复错误

### `EventEmitter`

**模块**: `praxis.orchestrator.events`

| 方法 | 说明 |
|---|---|
| `emit(event_type, data)` | 发射编排事件 |

**事件类型**: `turn_start`, `llm_request`, `llm_response`, `tool_call`, `tool_result`, `termination`

---

## 工具系统 API

### `ToolRegistry`

**模块**: `praxis.tools.registry`

| 方法 | 说明 |
|---|---|
| `register(definition, handler)` | 注册工具 |
| `unregister(name)` | 注销工具 |
| `get_entry(name)` | 获取工具条目 |
| `has_tool(name)` | 检查工具是否存在 |
| `list_tools()` | 列出所有工具名 |
| `get_tool_schemas(category, tags)` | 导出 OpenAI function calling 格式 Schema |

### `ToolExecutor`

**模块**: `praxis.tools.executor`

| 方法 | 说明 | 返回 |
|---|---|---|
| `execute(name, arguments, call_id)` | 执行工具 | `ToolResult` |

### `Sandbox`

**模块**: `praxis.tools.sandbox`

| 方法 | 说明 |
|---|---|
| `check_path(path)` | 检查路径是否在沙箱白名单内 |
| `check_network()` | 检查网络出站是否允许 |

---

## 护栏系统 API

### `GuardrailEngine`

**模块**: `praxis.guardrails.engine`

| 方法 | 说明 | 返回 |
|---|---|---|
| `check_input(user_message)` | 输入护栏检查 | `GuardrailVerdict` |
| `check_tool_call(name, args, metadata)` | 工具调用护栏检查 | `GuardrailVerdict` |
| `check_output(response)` | 输出护栏检查 | `GuardrailVerdict` |
| `register_rule(rule)` | 注册自定义规则 | `None` |

### `RuleEngine`

**模块**: `praxis.guardrails.rules`

| 方法 | 说明 |
|---|---|
| `register_rule(rule)` | 注册规则 |
| `register_builtin_rules()` | 注册内置规则集 |
| `evaluate(target, content, context)` | 评估规则 |

### `PermissionManager`

**模块**: `praxis.guardrails.permissions`

| 方法 | 说明 |
|---|---|
| `check_permission(tool_name, metadata, arguments)` | 检查工具权限 |
| `grant_temporary(tool_name, permission)` | 临时授权 |
| `revoke_temporary(tool_name)` | 撤销临时授权 |

---

## 记忆系统 API

### `MemoryPipeline`

**模块**: `praxis.memory.pipeline`

| 方法 | 说明 |
|---|---|
| `append_message(message)` | 追加消息到工作记忆 |
| `get_message_history(limit)` | 获取消息历史 |

### `VectorStore`

**模块**: `praxis.memory.vector_store`

长期记忆的向量存储和检索。

### `MemoryRetriever`

**模块**: `praxis.memory.retrieval`

基于语义相似度检索相关记忆。

---

## 上下文引擎 API

### `PromptAssembler`

**模块**: `praxis.context.assembler`

| 方法 | 说明 | 返回 |
|---|---|---|
| `assemble_prompt(turn_context)` | 组装完整 Prompt | `AssembledPrompt` |
| `update_with_response(msg)` | 追加助手响应 | `None` |
| `update_with_result(results)` | 追加工具结果 | `None` |
| `get_token_usage()` | 获取 Token 用量 | `TokenUsage` |
| `set_tool_schemas(schemas)` | 设置工具 Schema | `None` |

### `ToolInjector`

**模块**: `praxis.context.tool_injection`

| 方法 | 说明 |
|---|---|
| `get_tools_for_stage(stage, tags)` | 按阶段获取工具 Schema |

---

## 网关 API

### `GatewayRouter`

**模块**: `praxis.gateway.router`

封装 LiteLLM Router，提供多 Provider 路由和负载均衡。

### `chat(gateway, messages, model, ...)`

**模块**: `praxis.gateway.chat`

统一 LLM 调用入口。

### Token 计量

**模块**: `praxis.gateway.metering`

| 函数 | 说明 |
|---|---|
| `get_max_tokens(model)` | 查询模型最大上下文窗口 |
| `get_token_count(messages, model)` | 预估消息 Token 数量 |

---

## 持久化 API

### `PersistenceStore`

**模块**: `praxis.persistence.store`

| 方法 | 说明 |
|---|---|
| `save(namespace, key, data)` | 保存数据 |
| `load(namespace, key)` | 加载数据 |
| `delete(namespace, key)` | 删除数据 |
| `list_keys(namespace)` | 列出命名空间下的所有键 |
| `close()` | 关闭连接 |

**后端**: SQLite（默认）、Redis、文件系统

---

## 错误恢复 API

### `CircuitBreaker`

**模块**: `praxis.recovery.circuit_breaker`

三态模型：CLOSED → OPEN → HALF_OPEN

| 方法 | 说明 |
|---|---|
| `check()` | 检查当前状态 |
| `record_success()` | 记录成功 |
| `record_failure()` | 记录失败 |

### `RetryPolicy`

**模块**: `praxis.recovery.retry`

| 方法 | 说明 |
|---|---|
| `get_retry_decision(tool_name, attempt)` | 计算重试决策 |

---

## 子代理 API

### `IsolatedContext`

**模块**: `praxis.subagent.isolation`

| 方法 | 说明 |
|---|---|
| `create_isolated_session(spec, guardrails, parent_registry, model)` | 创建隔离子代理会话 |

### `SubagentSpec`

**模型**: `praxis.models.subagent`

| 字段 | 类型 | 说明 |
|---|---|---|
| `subagent_id` | `str` | 子代理 ID |
| `task` | `str` | 任务描述 |
| `mode` | `SubagentMode` | 运行模式 |
| `tool_names` | `list[str]` | 允许的工具名 |
| `max_turns` | `int` | 最大轮次 |

---

## 技能系统 API

### `SkillManager`

**模块**: `praxis.skills.manager`

| 方法 | 说明 |
|---|---|
| `register_skill(definition)` | 注册技能 |
| `load_skill(skill_id)` | 加载技能完整内容 |
| `get_skill_index()` | 获取技能索引（第一层披露） |
| `register_disclosure_tools()` | 注册 LLM 可调用的披露工具 |
| `activate_for_task(task)` | 自动激活匹配任务的技能 |
| `register_version(skill_id, definition)` | 注册新版本 |
| `record_usage(skill_id, action)` | 记录使用统计 |

---

## MCP 集成 API

### `MCPToolsBridge`

**模块**: `praxis.tools.mcp.tools`

| 方法 | 说明 |
|---|---|
| `discover_tools(server_name, session)` | 发现并注册 MCP Server 工具 |
| `call_tool(server_name, tool_name, arguments)` | 代理调用 MCP 工具 |

### `ElicitationManager`

**模块**: `praxis.tools.mcp.elicitation`

| 方法 | 说明 |
|---|---|
| `set_handler(handler)` | 注册用户界面处理器 |
| `handle_elicitation(request)` | 处理 Elicitation 请求 |

### `SamplingManager`

**模块**: `praxis.tools.mcp.sampling`

| 方法 | 说明 |
|---|---|
| `set_review_handler(handler)` | 注册 Human-in-the-loop 审核函数 |
| `handle_sampling(request)` | 处理 Sampling 请求 |

### `MCPConnectionManager`

**模块**: `praxis.tools.mcp.connection`

| 方法 | 说明 |
|---|---|
| `connect_server(config, session)` | 连接并协商能力 |
| `reconnect_server(server_name)` | 崩溃重连 |
| `disconnect_server(server_name)` | 断开连接 |
| `get_server_status(server_name)` | 获取连接状态 |
| `list_connected_servers()` | 列出已连接服务器 |

---

## 遥测 API

### 结构化日志

```python
from praxis.telemetry.logger import get_logger
log = get_logger("component_name")
log.info("消息", key="value")
```

### 指标

```python
from praxis.telemetry.metrics import emit_metric
emit_metric("metric_name", 1.0, {"label": "value"}, "counter")
```

### 审计

```python
from praxis.telemetry.audit import record_audit
from praxis.models.telemetry import AuditEvent
await record_audit(AuditEvent(
    event_type="guardrail_verdict",
    component="guardrails",
    action="check_input",
    details={"verdict": "pass"},
))
```

---

## 数据模型索引

所有数据模型位于 `praxis.models` 包下：

| 模块 | 关键模型 |
|---|---|
| `models.responses` | `ModelResponse`, `Usage` |
| `models.orchestrator` | `AgentResponse`, `LoopState`, `TerminationReason` |
| `models.tools` | `ToolDefinition`, `ToolMetadata`, `ToolCall`, `ToolResult` |
| `models.guardrails` | `GuardrailVerdict`, `VerdictType` |
| `models.context` | `TurnContext`, `TokenUsage`, `AssembledPrompt` |
| `models.memory` | `MemoryEntry`, `WorkingMemory`, `MemoryScope` |
| `models.session` | `SessionMetadata`, `SessionStatus` |
| `models.subagent` | `SubagentSpec`, `SubagentResult`, `SubagentMode` |
| `models.skills` | `SkillDefinition`, `SkillMetadata`, `SkillIndexEntry` |
| `models.mcp` | `MCPToolInfo`, `MCPSamplingRequest`, `MCPElicitationRequest` |
| `models.recovery` | `CircuitState`, `RetryDecision` |
| `models.telemetry` | `AuditEvent` |
