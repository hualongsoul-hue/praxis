# 开发路线图

## 概览

- **项目名称**：Praxis — AI Agent Harness
- **总阶段数**：16
- **预估总任务数**：88
- **基于文档**：PRD v2.0（2026-04-16）

## 模块依赖图

```
                    ┌─────────────────────────────────┐
                    │  Phase 14: S13 子代理协调         │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │  Phase 13: S12 会话管理           │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │  Phase 12: S11 编排循环           │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │  Phase 11: S7 上下文引擎          │
                    └───────────────┬─────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
┌─────────▼────────┐  ┌────────────▼───────────┐  ┌──────────▼────────┐
│ Phase 7~8: S6    │  │ Phase 9: S10 验证引擎   │  │ Phase 10: S14     │
│ 记忆系统          │  │                        │  │ 技能系统           │
└────────┬─────────┘  └────────────┬───────────┘  └──────────┬────────┘
         │                         │                         │
         │            ┌────────────┴───────────┐             │
         │            │ Phase 6: S8 护栏 + S9   │             │
         │            │ 错误恢复                │             │
         │            └────────────────────────┘             │
         │                                                   │
┌────────▼───────────────────────────────────────────────────▼────────┐
│                     Phase 5: S5 工具系统核心                         │
└────────────────────────────┬───────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    Phase 4: S4 模型网关       │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  Phase 3: S2 遥测 + S3 持久化 │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    Phase 2: S1 配置系统       │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   Phase 1: 项目初始化        │
              └─────────────────────────────┘

Phase 15: S5 MCP 完整集成（在核心系统稳定后独立推进）
Phase 16: 集成测试与收尾优化
```

---

## Phase 1: 项目初始化与基础架构

**目标**：搭建项目脚手架、依赖管理、目录结构和共享基础设施，使项目处于可编译可运行状态。
**对应需求**：PRD § 2.4 组件总览、§ 11 技术栈约束
**前置依赖**：无
**完成标志**：`uv run python -c "import praxis"` 成功执行；pytest 可运行空测试套件；14 个组件包目录结构就位。

### 任务清单

- [ ] **1.1** 使用 uv 初始化 Python 项目，创建 `pyproject.toml`，声明所有核心依赖（pydantic、pydantic-settings、litellm、mcp、httpx 等）及开发依赖（pytest、pytest-asyncio 等）
  - 输出物：`pyproject.toml`、`uv.lock`
  - 验证：`uv sync` 成功完成，无依赖冲突

- [ ] **1.2** 创建 14 个组件的包目录结构，每个组件独立子包，包含 `__init__.py`
  - 输出物：`src/praxis/config/`、`src/praxis/telemetry/`、`src/praxis/persistence/`、`src/praxis/gateway/`、`src/praxis/tools/`、`src/praxis/memory/`、`src/praxis/context/`、`src/praxis/guardrails/`、`src/praxis/recovery/`、`src/praxis/verification/`、`src/praxis/orchestrator/`、`src/praxis/session/`、`src/praxis/subagent/`、`src/praxis/skills/`
  - 验证：`from praxis.config import *` 等 14 个导入均无报错

- [ ] **1.3** 创建共享数据模型基础包 `praxis.models`，定义跨组件共享的基础模型（`Message`、`ToolCall`、`ToolResult`、`ModelResponse`、`ModelResponseChunk` 等）
  - 输出物：`src/praxis/models/__init__.py`、`src/praxis/models/messages.py`、`src/praxis/models/tools.py`、`src/praxis/models/responses.py`
  - 验证：所有模型类可正常实例化和序列化

- [ ] **1.4** 定义全局异常体系和通用协议（Protocol）——Praxis 内部异常基类、各组件异常分类、通用接口协议
  - 输出物：`src/praxis/exceptions.py`、`src/praxis/protocols.py`
  - 验证：异常类可正常抛出和捕获；Protocol 类型检查通过

- [ ] **1.5** 配置测试基础设施——pytest 配置、公共 fixture 模板、异步测试支持
  - 输出物：`pyproject.toml` 中 pytest 配置、`tests/conftest.py`
  - 验证：`uv run pytest` 成功运行（0 测试通过，0 失败）

---

## Phase 2: S1 配置系统

**目标**：实现全局配置加载、验证和分发，为所有后续组件提供配置基础。
**对应需求**：PRD § 3.1 S1 配置系统（F1.1 ~ F1.3）
**前置依赖**：Phase 1
**完成标志**：可通过 YAML 文件 + 环境变量加载完整 `PraxisConfig`，配置校验失败时报告具体错误。

### 任务清单

- [ ] **2.1** 实现 `PraxisConfig` 顶层配置模型（Pydantic Settings），包含每个组件的配置切片字段
  - 输出物：`src/praxis/config/settings.py`
  - 验证：`PraxisConfig()` 可用默认值实例化，所有组件配置切片可访问

- [ ] **2.2** 实现分层配置加载——默认值 → YAML 配置文件 → 环境变量 → 命令行参数，按优先级合并
  - 输出物：`src/praxis/config/loader.py`
  - 验证：创建测试 YAML 文件并设置环境变量，验证优先级覆盖行为正确

- [ ] **2.3** 实现组件配置隔离——每个组件独立 Section，命名空间隔离，组件仅可访问自身配置
  - 输出物：`src/praxis/config/schemas.py`（各组件配置 Schema 定义）
  - 验证：调用 `get_component_config("gateway")` 返回 S4 配置切片，无法获取其他组件配置

- [ ] **2.4** 实现配置验证与热更新——启动时全量校验、自定义验证器注册、运行时 `reload_config` 发射变更事件
  - 输出物：`src/praxis/config/validation.py`
  - 验证：提供无效配置值时抛出详细错误信息；修改 YAML 后调用 reload 获取更新

---

## Phase 3: S2 遥测系统 + S3 持久化引擎

**目标**：建立可观测性基础设施和统一存储层，为上层组件提供日志、指标、追踪和持久化能力。
**对应需求**：PRD § 3.2 S2（F2.1 ~ F2.4）、§ 3.3 S3（F3.1 ~ F3.3）
**前置依赖**：Phase 2（S1 配置）
**完成标志**：结构化日志可输出 JSON 格式；指标可采集和导出；SQLite 后端可执行 CRUD 和检查点操作。

### 任务清单

- [ ] **3.1** S2：实现结构化日志框架——`get_logger(name)` 返回结构化 Logger，每条日志包含时间戳、级别、组件名、会话 ID、轮次号，支持 JSON 和人类可读两种格式，级别按组件配置
  - 输出物：`src/praxis/telemetry/logger.py`
  - 验证：调用 `get_logger("gateway").info("test")` 输出格式化 JSON 日志

- [ ] **3.2** S2：实现指标采集与导出——`emit_metric(name, value, tags)` 支持 Counter/Gauge/Histogram 三种类型，可导出为 Prometheus 格式或写入本地文件
  - 输出物：`src/praxis/telemetry/metrics.py`
  - 验证：发射多个指标后导出 Prometheus 文本格式，内容完整

- [ ] **3.3** S2：实现分布式追踪——`start_span(name, parent?)` 创建 Span 上下文，每个 Span 携带组件名/操作类型/耗时/状态码，支持 OpenTelemetry 协议导出
  - 输出物：`src/praxis/telemetry/tracing.py`
  - 验证：创建父子 Span 关系链，追踪数据可正确序列化

- [ ] **3.4** S2：实现审计日志——`record_audit(event)` 记录不可篡改的操作记录（工具调用、LLM 调用、权限决策），审计日志独立于常规日志，通过 S3 持久化
  - 输出物：`src/praxis/telemetry/audit.py`
  - 验证：记录审计事件后可通过 S3 读回完整审计记录

- [ ] **3.5** S3：实现多后端存储抽象——统一接口 `save/load/delete/list_keys`，支持 SQLite（默认）、Redis、文件系统三种后端，通过 S1 配置切换
  - 输出物：`src/praxis/persistence/store.py`、`src/praxis/persistence/backends/sqlite.py`、`src/praxis/persistence/backends/redis.py`、`src/praxis/persistence/backends/filesystem.py`
  - 验证：SQLite 后端通过完整 CRUD 测试；配置切换后端后接口行为一致

- [ ] **3.6** S3：实现检查点管理与命名空间隔离——`save_checkpoint/load_checkpoint` 支持完整状态快照序列化/反序列化，按会话 ID 列出历史检查点，命名空间隔离不同组件和会话数据
  - 输出物：`src/praxis/persistence/checkpoint.py`、`src/praxis/persistence/namespace.py`
  - 验证：保存检查点后可完整恢复状态；不同命名空间数据互不干扰

---

## Phase 4: S4 模型网关

**目标**：基于 LiteLLM 实现统一 LLM 接入层，支持 100+ Provider、负载均衡、故障转移、Token 计量和成本追踪。
**对应需求**：PRD § 4.1 S4（F4.1 ~ F4.6）
**前置依赖**：Phase 3（S2 遥测）
**完成标志**：可通过统一接口调用至少两个不同 Provider（如 OpenAI + Anthropic），流式输出正常，Token 计数和成本追踪有数据。

### 任务清单

- [ ] **4.1** 实现 LiteLLM Router 集成——从 S1 加载 `model_list` 配置，初始化 `litellm.Router`，支持多 `model_name` 别名和同名负载均衡，运行时动态注册新模型
  - 输出物：`src/praxis/gateway/router.py`
  - 验证：配置两个同名 `default` 模型部署，Router 初始化成功，模型列表可查询

- [ ] **4.2** 实现 `chat` / `chat_stream` 接口——同步和异步流式调用，流式传输中工具调用增量解析，流中断时部分结果保留
  - 输出物：`src/praxis/gateway/chat.py`
  - 验证：`chat` 返回完整 `ModelResponse`；`chat_stream` 逐块输出 `ModelResponseChunk`

- [ ] **4.3** 实现 `summarize` / `judge` 便捷接口——摘要接口用于 S7 上下文压缩，评估接口用于 S10 推理型验证
  - 输出物：`src/praxis/gateway/tasks.py`
  - 验证：`summarize` 返回摘要文本；`judge` 返回结构化评估结果

- [ ] **4.4** 实现 Token 计量、成本追踪与预算管控——`get_token_count`（预估）、`get_max_tokens`（窗口查询）、`completion_cost`（成本计算），通过 S2 发射 `llm_tokens_*` 和 `llm_cost` 指标，调用前预算检查
  - 输出物：`src/praxis/gateway/metering.py`
  - 验证：Token 计数与 LLM 返回的 usage 误差 <10%；成本计算返回非零 USD 值

- [ ] **4.5** 实现路由策略、重试/故障转移、异常标准化和可观测性回调——配置路由策略（latency/usage/cost-based），自动重试 + 指数退避，fallback 链 + cooldown，LiteLLM 异常映射到 Praxis 异常体系，`success_callback`/`failure_callback` 桥接 S2
  - 输出物：`src/praxis/gateway/resilience.py`、`src/praxis/gateway/callbacks.py`
  - 验证：模拟 Provider 故障时自动切换到备用部署；异常类型正确映射；S2 收到调用指标

---

## Phase 5: S5 工具系统核心

**目标**：实现工具注册表、内置工具集和执行管线，为 Agent 提供基础操作能力。
**对应需求**：PRD § 5.1 S5（F5.1、F5.3 ~ F5.5）
**前置依赖**：Phase 3（S3 持久化）
**完成标志**：内置工具可注册和执行；工具执行管线（参数验证→沙箱→结果捕获→格式化）完整运行；沙箱限制生效。

### 任务清单

- [ ] **5.1** 实现工具注册表——`register_tool/get_tool_schemas/get_tool_metadata` 接口，工具定义包含名称、描述、参数 Schema（Pydantic）、返回类型、元数据（类别、权限级别、只读/写），支持运行时动态注册/注销
  - 输出物：`src/praxis/tools/registry.py`
  - 验证：注册 3 个工具后 `get_tool_schemas()` 返回 3 个 JSON Schema

- [ ] **5.2** 实现内置文件操作工具——`read_file`、`write_file`、`edit_file`、`list_dir`，每个工具独立文件实现，遵循统一 Schema 定义和 `ToolResult` 返回格式
  - 输出物：`src/praxis/tools/builtins/file_ops/read_file.py`、`file_ops/write_file.py`、`file_ops/edit_file.py`、`file_ops/list_dir.py`
  - 验证：各工具对测试文件执行 CRUD 操作成功，返回结构化 `ToolResult`

- [ ] **5.3** 实现内置搜索工具——`grep_search`、`find_by_name`、`code_search`，每个工具独立文件实现
  - 输出物：`src/praxis/tools/builtins/search/grep_search.py`、`search/find_by_name.py`、`search/code_search.py`
  - 验证：在测试目录中搜索已知模式，返回正确匹配结果

- [ ] **5.4** 实现内置 Shell/网络/系统/自治管理工具，每个工具独立文件，按功能域子文件夹组织——Shell：`run_command`（同步/后台）；网络：`web_fetch`、`web_search`；系统：`get_system_info`；自治管理：`update_plan`、`update_notes`、`ask_user`、`submit_result`
  - 输出物：`src/praxis/tools/builtins/shell/run_command.py`、`network/web_fetch.py`、`network/web_search.py`、`system/get_system_info.py`、`autonomy/update_plan.py`、`autonomy/update_notes.py`、`autonomy/ask_user.py`、`autonomy/submit_result.py`
  - 验证：`run_command("echo hello")` 返回 "hello"；`get_system_info` 返回当前系统信息

- [ ] **5.5** 实现工具执行管线——`execute_tool(name, arguments)` 按序执行：参数验证（Schema 校验）→ 沙箱执行 → 结果捕获 → 格式化为 LLM 可读观察结果，支持并发策略（只读并发、写串行），可配置超时
  - 输出物：`src/praxis/tools/executor.py`
  - 验证：执行合法工具返回成功 ToolResult；参数不合法返回错误信息；超时触发超时错误

- [ ] **5.6** 实现沙箱执行环境与工具覆盖机制——文件系统访问限制（白名单路径）、Shell 命令超时限制、网络出站规则，以及同名工具覆盖（保持 Schema 兼容）
  - 输出物：`src/praxis/tools/sandbox.py`、`src/praxis/tools/override.py`
  - 验证：尝试访问白名单外路径被拒绝；自定义工具覆盖内置工具后行为符合预期

---

## Phase 6: S8 护栏系统 + S9 错误恢复

**目标**：实现安全策略引擎和错误恢复策略，为编排循环提供安全和容错保障。
**对应需求**：PRD § 6.1 S8（F8.1 ~ F8.3）、§ 6.2 S9（F9.1 ~ F9.4）
**前置依赖**：Phase 3（S2 遥测）
**完成标志**：三层护栏可对输入/工具/输出做出裁决；错误分类返回正确策略；熔断器状态转换正确。

### 任务清单

- [ ] **6.1** S8：实现三层护栏架构——输入护栏 `check_input`（提示注入检测、恶意指令检测）、工具护栏 `check_tool_call`（基于元数据裁决）、输出护栏 `check_output`（敏感信息检测、内容安全），绊线触发时返回 block + tripwire 标记
  - 输出物：`src/praxis/guardrails/engine.py`
  - 验证：注入恶意输入时返回 block；安全输入返回 pass；危险工具返回 confirm

- [ ] **6.2** S8：实现权限分层系统——三级权限 `auto_approve/confirm/deny`，默认限制性策略，YAML 声明式权限配置（按工具名/类别/路径模式），运行时临时授权
  - 输出物：`src/praxis/guardrails/permissions.py`
  - 验证：只读工具自动批准；写工具默认需确认；配置放宽后行为变更

- [ ] **6.3** S8：实现护栏规则引擎——统一规则定义（Pydantic 模型）、内置规则集、自定义规则扩展、有序评估 + 短路逻辑、裁决审计日志（S2）
  - 输出物：`src/praxis/guardrails/rules.py`
  - 验证：注册自定义规则后评估顺序正确；首个 deny 短路后续评估

- [ ] **6.4** S9：实现四类错误分类与恢复策略——`classify_error` 返回 `ErrorClassification`（Transient/Model-Recoverable/User-Fixable/Unexpected），每类附带建议策略
  - 输出物：`src/praxis/recovery/classifier.py`
  - 验证：`ConnectionTimeout` 分类为 Transient；`InvalidToolArgs` 分类为 Model-Recoverable

- [ ] **6.5** S9：实现重试策略与熔断器——`get_retry_decision` 指数退避 + 随机抖动 + 可配置上限，`check_circuit` 三态模型（Closed→Open→Half-Open）+ 状态转换日志
  - 输出物：`src/praxis/recovery/retry.py`、`src/praxis/recovery/circuit_breaker.py`
  - 验证：连续 3 次失败后熔断器 Open；cooldown 后转为 Half-Open；探测成功回到 Closed

- [ ] **6.6** S9：实现优雅降级——`get_fallback` 维护工具降级映射，LLM Provider 降级通知 S4，降级事件通过 S2 记录
  - 输出物：`src/praxis/recovery/fallback.py`
  - 验证：配置 `web_fetch → web_search` 降级映射后，`get_fallback("web_fetch")` 返回 `"web_search"`

---

## Phase 7: S6 记忆系统 — 存储与检索

**目标**：实现四类认知记忆的存储、多作用域隔离和混合检索能力。
**对应需求**：PRD § 5.2 S6（F6.1、F6.4 ~ F6.8）
**前置依赖**：Phase 3（S3 持久化）、Phase 4（S4 模型网关）
**完成标志**：四类记忆可创建/存储/检索；多作用域隔离生效；语义搜索返回相关结果。

### 任务清单

- [ ] **7.1** 实现四类认知记忆数据模型——语义记忆（集合模式 + 档案模式）、情景记忆（学习范例）、程序记忆（行为准则）、工作记忆（消息序列），所有模型基于 Pydantic BaseModel
  - 输出物：`src/praxis/models/memory.py`
  - 验证：每种记忆类型可正常实例化、序列化/反序列化

- [ ] **7.2** 实现多作用域记忆隔离——`session/<id>`、`project/<name>`、`user/<id>`、`global` 四级作用域，复合查询支持，项目级记忆从 `praxis.md` 加载，跨作用域隔离防泄露
  - 输出物：`src/praxis/memory/scope.py`
  - 验证：不同作用域存储的记忆互不可见；复合查询（项目+用户）返回正确交集

- [ ] **7.3** 实现向量嵌入存储与语义搜索——记忆内容向量化存储，`search_memory(query)` 语义相似度检索
  - 输出物：`src/praxis/memory/vector_store.py`
  - 验证：存储 10 条记忆后，语义搜索 "数据库配置" 能命中包含 "PostgreSQL 设置" 的记忆

- [ ] **7.4** 实现元数据过滤 + 重排序 + 渐进式检索——按作用域/类型/标签/时间范围过滤，语义候选集二次评分重排，三层结构（轻量索引 ~150 字符/条 → 摘要 → 完整内容）
  - 输出物：`src/praxis/memory/retrieval.py`
  - 验证：`get_memory_index()` 返回轻量索引列表；`search_memory` 支持类型过滤；`load_memory_detail` 返回完整内容

- [ ] **7.5** 实现工作记忆 Scratchpad——`read_scratchpad/write_scratchpad` 接口，支持 `progress.json`、`todos.json`、`features.json`，持久化到 S3 文件系统
  - 输出物：`src/praxis/memory/scratchpad.py`
  - 验证：写入后读取 Scratchpad 内容一致；重启后通过 S3 恢复

- [ ] **7.6** 实现记忆保留与版本管理——时间衰减（可配置衰减曲线）、动态遗忘（低相关性标记 INACTIVE）、版本历史（关键事实更新保留版本链）、不可变审计（ACTIVE/INACTIVE/SUPERSEDED 状态标记，完整变更历史），通过 S2 记录使用统计
  - 输出物：`src/praxis/memory/retention.py`
  - 验证：记忆更新后旧版本可追溯；长期未检索记忆被标记 INACTIVE

---

## Phase 8: S6 记忆系统 — 智能管线

**目标**：实现模型辅助的记忆提取、整合管线和后台自治处理机制。
**对应需求**：PRD § 5.2 S6（F6.2 ~ F6.3）
**前置依赖**：Phase 7（S6 存储与检索）
**完成标志**：`append_message` 后后台自动提取记忆并整合；梦境整理可手动触发并产出整理报告。

### 任务清单

- [ ] **8.1** 实现模型辅助记忆提取（Extraction）——每种认知类型使用独立提取提示，通过 S4 调用 LLM 分析对话内容，区分有意义洞察和例行对话，单次对话可提取多条多类型记忆
  - 输出物：`src/praxis/memory/extraction.py`
  - 验证：输入包含用户偏好的对话，提取出语义类型记忆条目

- [ ] **8.2** 实现记忆整合（Consolidation）——新记忆经语义搜索匹配已有记忆，LLM 评估做出 ADD/UPDATE/NOOP 决策，冲突解决（优先最新，旧标 INACTIVE），语义去重，不可变审计日志，失败保守 ADD
  - 输出物：`src/praxis/memory/consolidation.py`
  - 验证：新增不同记忆返回 ADD；重复记忆返回 NOOP；更新记忆返回 UPDATE 且旧记忆标 INACTIVE

- [ ] **8.3** 实现双路径处理——热路径通过公开接口（`save_memory/search_memory`）供 Agent 主动调用，后台路径在 `append_message` 内部触发异步任务自动处理
  - 输出物：`src/praxis/memory/pipeline.py`
  - 验证：`save_memory` 即时写入（热路径）；`append_message` 后后台任务自动执行提取

- [ ] **8.4** 实现后台异步自治任务——asyncio.Task 随 S6 实例创建自动启动，消息游标持久化，可配置批量阈值，背压控制（不阻塞 append_message），优雅关闭（clear_session 等待完成），失败容错（记入 S2，下次重试），import_state 恢复游标后重启
  - 输出物：`src/praxis/memory/background.py`
  - 验证：连续 append 10 条消息后游标正确推进；export_state 包含游标；import_state 后从游标继续

- [ ] **8.5** 实现梦境整理（Dream Consolidation）——触发条件判断（>24h + ≥5 次新会话），通过 S4 驱动 LLM 执行时间锚定、矛盾消解、陈旧清理、索引精简，整理报告通过 S2 记录
  - 输出物：`src/praxis/memory/dream.py`
  - 验证：手动触发 `run_dream()` 产出整理报告（整理/删除/合并条目数）

---

## Phase 9: S10 验证引擎

**目标**：实现计算型、推理型和视觉三种验证能力，支持 GAV 循环和质量左移。
**对应需求**：PRD § 6.3 S10（F10.1 ~ F10.6）
**前置依赖**：Phase 4（S4 模型网关）、Phase 5（S5 工具系统）
**完成标志**：计算型验证可运行测试/lint 并返回结构化结果；推理型验证可调用 LLM 评估代码质量。

### 任务清单

- [ ] **9.1** 实现计算型验证（Computational Verification）——Verifier Protocol，内置验证器：测试套件（通过 S5 运行）、类型检查、Lint、Schema 校验，返回结构化失败详情（文件/行号/消息）
  - 输出物：`src/praxis/verification/computational.py`
  - 验证：对有语法错误的 Python 文件运行 Lint 验证器，返回包含行号的失败详情

- [ ] **9.2** 实现推理型验证（Inferential Verification / LLM-as-Judge）——通过 S4 `judge` 接口独立评估，评估代理与执行代理分离（不同上下文），自定义评估标准，返回数值评分 + 判定 + 文字反馈
  - 输出物：`src/praxis/verification/inferential.py`
  - 验证：对一段代码进行正确性和完整性评估，返回评分和反馈文本

- [ ] **9.3** 实现视觉验证——通过 Playwright 截取页面截图，截图提交 S4 多模态能力比对，返回截图 + 视觉判定 + 差异描述
  - 输出物：`src/praxis/verification/visual.py`
  - 验证：对本地 HTML 页面截图并验证是否包含特定元素

- [ ] **9.4** 实现 GAV 循环支持——作为 Verify 阶段执行者，验证失败时返回结构化结果供 S11 注入上下文，前馈/反馈控制矩阵覆盖四象限
  - 输出物：`src/praxis/verification/gav.py`
  - 验证：模拟 GAV 流程：验证失败 → 返回详情 → 验证通过 → 返回 pass

- [ ] **9.5** 实现验证器注册与质量左移——`register_verifier` 接口，支持自定义验证器扩展，质量左移策略（集成前/集成后/持续监控/运行时反馈）配置
  - 输出物：`src/praxis/verification/registry.py`
  - 验证：注册自定义验证器后可通过 `run_computational` 调用

---

## Phase 10: S14 技能系统

**目标**：实现技能的发现、加载、渐进式披露和生命周期管理，为 Agent 提供程序化知识能力。
**对应需求**：PRD § 5.4 S14（F14.1 ~ F14.8）
**前置依赖**：Phase 5（S5 工具系统）
**完成标志**：可从本地目录发现技能；三层渐进式披露正常工作；技能附带脚本可通过 S5 执行。

### 任务清单

- [ ] **10.1** 实现技能格式解析——解析 `SKILL.md`（YAML frontmatter + Markdown 正文），目录结构验证，附属文件发现
  - 输出物：`src/praxis/skills/parser.py`
  - 验证：解析包含完整 frontmatter 的 SKILL.md，返回正确的元数据和内容

- [ ] **10.2** 实现渐进式披露——第一层（name + description ~150 字符索引）、第二层（完整 SKILL.md 正文）、第三层（附属文件按需加载），`get_skill_index` / `load_skill` / `load_skill_file` 接口
  - 输出物：`src/praxis/skills/disclosure.py`
  - 验证：`get_skill_index()` 返回轻量列表；`load_skill()` 返回完整内容；`load_skill_file()` 返回附属文件

- [ ] **10.3** 实现技能发现与安装——从配置路径（`~/.praxis/skills/`、`.praxis/skills/`）扫描，版本控制共享支持，安装安全审计（代码依赖、网络连接、敏感操作检查）
  - 输出物：`src/praxis/skills/discovery.py`
  - 验证：在测试技能目录放置 3 个技能，`discover_skills(paths)` 返回 3 个技能元数据

- [ ] **10.4** 实现技能触发与激活——自动触发（语义匹配任务描述）和手动触发，`evaluate_relevance` 接口，多技能并行激活，激活事件通过 S2 记录
  - 输出物：`src/praxis/skills/activation.py`
  - 验证：任务描述 "处理 PDF" 时 `evaluate_relevance` 将 pdf-processor 排在首位

- [ ] **10.5** 实现技能与工具系统协同——技能声明工具依赖（激活时确保工具已注册），技能附带 Python 脚本通过 S5 执行，`list_skill_tools` 接口
  - 输出物：`src/praxis/skills/tools_bridge.py`
  - 验证：激活含脚本的技能后，脚本工具出现在 S5 工具注册表中

- [ ] **10.6** 实现技能统一管理——`register_skill/unregister_skill` 运行时热加载，版本管理（多版本共存、升级、回退），使用统计（S2 记录触发次数/成功率/上下文消耗），索引缓存通过 S3 持久化
  - 输出物：`src/praxis/skills/manager.py`
  - 验证：运行时注册新技能后 `get_skill_index()` 立即返回更新列表

---

## Phase 11: S7 上下文引擎

**目标**：实现分层 Prompt 组装、上下文压缩和智能检索，为每轮 LLM 调用组装最优上下文。
**对应需求**：PRD § 5.3 S7（F7.1 ~ F7.5）
**前置依赖**：Phase 4（S4）、Phase 5（S5）、Phase 7~8（S6）、Phase 10（S14）
**完成标志**：`assemble_prompt` 产出完整 Prompt（系统提示 + 工具 + 记忆 + 用户消息）；Token 超限时自动触发压缩。

### 任务清单

- [ ] **11.1** 实现分层 Prompt 组装——按 8 层优先级栈组装（系统提示 → 工具定义 → 开发者指令 → 用户指令 → 记忆索引 → 工作记忆 → 语义检索 → 当前消息），XML/Markdown 标签分隔，关键内容放置首尾
  - 输出物：`src/praxis/context/assembler.py`
  - 验证：`assemble_prompt` 输出包含所有 8 层内容，Token 计数正确

- [ ] **11.2** 实现动态工具集注入——根据任务阶段从 S5 过滤工具 Schema，支持工具分组和懒加载，按 LLM 原生 function calling 格式注入
  - 输出物：`src/praxis/context/tool_injection.py`
  - 验证：不同任务阶段注入不同工具子集；初始仅核心工具，扩展工具按需加载

- [ ] **11.3** 实现上下文压缩（Compaction）——Token 超过阈值（默认 80%）自动触发，调用 S4 `summarize` 生成摘要，保留规则（架构决策/Bug/实现细节保留，冗余丢弃），保留最近 5 个关键文件引用，工具结果清除策略
  - 输出物：`src/praxis/context/compaction.py`
  - 验证：180K Token 触发压缩后降至 ~72K；关键信息保留完整

- [ ] **11.4** 实现观察遮蔽（Observation Masking）——隐藏旧工具输出内容但保持调用记录可见，策略可配置：按轮次距离（默认 >10 轮）、按输出大小（默认 >2000 Token）
  - 输出物：`src/praxis/context/masking.py`
  - 验证：15 轮前的大工具输出被遮蔽，调用记录仍可见

- [ ] **11.5** 实现即时检索（JIT Retrieval）——维护轻量标识符（文件名、函数签名），按需动态加载完整内容，Few-shot 示例管理（按任务类型索引和注入）
  - 输出物：`src/praxis/context/jit_retrieval.py`
  - 验证：标识符列表存在但完整内容仅在请求时加载

---

## Phase 12: S11 编排循环

**目标**：实现 TAO/ReAct 核心循环，协调所有组件完成完整 Agent 轮次。
**对应需求**：PRD § 7.1 S11（F11.1 ~ F11.6）
**前置依赖**：Phase 4（S4）、Phase 5（S5）、Phase 6（S8+S9）、Phase 7~8（S6）、Phase 9（S10）、Phase 10（S14）、Phase 11（S7）
**完成标志**：`run` 接口可完整执行多轮 TAO 循环（Prompt 组装 → LLM 推理 → 工具执行 → 终止判定）；事件流实时输出。

### 任务清单

- [ ] **12.1** 实现 TAO/ReAct 核心循环引擎——`run/run_stream/abort/get_state` 接口，完整循环（消息记录 → Prompt 组装 → LLM 推理 → 工具执行 → 上下文更新 → 终止检查），支持同步/异步/流式三种模式，单次循环开销 <50ms
  - 输出物：`src/praxis/orchestrator/loop.py`
  - 验证：发送用户消息后 Agent 完成一次完整工具调用循环并返回最终响应

- [ ] **12.2** 实现输出解析——依赖 LLM 原生 `tool_calls` 结构化输出，判断逻辑（有 tool_calls→执行→继续，无→最终响应→退出），并行工具调用批量解析，Handoff 请求检测，Pydantic Schema 约束响应
  - 输出物：`src/praxis/orchestrator/parser.py`
  - 验证：解析含多个 tool_calls 的响应，正确识别每个工具调用

- [ ] **12.3** 实现终止条件管理——6 层优先级评估：护栏绊线 → 用户中断 → 安全拒绝 → 自然终止 → 最大轮次 → Token 耗尽，所有阈值通过 S1 配置
  - 输出物：`src/praxis/orchestrator/termination.py`
  - 验证：达到最大轮次时强制终止并返回中间状态；绊线触发时立即终止

- [ ] **12.4** 实现工具调用协调流程——按序调用 S8.check_tool_call → S9.check_circuit → S5.execute_tool → S9.record_outcome → 失败时 S9.classify_error 决策 → 可选 S10 验证
  - 输出物：`src/praxis/orchestrator/tool_coordination.py`
  - 验证：模拟完整工具调用链路，护栏拒绝/熔断/执行失败/成功各路径正确

- [ ] **12.5** 实现循环策略选择——ReAct 模式（默认，交叉推理与行动）和 Plan-and-Execute 模式（先规划后执行），运行时可切换
  - 输出物：`src/praxis/orchestrator/strategy.py`
  - 验证：ReAct 模式逐步执行；Plan-and-Execute 模式先输出计划再批量执行

- [ ] **12.6** 实现事件发射系统——关键节点发射结构化事件（turn_start/llm_request/llm_response/tool_call_start/tool_call_end/verification_result/turn_end/termination），通过 S2 记录并通过流式接口实时推送
  - 输出物：`src/praxis/orchestrator/events.py`
  - 验证：`run_stream` 输出事件流包含预期的事件序列

---

## Phase 13: S12 会话管理

**目标**：实现会话全流程管理——创建、运行、检查点、恢复、时间旅行，为长时间运行 Agent 提供状态持续性。
**对应需求**：PRD § 7.2 S12（F12.1 ~ F12.5）
**前置依赖**：Phase 12（S11 编排循环）
**完成标志**：完整会话可创建/运行/暂停/恢复；检查点自动保存并可回退。

### 任务清单

- [ ] **13.1** 实现会话初始化——`create_session` 创建新会话时初始化所有组件实例（S4~S11、S14），注入配置，加载项目级记忆/工具/权限，生成会话 ID 和初始检查点
  - 输出物：`src/praxis/session/core.py`
  - 验证：`create_session` 返回 Session 对象，所有组件实例可访问

- [ ] **13.2** 实现会话恢复——`resume_session` 从 S3 加载检查点，恢复 S6 记忆状态（自动从游标继续后台处理）、S7 上下文状态，重建无状态组件，验证完整性
  - 输出物：`src/praxis/session/resume.py`
  - 验证：保存检查点后终止会话，`resume_session` 恢复后上下文完整

- [ ] **13.3** 实现自动检查点——每次 S11 循环终止后自动保存，检查点包含会话元数据 + S6/S7/S11 状态快照，通过 S3 写入，延迟 <200ms
  - 输出物：`src/praxis/session/checkpoint.py`
  - 验证：完成一轮对话后检查点自动保存；写入耗时 <200ms

- [ ] **13.4** 实现跨上下文窗口续接——两阶段模式（初始化 + 增量进度），标准热身序列（检查目录 → 读进度文件 → 验证基础功能 → 开始工作），长时间任务支持（一次一功能、干净状态、增量前进）
  - 输出物：`src/praxis/session/continuation.py`
  - 验证：新会话执行初始化阶段；恢复会话执行热身序列后继续工作

- [ ] **13.5** 实现时间旅行调试——`rollback` 回退到任意历史检查点，`list_checkpoints` 查看历史，回退后 S6/S7/S11 状态全部恢复，可从回退点重新运行
  - 输出物：`src/praxis/session/time_travel.py`
  - 验证：执行 5 轮后回退到第 3 轮检查点，状态完全恢复；从第 3 轮重新运行产生不同路径

---

## Phase 14: S13 子代理协调

**目标**：实现多 Agent 场景的子代理创建、上下文隔离和结果聚合。
**对应需求**：PRD § 7.3 S13（F13.1 ~ F13.4）
**前置依赖**：Phase 12（S11）、Phase 13（S12）
**完成标志**：Agent-as-Tool 模式可创建子代理执行子任务并返回精炼摘要。

### 任务清单

- [ ] **14.1** 实现 Agent-as-Tool 执行模型——`spawn_agent_as_tool` 创建专家子代理，独立 S11 循环和上下文窗口，工具集为主代理子集，执行后返回精炼摘要（~1,500 Token）
  - 输出物：`src/praxis/subagent/spawn.py`
  - 验证：主代理委托子任务后收到精炼摘要结果

- [ ] **14.2** 实现 Handoff 执行模型——`handoff` 将控制权转移到专家代理，传递精炼上下文摘要（非完整历史），完成后返回主代理
  - 输出物：`src/praxis/subagent/handoff.py`
  - 验证：Handoff 后主代理暂停，专家代理执行完毕后控制权正确返回

- [ ] **14.3** 实现 Fork 执行模型——`fork` 子代理获得父上下文只读副本，并行独立执行，`collect_results` 聚合多个子代理输出
  - 输出物：`src/praxis/subagent/fork.py`
  - 验证：Fork 两个子代理并行执行，结果正确聚合

- [ ] **14.4** 实现上下文隔离与结果聚合——每个子代理独立组件实例集，子代理间不共享可变状态，遥测关联父追踪链路，结果聚合 + 冲突协调
  - 输出物：`src/praxis/subagent/isolation.py`、`src/praxis/subagent/aggregation.py`
  - 验证：子代理的状态变更不影响主代理；冲突结果被标记

- [ ] **14.5** 实现资源管控——并发数上限（默认 5），独立 Token 预算和轮次上限，超时控制（超时强制终止返回部分结果）
  - 输出物：`src/praxis/subagent/resource_control.py`
  - 验证：同时创建 6 个子代理时第 6 个等待；子代理超时后强制终止

---

## Phase 15: S5 MCP 完整集成

**目标**：实现 MCP 协议的完整支持——传输层、三大服务器原语、三大客户端特性，使 Praxis 成为合格的 MCP Host。
**对应需求**：PRD § 5.1 S5（F5.2）
**前置依赖**：Phase 5（S5 核心）、Phase 12（S11，Sampling/Elicitation 需要编排循环）
**完成标志**：可连接外部 MCP Server 并使用其 Tools/Resources/Prompts；Elicitation 和 Sampling 流程完整。

### 任务清单

- [ ] **15.1** 实现 MCP 传输层——Stdio 传输（子进程 stdin/stdout）和 Streamable HTTP 传输（SSE + 多客户端并发），上层统一 MCP Client 接口
  - 输出物：`src/praxis/tools/mcp/transport.py`
  - 验证：通过 Stdio 连接本地 MCP Server 成功；HTTP 连接远程 Server 成功

- [ ] **15.2** 实现 MCP Tools 集成——`tools/list` 发现工具 + `tools/call` 执行 + 动态更新（`notifications/tools/list_changed`），MCP 工具与内置工具在注册表中统一管理
  - 输出物：`src/praxis/tools/mcp/tools.py`
  - 验证：连接测试 MCP Server 后其工具出现在 `get_tool_schemas()` 中；调用 MCP 工具返回结果

- [ ] **15.3** 实现 MCP Resources 集成——`resources/list` 发现 + `resources/read` 读取 + 资源模板（参数化 URI）+ `resources/subscribe` 订阅变更
  - 输出物：`src/praxis/tools/mcp/resources.py`
  - 验证：读取 MCP Server 提供的资源，返回正确内容和 MIME 类型

- [ ] **15.4** 实现 MCP Prompts 集成——`prompts/list` 发现 + `prompts/get` 获取 + 参数补全（Completion）
  - 输出物：`src/praxis/tools/mcp/prompts.py`
  - 验证：获取参数化提示模板并填充参数后返回完整消息列表

- [ ] **15.5** 实现 MCP 客户端特性——Elicitation（结构化信息征询 + URL Mode）、Sampling（代理 LLM 调用 + Human-in-the-loop + 工具调用循环）、Roots（工作目录声明）
  - 输出物：`src/praxis/tools/mcp/elicitation.py`、`src/praxis/tools/mcp/sampling.py`、`src/praxis/tools/mcp/roots.py`
  - 验证：MCP Server 发起 Elicitation 请求后正确转发给用户；Sampling 请求通过 S4 完成

- [ ] **15.6** 实现 MCP 服务器连接管理——初始化阶段能力协商、崩溃自动重连、会话状态检查点（S3）、独立安全边界
  - 输出物：`src/praxis/tools/mcp/connection.py`
  - 验证：MCP Server 崩溃后自动重连；重连后状态恢复

- [ ] **15.7** 实现 MCP 授权与 Tasks——OAuth 2.0/PKCE 授权流程、Token 管理、Protected Resource Metadata 发现，以及实验性 MCP Tasks 原语支持
  - 输出物：`src/praxis/tools/mcp/auth.py`、`src/praxis/tools/mcp/tasks.py`
  - 验证：OAuth 授权流程完成后可访问受保护的 MCP Server

---

## Phase 16: 集成测试与收尾优化

**目标**：端到端验证所有组件协作、性能基准、非功能需求达标，完成发布准备。
**对应需求**：PRD § 8 跨组件调用链（场景一~八）、§ 9 非功能需求
**前置依赖**：Phase 1 ~ Phase 15 全部完成
**完成标志**：8 个 PRD 场景全部通过端到端测试；性能指标满足 PRD 要求；可发布 v1.0。

### 任务清单

- [ ] **16.1** 端到端场景测试——覆盖 PRD 8 个场景：单轮次执行、错误恢复、上下文溢出、会话恢复、子代理委托、护栏绊线、技能工作流、MCP 交互
  - 输出物：`tests/e2e/test_scenario_*.py`（8 个场景测试文件）
  - 验证：8 个场景测试全部通过

- [ ] **16.2** 跨组件集成测试——验证依赖链路完整性（S12→S11→S7→S6→S4→S2→S1），验证状态传递和接口契约
  - 输出物：`tests/integration/test_component_chain.py`
  - 验证：完整调用链路无断裂，返回值类型匹配接口契约

- [ ] **16.3** 性能基准测试——循环开销 <50ms、工具延迟 <100ms、检查点写入 <200ms、压缩 <5s、Prompt 组装 <20ms、护栏裁决 <10ms
  - 输出物：`tests/benchmarks/test_performance.py`
  - 验证：所有性能指标满足 PRD § 9.1 目标

- [ ] **16.4** 非功能需求验证——安全性（沙箱隔离、敏感数据防护、审计完整率 100%）、可靠性（单步 ≥99.5%、恢复 100%）、可扩展性（100+ 工具、1000+ 轮次、10000+ 记忆）
  - 输出物：`tests/nfr/test_security.py`、`tests/nfr/test_reliability.py`、`tests/nfr/test_scalability.py`
  - 验证：所有非功能需求指标达标

- [ ] **16.5** 文档与示例——API 文档生成、使用指南（快速开始 + 配置参考 + 技能编写指南）、配置模板
  - 输出物：`docs/API.md`、`docs/GUIDE.md`、`docs/CONFIG_REFERENCE.md`、`config.example.yaml`
  - 验证：按文档步骤操作可成功启动和使用 Praxis

- [ ] **16.6** 发布准备——版本号设定、CHANGELOG 编写、PyPI 发布配置、README 完善
  - 输出物：`CHANGELOG.md`、更新后的 `pyproject.toml`（version + metadata）、`README.md`
  - 验证：`uv build` 成功生成可分发包
