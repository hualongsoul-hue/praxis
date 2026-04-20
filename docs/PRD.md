# Praxis — AI Agent Harness 产品需求文档（PRD）

> **Praxis** (普拉克西斯 / 算子)：古希腊语，意为"将理论转化为行动的过程"或"实践"。
>
> 文档版本：v2.0 | 创建日期：2026-04-16

---

## 1. 概述

### 1.1 产品定位

Praxis 是一个 **AI Agent Harness（智能体运行时基础设施）**——包裹在 LLM 外层的完整软件基础设施，负责管理 Agent 的生命周期、上下文、工具调用、技能加载和与外部世界的一切交互。

**核心理念**：Agent 是涌现行为（目标驱动、工具使用、自我纠错的实体），Harness 是产生该行为的机器。如果不是模型本身，那就是 Harness 的一部分。

> *"原始 LLM 是没有 RAM、磁盘和 I/O 的 CPU。上下文窗口是 RAM，外部数据库是磁盘，工具集成是设备驱动。Harness 就是操作系统。"* —— Beren Millidge

### 1.2 要解决的问题

无 Harness 的 Agent 在生产环境中面临系统性失败：

- **上下文腐烂（Context Rot）**：窗口填满后模型遗忘原始目标或忽略关键指令
- **无状态健忘（AI Amnesia）**：网络错误或系统重启导致所有进度丢失
- **工具调用错误**：模型产生错误参数的工具调用，无验证层则陷入重复失败
- **无限循环**：Agent 缺乏记忆来意识到已尝试过同一路径
- **幻觉式工具使用**：Agent 调用不存在的工具或使用错误参数
- **资源失控消耗**：反复调用昂贵 API，成本膨胀而任务未完成
- **复合失败率**：10 步流程每步 99% 成功率，端到端仅 ~90.4%

### 1.3 目标用户

| 用户角色 | 使用场景 |
|---------|---------|
| AI 应用开发者 | 基于 Praxis 构建编码助手、研究代理、自动化工作流等 Agent 应用 |
| 平台工程师 | 部署和运维长时间运行的自治 Agent |
| Agent 终端用户 | 通过 CLI/API 与 Praxis 驱动的 Agent 交互 |

### 1.4 设计原则

| 原则 | 说明 |
|------|------|
| **组件组合** | Harness 由独立组件组合而成，每个组件有明确边界和公开接口 |
| **分层依赖** | 上层依赖下层，禁止反向依赖和跨层循环依赖 |
| **薄 Harness** | 最小化硬编码逻辑，信任模型能力，随模型进步削减脚手架复杂度 |
| **上下文即稀缺资源** | 每个 Token 有注意力成本；追求最小高信号 Token 集合 |
| **计算确定性优先** | 能用 CPU 确定性解决的不用推理 GPU 解决 |
| **模型可替换** | 业务逻辑与模型解耦，支持热替换 LLM Provider |
| **渐进式自治** | 从人机协作到有监督自治再到全自治，分阶段提升信任 |

---

## 2. 组件架构

### 2.1 设计哲学：组件组合

Praxis 不是一个单体应用，而是 **14 个独立组件的组合体**。每个组件：

- 拥有明确的职责边界（单一职责）
- 通过公开接口契约与其他组件交互
- 可独立测试和替换
- 严格遵循分层依赖规则

### 2.2 五层架构模型

```
╔══════════════════════════════════════════════════════════════════════╗
║  Layer 4 · 编排层 (Orchestration)                                     ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                ║
║  │S11 编排循环   │  │S12 会话管理   │  │S13 子代理协调 │                ║
║  └──────────────┘  └──────────────┘  └──────────────┘                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Layer 3 · 控制层 (Control)                                           ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                ║
║  │S8 护栏系统    │  │S9 错误恢复   │  │S10 验证引擎   │                ║
║  └──────────────┘  └──────────────┘  └──────────────┘                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Layer 2 · 核心能力层 (Core Capabilities)                              ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ ┌────────────┐ ║
║  │S5 工具系统    │  │S6 记忆系统   │  │S7 上下文引擎  │ │S14 技能系统│ ║
║  └──────────────┘  └──────────────┘  └──────────────┘ └────────────┘ ║
╠══════════════════════════════════════════════════════════════════════╣
║  Layer 1 · 模型接入层 (Model Access)                                   ║
║  ┌──────────────────────────────────────────────────────────┐       ║
║  │                S4 模型网关 (Model Gateway)                 │       ║
║  └──────────────────────────────────────────────────────────┘       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Layer 0 · 基础设施层 (Foundation)                                     ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                ║
║  │S1 配置系统    │  │S2 遥测系统   │  │S3 持久化引擎  │                ║
║  └──────────────┘  └──────────────┘  └──────────────┘                ║
╚══════════════════════════════════════════════════════════════════════╝
```

**分层规则**：每一层仅允许依赖同层或更低层的组件，禁止向上依赖。

### 2.3 组件依赖图

```
S13(子代理协调) ───▶ S11, S12
S12(会话管理)   ───▶ S3, S6, S7, S11, S14
S11(编排循环)   ───▶ S4, S5, S6, S7, S8, S9, S10, S14
                          │
S10(验证引擎)   ───▶ S4, S5
S9 (错误恢复)   ───▶ (无组件依赖，仅 S1, S2)
S8 (护栏系统)   ───▶ (无组件依赖，仅 S1, S2)
                          │
S14(技能系统)   ───▶ S3, S5
S7 (上下文引擎) ───▶ S4, S5, S6, S14
S6 (记忆系统)   ───▶ S3, S4
S5 (工具系统)   ───▶ S3
                          │
S4 (模型网关)   ───▶ (无组件依赖，仅 S1, S2)
                          │
S3 (持久化引擎) ───▶ (无组件依赖，仅 S1)
S2 (遥测系统)   ───▶ (无组件依赖，仅 S1)
S1 (配置系统)   ───▶ (无依赖)
```

> **注意**：所有组件隐式依赖 S1（配置）和 S2（遥测），上图中省略这两个通用依赖以保持清晰。

### 2.4 组件总览

| 编号 | 组件 | 包路径 | 层级 | 职责边界 |
|------|--------|--------|------|---------|
| S1 | 配置系统 | `praxis.config` | L0 | 全局配置加载、验证、热更新；所有组件的配置来源 |
| S2 | 遥测系统 | `praxis.telemetry` | L0 | 结构化日志、指标采集、分布式追踪、审计日志 |
| S3 | 持久化引擎 | `praxis.persistence` | L0 | 存储抽象（SQLite/Redis/文件系统）；检查点读写 |
| S4 | 模型网关 | `praxis.gateway` | L1 | 基于 LiteLLM 的统一 LLM 接入；100+ Provider 开箱即用；Router 负载均衡与故障转移；流式传输；Token 计量与成本追踪 |
| S5 | 工具系统 | `praxis.tools` | L2 | 工具注册、Schema 管理、MCP 集成、沙箱执行 |
| S6 | 记忆系统 | `praxis.memory` | L2 | 四类认知记忆（语义/情景/程序/工作）；模型辅助提取与整合；混合检索；跨会话持续性 |
| S7 | 上下文引擎 | `praxis.context` | L2 | Prompt 组装、上下文压缩、遮蔽、即时检索 |
| S8 | 护栏系统 | `praxis.guardrails` | L3 | 权限策略、输入/输出/工具护栏、绊线机制 |
| S9 | 错误恢复 | `praxis.recovery` | L3 | 错误分类、重试策略、熔断器、优雅降级 |
| S10 | 验证引擎 | `praxis.verification` | L3 | 计算型/推理型/视觉验证；Gather-Act-Verify 循环 |
| S11 | 编排循环 | `praxis.orchestrator` | L4 | TAO/ReAct 核心循环、输出解析、终止判定 |
| S12 | 会话管理 | `praxis.session` | L4 | 会话初始化/恢复、检查点、跨窗口续接 |
| S13 | 子代理协调 | `praxis.subagent` | L4 | 子代理创建、上下文隔离、结果聚合 |

---

## 3. Layer 0 · 基础设施层

### 3.1 S1：配置系统（Configuration System）

**包路径**：`praxis.config`
**依赖**：无
**被依赖**：所有其他组件

#### 职责边界

配置系统是所有组件的配置唯一来源。它负责加载、合并、验证和分发配置数据，但不负责任何业务逻辑。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `load_config` | 配置文件路径、环境变量 | `PraxisConfig` 对象 | Praxis 启动入口 |
| `get_component_config(name)` | 组件名称 | 对应组件的配置切片 | 各组件初始化时 |
| `reload_config` | — | 更新后的配置 + 变更事件 | 运行时热更新 |

#### 功能需求

**F1.1 分层配置加载**

- 配置来源优先级（从低到高）：默认值 → 配置文件（YAML）→ 环境变量 → 命令行参数
- 基于 Pydantic Settings 实现，所有配置项强类型、带校验规则
- 顶层配置对象 `PraxisConfig` 包含每个组件的配置切片

**F1.2 组件配置隔离**

- 每个组件拥有独立的配置 Section，命名空间隔离
- 组件仅能访问自身配置切片，不可越权访问其他组件配置
- 配置 Schema 由各组件自行定义，注册到配置系统

**F1.3 配置验证**

- 启动时全量校验所有配置项，校验失败拒绝启动并报告具体错误
- 支持自定义验证器：组件可注册跨字段验证规则
- 运行时配置变更同样必须通过验证

---

### 3.2 S2：遥测系统（Telemetry System）

**包路径**：`praxis.telemetry`
**依赖**：S1（配置）
**被依赖**：所有其他组件

#### 职责边界

遥测系统提供统一的可观测性基础设施，包括结构化日志、指标采集、分布式追踪和审计日志。它不做业务判断，只负责记录和传输遥测数据。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `get_logger(name)` | 模块名 | 结构化 Logger 实例 | 所有组件 |
| `emit_metric(name, value, tags)` | 指标名、值、标签 | — | 所有组件 |
| `start_span(name, parent?)` | Span 名、父 Span | Span 上下文 | S11(编排循环)、S5(工具) |
| `record_audit(event)` | 审计事件对象 | — | S8(护栏)、S5(工具)、S11(编排) |

#### 功能需求

**F2.1 结构化日志**

- 所有日志条目包含：时间戳、级别、组件名、会话 ID、轮次号、消息、结构化字段
- 日志级别配置化，支持按组件独立设置
- 日志输出支持 JSON 格式（机器可读）和人类可读格式

**F2.2 指标采集**

- 内置关键指标：每轮次 Prompt 大小、LLM 延迟、工具调用耗时、Token 消耗量、成功/失败计数
- 支持 Counter、Gauge、Histogram 三种指标类型
- 指标可导出为 Prometheus 格式或写入本地文件

**F2.3 分布式追踪**

- 主 Agent → 子代理 → 工具调用的完整调用链路可追踪
- 每个 Span 携带：组件名、操作类型、耗时、状态码
- 追踪数据支持 OpenTelemetry 协议导出

**F2.4 审计日志**

- 不可篡改的操作记录：每次工具调用（时间、工具名、参数摘要、结果、权限判定）
- 每次 LLM 调用（模型、Token 用量、是否触发护栏）
- 每次权限决策（允许/拒绝/需确认、触发规则）
- 审计日志独立于常规日志，持久化到 S3（持久化引擎）

---

### 3.3 S3：持久化引擎（Persistence Engine）

**包路径**：`praxis.persistence`
**依赖**：S1（配置）
**被依赖**：S5（工具）、S6（记忆）、S12（会话管理）

#### 职责边界

持久化引擎提供统一的存储抽象层，屏蔽底层存储差异。它只负责数据的序列化/反序列化和 CRUD 操作，不理解数据的业务含义。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `save(namespace, key, data)` | 命名空间、键、可序列化数据 | — | S6(记忆)、S12(会话管理) |
| `load(namespace, key)` | 命名空间、键 | 反序列化后的数据或 None | S6(记忆)、S12(会话管理) |
| `delete(namespace, key)` | 命名空间、键 | — | S6(记忆)、S12(会话管理) |
| `list_keys(namespace, prefix?)` | 命名空间、可选前缀 | 键列表 | S6(记忆)、S12(会话管理) |
| `save_checkpoint(session_id, state)` | 会话 ID、状态快照 | 检查点 ID | S12(会话管理) |
| `load_checkpoint(checkpoint_id)` | 检查点 ID | 状态快照 | S12(会话管理) |

#### 功能需求

**F3.1 多后端存储抽象**

- 统一接口，支持三种后端：
  - **SQLite**：默认本地后端，零配置启动
  - **Redis**：分布式部署场景
  - **文件系统**：简单场景，人类可读（JSON/YAML 文件）
- 后端通过配置切换，上层代码无需修改

**F3.2 检查点管理**

- 检查点包含完整 Agent 状态快照（消息历史、记忆状态、轮次计数器、工具状态）
- 支持按会话 ID 列出所有历史检查点
- 支持时间旅行：加载任意历史检查点恢复到该时间点
- 检查点数据自动序列化/反序列化（基于 Pydantic 模型）

**F3.3 命名空间隔离**

- 不同组件的数据通过命名空间隔离
- 同一组件的不同会话通过会话 ID 隔离
- 支持按命名空间批量清理过期数据

---

## 4. Layer 1 · 模型接入层

### 4.1 S4：模型网关（Model Gateway）

**包路径**：`praxis.gateway`
**依赖**：S1（配置）、S2（遥测）
**被依赖**：S6（记忆系统）、S7（上下文引擎）、S10（验证引擎）、S11（编排循环）
**核心三方库**：`litellm`（LiteLLM Python SDK）

#### 职责边界

模型网关是 Praxis 与 LLM 的唯一通道。底层基于 **LiteLLM** 实现，天然支持 100+ Provider 的统一调用格式（OpenAI 兼容），无需为每个 Provider 编写独立适配器。模型网关不理解对话语义，只负责请求路由、流式传输、Token 计量和故障转移。

S4 对 LiteLLM 的封装策略：**薄封装，不二次抽象**。S4 直接使用 `litellm.Router` 作为核心调用引擎，在其上叠加 Praxis 特有的遥测集成（S2）、配置驱动（S1）和预算管控逻辑，但不重新发明 Provider 适配、重试、负载均衡等 LiteLLM 已成熟解决的问题。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `chat(messages, tools?, config?)` | 消息列表、可选工具 Schema、推理参数 | `ModelResponse`（含 content、tool_calls、usage） | S11(编排循环) |
| `chat_stream(messages, tools?, config?)` | 同上 | `AsyncIterator[ModelResponseChunk]` | S11(编排循环) |
| `summarize(content, instruction)` | 待摘要内容、摘要指令 | 摘要文本 | S7(上下文引擎) |
| `judge(criteria, content)` | 评估标准、待评估内容 | 评估结果 | S10(验证引擎) |
| `get_token_count(messages)` | 消息列表 | Token 数量 | S7(上下文引擎) |
| `get_max_tokens(model?)` | 可选模型名 | 模型最大 Token 数 | S7(上下文引擎) |
| `completion_cost(response)` | 模型响应对象 | 成本（USD） | S2(遥测) |

#### 功能需求

**F4.1 LiteLLM 集成与多 Provider 统一**

- 底层使用 `litellm.Router` 管理所有模型部署，通过 `model_list` 配置声明可用模型及其 Provider 参数
- 模型配置通过 S1（配置系统）加载，格式与 LiteLLM `model_list` 对齐：

```
model_list:
  - model_name: "default"           # Praxis 内部模型别名
    litellm_params:
      model: "anthropic/claude-sonnet-4-20250514"
      api_key: "env/ANTHROPIC_API_KEY"
  - model_name: "default"           # 同名 = 自动负载均衡
    litellm_params:
      model: "openai/gpt-4o"
      api_key: "env/OPENAI_API_KEY"
  - model_name: "fast"              # 低延迟模型
    litellm_params:
      model: "anthropic/claude-haiku"
  - model_name: "local"             # 本地模型
    litellm_params:
      model: "ollama/llama3"
      api_base: "http://localhost:11434"
```

- 同一 `model_name` 下的多个部署自动负载均衡（LiteLLM Router 内置）
- 支持 100+ Provider 开箱即用：OpenAI、Anthropic、Azure、Bedrock、Vertex AI、Ollama、vLLM、Deepseek 等
- 请求/响应格式统一为 OpenAI 兼容格式，LiteLLM 内部自动完成 Provider 特定的参数转换
- 通过 `litellm.register_model()` 支持运行时动态注册新模型

**F4.2 流式传输**

- 通过 `litellm.Router.acompletion(stream=True)` 实现异步流式输出
- 流式传输中的工具调用识别和增量解析（LiteLLM 内部处理 Provider 差异）
- 流中断时的部分结果保留

**F4.3 Token 计量与预算**

- **Token 计数**：使用 `litellm.token_counter()` 进行预调用 Token 估算，支持各 Provider 特定的 tokenizer（OpenAI tiktoken、Anthropic、Cohere、Llama 等），未知模型自动回退到 tiktoken
- **用量记录**：每次调用从 `ModelResponse.usage` 中提取精确的输入/输出/总 Token 数，通过 S2（遥测）记录
- **成本追踪**：使用 `litellm.completion_cost()` 从社区维护的模型价格表自动计算每次调用成本（USD），通过 S2（遥测）发射 `llm_cost` 指标
- **预算管控**：调用前预估 Token 消耗，逼近预算时通知 S11（编排循环）
- **窗口查询**：通过 `litellm.get_max_tokens()` 查询任意模型的最大上下文窗口大小，供 S7（上下文引擎）压缩决策使用

**F4.4 路由、重试与故障转移**

- **负载均衡**：`litellm.Router` 内置多种路由策略——最少延迟（latency-based）、最少使用量（usage-based）、速率限制感知（rate-limit-aware）、最低成本（cost-based），通过配置选择
- **自动重试**：请求失败时 LiteLLM Router 自动重试（可配置 `num_retries`、指数退避），跨部署重试时自动选择不同区域/Provider
- **故障转移**：主 Provider 不可用时自动 fallback 到同 `model_name` 下的其他部署。支持跨 `model_name` 的显式 fallback 链配置
- **冷却机制**：失败的部署自动进入 cooldown 期间，避免反复调用故障端点
- **所有路由事件**（切换、重试、cooldown）通过 S2（遥测）记录

**F4.5 可观测性集成**

- 通过 `litellm.success_callback` / `litellm.failure_callback` 注册自定义回调，将调用详情（模型、延迟、Token、成本、错误）桥接到 S2（遥测）
- 每次调用自动发射结构化指标：`llm_latency`、`llm_tokens_input`、`llm_tokens_output`、`llm_cost`、`llm_error`
- 支持可选集成外部可观测平台（Langfuse、MLflow 等），通过 LiteLLM 内置 callback 机制一行配置启用

**F4.6 异常标准化**

- LiteLLM 将所有 Provider 的错误映射为 OpenAI 兼容异常类型（`AuthenticationError`、`RateLimitError`、`APIError` 等）
- S4 将 LiteLLM 异常转换为 Praxis 内部异常体系，供 S9（错误恢复）进行分类和策略决策

---

## 5. Layer 2 · 核心能力层

### 5.1 S5：工具系统（Tool System）

**包路径**：`praxis.tools`
**依赖**：S1（配置）、S2（遥测）、S3（持久化）
**被依赖**：S7（上下文引擎）、S10（验证引擎）、S11（编排循环）、S14（技能系统）

#### 职责边界

工具系统管理 Agent 可用的所有工具——注册、发现、Schema 管理和执行。它负责"能做什么"和"怎么做"，但不负责"是否允许做"（权限由 S8 护栏系统判定，调用流程由 S11 编排循环协调）。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `register_tool(tool_def)` | 工具定义（名称、描述、Schema、处理函数） | — | 启动时 / 运行时动态注册 |
| `get_tool_schemas(filter?)` | 可选过滤条件（类别、标签） | 工具 Schema 列表（LLM 可注入格式） | S7(上下文引擎) |
| `execute_tool(name, arguments)` | 工具名、参数字典 | `ToolResult`（成功结果或错误信息） | S11(编排循环) |
| `discover_mcp_server(server_config)` | MCP 服务器配置 | 发现的工具/资源/提示列表 | 启动时 / 运行时 |
| `get_tool_metadata(name)` | 工具名 | 工具元数据（类别、权限需求、只读/写） | S8(护栏)、S11(编排) |
| `read_mcp_resource(server, uri)` | MCP 服务器标识、资源 URI | 资源内容（文本/二进制 + MIME） | S7(上下文引擎)、S14(技能) |
| `list_mcp_resources(server, filter?)` | MCP 服务器标识、可选过滤 | 资源描述符列表 | S7(上下文引擎)、S14(技能) |
| `get_mcp_prompt(server, name, args?)` | MCP 服务器、提示名、可选参数 | 完整提示模板消息列表 | S11(编排循环)、S14(技能) |
| `handle_mcp_elicitation(request)` | MCP 服务器信息请求 | 用户响应或取消 | S11(编排循环) |
| `handle_mcp_sampling(request)` | MCP 服务器 LLM 补全请求 | 模型生成结果 | S4(模型网关)、S11(编排) |

#### 功能需求

**F5.1 工具注册表**

- 基于注册表模式的工具架构，所有工具通过注册表统一管理
- 工具定义包含：名称、描述（LLM 可理解）、参数 Schema（JSON Schema / Pydantic）、返回类型、元数据（类别、权限级别、只读/写标记）
- 内置工具类别：
  - **文件操作类**：`read_file`、`write_file`、`edit_file`、`list_dir`
  - **搜索类**：`grep_search`、`find_by_name`、`code_search`
  - **Shell 执行类**：`run_command`、`run_command_background`
  - **网络类**：`web_fetch`、`web_search`
  - **系统信息类**：`get_system_info`
  - **自治管理类**：`update_plan`、`update_notes`、`ask_user`、`attempt_completion`
- 支持运行时动态注册和注销工具

**F5.2 MCP 完整集成**

Praxis 作为 MCP Host，为每个连接的 MCP Server 创建独立的 MCP Client 实例。MCP 集成基于 JSON-RPC 2.0 有状态会话协议，完整支持协议定义的三大服务器原语和三大客户端特性。

**F5.2.1 传输层**

- **Stdio 传输**：本地 MCP 服务器，通过子进程 stdin/stdout 通信，单客户端独占连接
- **Streamable HTTP 传输**：远程 MCP 服务器，支持多客户端并发、SSE 流式传输、会话恢复
- 传输层抽象：上层逻辑无感传输协议差异，MCP Client 统一接口

**F5.2.2 MCP 服务器原语——Tools（模型控制）**

- 通过 `tools/list` 发现服务器提供的工具集，获取 JSON Schema 定义
- 通过 `tools/call` 执行具体工具，返回结构化结果（文本、图片、音频、资源链接）
- MCP 工具与内置工具在注册表中统一管理，调用方无需区分来源
- 监听 `notifications/tools/list_changed` 通知，动态更新可用工具集
- 工具执行前通过 S8（护栏）进行权限检查，支持用户审批机制

**F5.2.3 MCP 服务器原语——Resources（应用控制）**

- 通过 `resources/list` 发现服务器提供的只读数据源（文件、API 响应、数据库 Schema 等）
- 每个 Resource 通过 URI（如 `file:///path`、`git://repo`、`https://api/data`）唯一标识，声明 MIME 类型
- 支持 **直接资源**（固定 URI）和 **资源模板**（参数化 URI，如 `weather://forecast/{city}/{date}`）
- 通过 `resources/read` 读取具体资源内容，返回文本或 base64 编码数据
- 支持 `resources/subscribe` 订阅资源变更通知，实现实时上下文更新
- Resources 为**应用控制**：Praxis 决定何时检索、如何处理（嵌入向量搜索、全文注入、智能筛选）
- S7（上下文引擎）可将 MCP Resources 作为上下文源注入到模型请求中

**F5.2.4 MCP 服务器原语——Prompts（用户控制）**

- 通过 `prompts/list` 发现服务器提供的预定义提示模板
- 通过 `prompts/get` 获取完整提示模板（参数化消息序列）
- Prompts 为**用户控制**：需要显式调用（如斜杠命令 `/plan-vacation`），不自动触发
- MCP Prompts 可引用 Resources 和 Tools，形成完整工作流
- Prompts 支持参数补全（Completion），帮助用户发现有效参数值
- Prompts 与 S14（技能系统）协同：MCP Prompts 提供服务器级工作流模板，Skills 提供客户端级程序化知识

**F5.2.5 MCP 客户端特性——Elicitation（信息征询）**

- 当 MCP Server 需要用户提供额外信息时，通过 `elicitation/create` 发起结构化请求
- Praxis 将 Elicitation 请求转发给用户界面（CLI/API），用户可提供信息、拒绝或取消
- 支持 Form Mode（结构化表单）和 URL Mode（外部 URL 跳转，适用于 OAuth 流程）
- 响应通过 JSON Schema 验证后返回服务器
- 安全约束：不请求密码/API Key，用户审查数据后发送

**F5.2.6 MCP 客户端特性——Sampling（LLM 采样）**

- 当 MCP Server 需要 LLM 推理能力时，通过 `sampling/createMessage` 请求 Praxis 代理调用
- Praxis 将请求转发给 S4（模型网关），确保 Server 无需直接集成 LLM
- 支持 Human-in-the-loop：用户可审核和修改请求/响应
- Server 可指定模型偏好（智能优先、速度优先、成本优先），Praxis 映射到实际可用模型
- Sampling 请求在独立上下文中执行，与主 Agent 循环隔离
- 支持 Sampling with Tools：Server 可在采样请求中提供工具，Praxis 编排多轮工具调用循环

**F5.2.7 MCP 客户端特性——Roots（作用域）**

- 通过 Roots 向 MCP Server 声明 Praxis 的工作目录范围
- Server 可通过 `roots/list` 查询允许操作的目录，限制自身访问范围
- Roots 变更时通过 `notifications/roots/list_changed` 通知 Server

**F5.2.8 MCP 服务器连接管理**

- 完整的初始化→操作→关闭连接管理
- 初始化阶段：能力协商（Capability Negotiation），客户端和服务器交换支持的特性集
- MCP Server 崩溃自动重连（可配置重连策略）
- 会话状态通过 S3（持久化）检查点保存，支持恢复
- 每个 MCP Client 实例维护独立安全边界，Server 之间隔离

**F5.2.9 MCP 授权**

- 远程 MCP Server 支持 OAuth 2.0 / PKCE 授权流程
- 支持 Protected Resource Metadata 发现、Authorization Server Discovery
- Token 管理：存储、刷新、过期处理
- 支持通过 Elicitation URL Mode 完成用户在线授权

**F5.2.10 MCP Tasks（实验性）**

- 支持 MCP Tasks 原语，用于长时间运行的服务器端操作
- Task 状态生命周期：pending → running → completed/failed/cancelled
- 支持 Task 进度通知、取消、结果检索
- Task 可与 Elicitation 和 Sampling 组合，实现复杂的服务器端工作流

**F5.3 工具执行管线**

- 完整的工具执行管线（S11 编排循环依次调用）：
  1. **参数验证**：基于 Schema 校验参数类型和格式，无效参数立即返回错误
  2. **沙箱执行**：工具处理函数在受控环境中执行
  3. **结果捕获**：标准化返回 `ToolResult`（成功值或错误信息）
  4. **结果格式化**：清理技术细节，格式化为 LLM 可读的观察结果
- **并发策略**：只读工具（`read_file`、`grep_search`）可并发执行；写工具（`write_file`、`run_command`）串行执行
- 工具执行超时可配置（默认 30s），超时返回超时错误

**F5.4 工具覆盖机制**

- 允许同名自定义工具覆盖内置工具
- 覆盖后必须保持兼容的 Schema 接口契约
- 覆盖关系在注册表中可查询和追溯

**F5.5 沙箱执行环境**

- 文件系统访问限制：工具只能访问工作目录白名单内的路径
- Shell 命令执行超时限制（可配置）
- 网络访问可配置出站规则
- 进程隔离：Shell 命令通过 `asyncio.create_subprocess_exec` 执行，独立进程

---

### 5.2 S6：记忆系统（Memory System）

**包路径**：`praxis.memory`
**依赖**：S1（配置）、S2（遥测）、S3（持久化）、S4（模型网关）
**被依赖**：S7（上下文引擎）、S12（会话管理）

#### 职责边界

记忆系统管理 Agent 在不同时间尺度和认知维度上的信息存储、提取、整合和检索。它负责"记住什么"、"如何存取"和"如何进化记忆"，但不负责"何时注入上下文"（由 S7 上下文引擎决定）。

记忆系统借鉴认知科学的记忆分类模型（CoALA 框架），将记忆划分为四种认知类型：语义记忆（事实与知识）、情景记忆（经历与交互）、程序记忆（工作流与技能模式）和工作记忆（当前会话上下文）。同时引入**模型辅助管线**，通过 S4（模型网关）驱动记忆的自动提取和智能整合。

> *"Context window ≠ memory。RAG alone is not enough。持久化的多类型记忆 + 自改进循环 = 2026 年智能体记忆的新标准。"*

#### 公开接口契约

S6 对外以单一门面 `CognitiveMemory` 暴露下列接口。所有接口均为 async（除 `get_message_history`/`export_state` 同步取值外）。

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `append_message(message)` | `WorkingMemoryMessage` | — | S11(编排循环) |
| `get_message_history(limit?)` | 可选数量限制 | 消息列表 | S7(上下文引擎) |
| `search_memory(query, scopes?, memory_type?, tags?, top_k?)` | 查询文本；可选作用域列表 / 认知类型 / 标签 / 返回数量 | 记忆搜索结果列表（按相关性排序） | S7(上下文引擎)、S11(编排循环) |
| `save_memory(content, scope?, memory_type?, tags?, metadata?)` | 记忆内容；可选作用域、类型、标签、元数据 | 记忆 ID | S11(编排循环) 通过工具 |
| `update_memory(memory_id, content)` | 记忆 ID、更新内容 | — | S11(编排循环)、后台整合 |
| `delete_memory(memory_id)` | 记忆 ID | — | S11(编排循环) |
| `get_memory_index(scopes?)` | 可选作用域列表 | 轻量索引列表（~150 字符/条） | S7(上下文引擎) |
| `load_memory_detail(scope, memory_id)` | 作用域、记忆 ID | 完整记忆条目或 None | S7(上下文引擎) |
| `get_profile(scope, schema_name)` | 作用域、档案名 | 结构化档案字典或 None | S11、S7 |
| `update_profile(scope, schema_name, fields)` | 作用域、档案名、字段字典 | — | S11 通过工具 |
| `run_dream(scopes)` | 作用域列表 | 整理报告 | S12、手动触发 |
| `read_scratchpad(key)` | 草稿键名（progress/todos/features） | 草稿内容或 None | S7、S12 |
| `write_scratchpad(key, content)` | 草稿键名、内容 | — | S11 通过工具 |
| `clear_session()` | — | — | S12(会话管理) |
| `export_state()` | — | 可序列化的状态快照（含消息游标） | S12(会话管理) |
| `import_state(snapshot)` | 状态快照 | — | S12(会话管理) |
| `start()` / `stop()` | — | — | **仅 S12 生命周期管理**调用，用于后台任务的启停 |

> **接口使用契约**：`start()` / `stop()` 是内部生命周期方法，由 S12（`SessionFactory.create_session` 和 `Session.terminate`）负责调用；Agent 与业务代码**不应**直接调用。所有其他接口在实例构造后立即可用（后台特性由 `start()` 驱动，未启动时记忆写入仍立即生效，但不触发异步提取）。

#### 功能需求

**F6.1 认知记忆类型**

记忆系统基于认知科学四类记忆模型，每种类型有独立的存储策略和检索特性：

| 类型 | 含义 | 存储内容 | 检索策略 | 示例 |
|------|------|---------|---------|------|
| **语义记忆** | 事实与知识 | 用户偏好、项目规范、技术事实、实体关系 | 语义搜索 + 元数据过滤 | "用户偏好 Python 开发"、"项目使用 PostgreSQL" |
| **情景记忆** | 经历与交互 | 会话摘要、关键决策、问题解决过程 | 时间线检索 + 语义搜索 | "上次会话修复了认证模块的 Bug" |
| **程序记忆** | 工作流与模式 | 用户惯用流程、工具使用模式、代码风格 | 场景匹配 + 语义搜索 | "PR 审查流程：lint → test → review → merge" |
| **工作记忆** | 当前会话上下文 | 消息序列、中间推理、临时状态 | 直接访问（内存） | 当前对话的完整消息历史 |

- **语义记忆**支持两种模式：
  - **集合模式（Collection）**：无界知识存储，每条记忆独立文档，存储时经 `save_memory` 走整合判定，检索时经 `search_memory` 走语义搜索
  - **档案模式（Profile）**：每 `(scope, schema_name)` 对应唯一档案文档，字段字典就地合并更新，通过 `get_profile` / `update_profile` 专用接口访问；档案不参与 `search_memory` 的语义检索，但出现在 `get_memory_index` 中
- **情景记忆**包含结构化字段：`context_description`（情境上下文）、`reasoning`（推理过程）、`action_taken`（采取行动）、`outcome`（达成结果）
- **程序记忆**包含结构化字段：`steps`（步骤列表）、`applicable_scenarios`（适用场景列表）
- **工作记忆**为进程内 `WorkingMemory` 对象，维护消息序列及 `max_messages` 裁剪；不持久化为独立记忆条目，但随检查点快照一起 export/import

**F6.2 模型辅助记忆管线**

记忆的提取和整合由 LLM 驱动（通过 S4 模型网关），而非简单的规则匹配或全量存储。

**F6.2.1 记忆提取（Extraction）**

- **触发时机**：`append_message` 写入后设置信号，由内部后台任务消费（见 F6.3.1）。不再按"每轮结束"触发
- LLM 分析对话内容，区分**有意义的洞察**（值得长期存储）和**例行对话**（不需存储）
- 单次对话可提取多条不同类型的记忆
- 每种认知类型使用独立的提取提示（Extraction Prompt），返回对应子类的结构化字段：
  - **语义提取** → `SemanticMemory`（content + tags + confidence）
  - **情景提取** → `EpisodicMemory`（content + context_description + reasoning + action_taken + outcome）
  - **程序提取** → `ProceduralMemory`（content + steps + applicable_scenarios）
- 提取提示可通过 `MemoryConfig.extraction_prompts` 配置覆盖
- 所有提取结果公共字段：内容、摘要、认知类型、作用域、元数据标签、时间戳、置信度

**F6.2.2 记忆整合（Consolidation）**

- 新提取的记忆不直接写入存储，先经过智能整合判定
- 对每条新记忆，在**同作用域、同类型**下通过语义搜索检索最相似的已有记忆（相似度阈值可配置，默认 0.75）
- 若无相似记忆 → 直接 ADD；若有 → LLM 评估做出决策：
  - **ADD**：新信息与已有记忆不同，新增存储
  - **UPDATE**：新信息补充或更新已有记忆，合并后旧版标记 SUPERSEDED、新版继承 `version+1`
  - **NOOP**：新信息冗余，无需操作
- **冲突解决**：当新信息与旧信息矛盾时，旧版标记 `SUPERSEDED` 并写入 `superseded_by` 指针；仅"长期未访问且相关性过低"的孤立记忆由衰减扫描标记为 `INACTIVE`
- **语义去重**：语义等价的记忆合并（如"喜欢 pizza"和"爱吃 pizza"视为相同信息）
- 整合过程将旧条目的版本快照写入 `memory_versions` 命名空间，构成不可变审计日志
- 整合失败时，保守策略为直接 ADD（防止信息丢失）；失败消息保留在 pending 队列供下次消费

**F6.2.3 记忆梦境整理（Dream Consolidation）**

借鉴 Anthropic Claude Code 的 Auto Dream 机制和人类 REM 睡眠的概念，由 S6 内部定时调度器执行：

- **触发条件**：距上次整理 >24h **且** 自上次 dream 以来完成的会话数 ≥5（两者均可通过 `MemoryConfig` 配置），或通过 `run_dream(scopes)` 手动触发
- **会话计数**：`clear_session()` 内部每次被调用时 dream 会话计数器 +1，该计数随 `export_state` 持久化
- **调度器**：`CognitiveMemory.start()` 启动一个独立 `asyncio.Task`，按 `dream_check_interval_seconds`（默认 3600）周期性检查触发条件并自动调用 `run_dream`
- **LLM 驱动的整理动作**：
  - **时间锚定**：将模糊时间引用替换为具体日期（"昨天的部署问题" → "2026-04-15 部署问题"）
  - **矛盾消解**：检测并解决互相矛盾的记忆条目
  - **陈旧清理**：标记引用已不存在的文件、已完成的任务等过时记忆为 `INACTIVE`
  - **索引精简**：建议合并冗余条目
- 整理报告通过 S2（遥测）记录：整理条目数、标记陈旧数、合并数、耗时

**F6.3 双路径处理与后台自治**

记忆系统支持两种处理路径，平衡实时性和完整性：

| 维度 | 热路径（Agent 驱动） | 后台路径（内部自治） |
|------|------|------|
| **触发时机** | Agent 显式调用 `save_memory` / `update_memory` | `append_message` 写入时设信号 |
| **执行方式** | 同步走整合链路写入 | 独立 asyncio.Task 异步批量消费 |
| **延迟影响** | 影响该工具调用延迟 | 不影响 `append_message` 延迟 |
| **适用场景** | 关键信息即时保存（用户显式要求记住） | 全量对话洞察分析、增量整合 |
| **外部可见性** | 通过 `save_memory` 等接口调用 | 完全透明，Agent 无感知 |

- **热路径**：Agent 在编排循环中通过 `save_memory`/`update_memory`/`delete_memory` 工具主动写入，所有写入均走整合链路
- **后台路径**：由 S6 内部异步任务驱动，Agent 无需感知；仅 S12 生命周期管理负责其启停

**F6.3.1 后台自治机制（内部实现）**

后台管线由 S6 内部拥有的 `BackgroundWorker` 管理，生命周期与会话绑定，启停由 S12 在 `create_session`/`terminate` 阶段统一调用 `CognitiveMemory.start()` / `stop()`。Agent 与业务代码**不直接接触** start/stop。

```
生命周期调用（仅 S12）：
    factory.create_session(..., memory=ms)
        └─ ms.start()  ── 启动后台 Worker 与 Dream 调度器

    session.terminate()
        └─ ms.stop()   ── 等待当前整合完成 → 取消后台任务

Agent / 业务调用方视角（S11 只做这些）：
    ms.append_message(user_msg)       ─── 注入消息（同步、立即返回）
    ms.append_message(assistant_msg)
    ms.save_memory(content, ...)       ─── 热路径直写
    ms.search_memory(query, ...)       ─── 检索
    ms.export_state() / import_state() ── 检查点保存/恢复
    ms.clear_session()                 ─── 会话级清理（不停止 Worker）

S6 内部自动执行：
    append_message() 内部触发：
        └─ 将消息写入工作记忆
        └─ 推入 pending 队列（保留 message_id 指针，非清空-重建）
        └─ 设置 new_message_signal

    BackgroundWorker 主循环（asyncio.Task）：
        loop:
            await new_message_signal 或超时
            if pending_count < batch_threshold:
                continue
            # 从游标取快照但不清空 pending（消费完成后按 message_id 推进游标）
            snapshot = pending[cursor:]
            try:
                extracted = await extract(snapshot)   # LLM (S4)
                await consolidate_batch(extracted)    # LLM (S4)
                cursor = snapshot[-1].message_id      # 成功后才推进游标
            except Exception:
                emit_metric("memory_background_error")
                # pending 保留，下次循环重试

    DreamScheduler 主循环（独立 asyncio.Task）：
        loop:
            sleep(dream_check_interval_seconds)
            if should_run():
                await run_dream(configured_scopes)
```

- **消息游标**：`BackgroundWorker` 维护 `last_processed_message_id`（字符串，非索引）。`export_state` 包含游标；`import_state` 恢复游标后，若 Worker 已启动则继续按游标消费
- **触发频率**：默认每条消息写入后触发信号；`background_batch_threshold`（默认 3）控制批量大小
- **背压控制**：pending 队列仅追加，只在成功消费后按游标推进，杜绝"快照-清空"之间的丢消息竞态
- **优雅关闭**：`stop()` 设置 shutdown 事件 → 等待当前处理完成 → 取消任务
- **失败容错**：单批提取/整合失败时 pending 保留，下次循环重试；错误指标写入 S2
- **会话级清理 vs 生命周期终止**：`clear_session` 只清空工作记忆和 pending 队列（不停 Worker）；`stop` 停止 Worker 与 Dream 调度器

**F6.4 多作用域记忆隔离**

记忆按作用域组织，支持复合查询和精确检索：

| 作用域 | 含义 | 生命周期 | 示例 |
|--------|------|---------|------|
| `session/<id>` | 单次会话/工作流 | 会话结束后归档 | 当前任务的中间决策 |
| `project/<name>` | 项目级别 | 项目存续期间 | 项目规范、编码风格、架构约束 |
| `user/<id>` | 用户级别 | 跨所有会话持久化 | 用户偏好、工作习惯 |
| `global` | 全局共享 | 永久 | 通用知识、组织规范 |

- 检索时作用域可复合：如"检索当前项目中当前用户的所有语义记忆"
- **项目级记忆预加载**：如果项目根目录存在 `praxis.md`，`CognitiveMemory.start()` 在首次启动时自动加载为 `project/<name>` 作用域的语义记忆。格式约定：
  - 文件顶部可选 YAML frontmatter（`---` 分隔），字段包括 `project`（覆盖默认名称）、`tags`
  - 正文每个 `## <heading>` 二级标题段作为独立记忆条目，heading 作为 summary，段内容作为 content
  - 预加载条目 `metadata.source = "project_praxis_md"`，避免重复导入
- 作用域之间严格隔离，`ScopedStore.query` 仅返回明确指定的作用域，防止跨项目/跨用户泄露
- 每条记忆携带结构化元数据（标签、分类、来源），支持元数据过滤检索

**F6.5 混合检索架构**

- **语义搜索**：通过向量嵌入（LiteLLM embedding via S4）进行余弦相似度检索，默认检索路径
- **元数据过滤**：按作用域、认知类型、标签、时间范围过滤
- **综合重排序**：向量搜索召回候选集后，以加权公式重排：
  `final_score = 0.6·semantic + 0.2·freshness + 0.1·access_popularity + 0.1·confidence`
  其中 freshness 按更新时间指数衰减，access_popularity 按 `access_count` 归一化
- **渐进式检索**：三层结构确保高效访问
  - **轻量索引**（`get_memory_index`，~150 字符/条，始终加载到系统提示）
  - **语义结果**（`search_memory`，含 relevance_score 与 source）
  - **完整内容**（`load_memory_detail`，按需加载原始详细内容，同时累计 `access_count`）
- 检索结果按相关性排序，包含：记忆内容、来源作用域、时间戳、置信度、综合评分

**F6.6 工作记忆（Scratchpad）**

- Agent 主动维护的结构化笔记，通过 S3 键值存储持久化（命名空间 `scratchpad`，键前缀 `<session_id>:`），不占上下文 Token 预算
- **已知键白名单**（`Scratchpad.KNOWN_KEYS`，强校验）：
  - `progress.json`：已完成工作列表、当前步骤、下一步计划
  - `todos.json`：结构化任务跟踪（ID、描述、状态、优先级）
  - `features.json`：长期项目的功能清单（描述、步骤、pass/fail 状态）
- `write_scratchpad` 对未在白名单中的 key 抛出 `ValueError`
- 上下文重置后，Agent 可通过读取草稿快速恢复工作状态

**F6.7 记忆即提示（Memory-as-Hint）**

- 记忆系统返回的所有记忆均视为**提示而非事实**
- Agent 在使用记忆指导行动前，应通过实际验证（如读取代码、运行测试）确认记忆准确性
- 该原则在系统提示中显式声明，降低因记忆陈旧导致的幻觉行为
- 每条记忆附带时间戳和置信度，辅助 Agent 判断可信程度

**F6.8 记忆生命周期**

- **时间衰减**：记忆的相关性评分随时间递减（可配置衰减曲线）
- **动态遗忘**：低相关性、长期未检索的记忆标记为 INACTIVE，不参与常规检索
- **版本历史**：关键事实的更新保留版本链（如"用户使用 PostgreSQL" → "用户迁移到 MySQL"），支持审计追溯
- **不可变审计**：记忆不物理删除，仅标记状态（ACTIVE/INACTIVE/SUPERSEDED），完整变更历史可查
- 记忆使用统计：通过 S2（遥测）记录检索频次、命中率、提取/整合吞吐量

---

### 5.3 S7：上下文引擎（Context Engine）

**包路径**：`praxis.context`
**依赖**：S1（配置）、S2（遥测）、S4（模型网关）、S5（工具系统）、S6（记忆系统）、S14（技能系统）
**被依赖**：S11（编排循环）、S12（会话管理）

#### 职责边界

上下文引擎是 Harness 中最关键的组件之一。它负责在每一步为模型组装最优上下文——决定模型"看到什么"以及"何时看到"。它消费 S5（工具 Schema）、S6（记忆内容）和 S4（Token 计数/压缩摘要），产出可直接送入 LLM 的完整 Prompt。

> *研究表明：关键内容落入窗口中部位置时模型性能下降 30%+。上下文必须被视为具有递减边际回报的有限资源。*

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `assemble_prompt(turn_context)` | 当前轮次上下文（用户消息、系统指令覆盖等） | 完整 Prompt（消息列表 + 工具定义列表） | S11(编排循环) |
| `update_with_result(tool_results)` | 工具执行结果列表 | — | S11(编排循环) |
| `update_with_response(assistant_msg)` | 助手响应消息 | — | S11(编排循环) |
| `trigger_compaction()` | — | 压缩后的上下文状态 | S11(编排循环)、自动触发 |
| `get_token_usage()` | — | 当前上下文 Token 用量和窗口剩余 | S11(编排循环) |
| `export_state()` | — | 可序列化的上下文状态 | S12(会话管理) |

#### 功能需求

**F7.1 分层 Prompt 组装**

- 按优先级栈组装（从高到低）：
  1. 系统提示（System Prompt）：Harness 行为指令
  2. 工具定义：从 S5 获取当前可用工具 Schema
  3. 开发者指令：应用级别的自定义指令
  4. 用户指令：级联配置文件中的用户偏好（大小上限可配置）
  5. 持久化记忆索引：从 S6 获取记忆索引（`get_memory_index`）
  6. 工作记忆：从 S6 获取工作记忆内容（`get_message_history`）
  7. 语义检索：从 S6 按任务相关性检索记忆（`search_memory`）
  8. 当前用户消息
- 使用 XML 标签或 Markdown 标题明确分隔各区段
- 追求最小完整信息集：足够引导行为，同时避免冗余
- 关键上下文放置在首尾位置（"Lost in the Middle" 效应对策）

**F7.2 动态工具集注入**

- 根据当前任务阶段从 S5 过滤工具集，仅注入相关工具 Schema
- 支持工具分组和懒加载：初始只加载核心工具，按需加载扩展工具
- 工具 Schema 按 LLM 原生格式（function calling）注入

**F7.3 上下文压缩（Compaction）**

- **触发条件**：当前 Token 用量超过窗口限制的可配置阈值（默认 80%）
- **压缩策略**：调用 S4（模型网关）的 `summarize` 接口，将消息历史压缩为高保真摘要
- **保留规则**：架构决策、未解决 Bug、实现细节保留；冗余工具输出、重复消息丢弃
- 压缩后保留最近 5 个关键文件引用
- **工具结果清除**：最安全的轻量级压缩——清除历史深处的工具原始输出，保留工具调用记录

**F7.4 观察遮蔽（Observation Masking）**

- 隐藏旧的工具输出内容，同时保持工具调用记录可见
- 遮蔽策略可配置：按轮次距离（默认 >10 轮）、按输出大小（默认 >2000 Token）

**F7.5 即时检索（Just-in-Time Retrieval）**

- 维护轻量级标识符（文件名、函数签名），仅在需要时动态加载完整内容
- 支持 RAG：在 Agent 需要特定信息时，通过 S5 的搜索工具检索并注入
- Few-shot 示例管理：维护精选的多样化示例集，按任务类型索引和按需注入

---

### 5.4 S14：技能系统（Skill System）

**包路径**：`praxis.skills`
**依赖**：S1（配置）、S2（遥测）、S3（持久化）、S5（工具系统）
**被依赖**：S7（上下文引擎）、S11（编排循环）、S12（会话管理）

#### 职责边界

技能系统管理 Agent 的**程序化知识**——将领域专长、工作流程和最佳实践封装为可发现、可组合、可复用的技能单元。技能（Skill）是"教 Agent 如何做事"的知识包，而工具（Tool）是"Agent 可以做什么"的能力。技能系统与工具系统协同：技能教 Agent 何时和如何使用工具，工具提供实际执行能力。

> *"构建 Agent 的技能就像为新员工准备入职指南。不是为每个用例构建碎片化的定制 Agent，而是通过封装和共享程序化知识，让任何人都可以用可组合的能力来专门化他们的 Agent。"* —— Anthropic

技能系统不执行工具（由 S5 执行），不控制编排循环（由 S11 控制），不管理上下文组装（由 S7 管理）。它只负责技能的发现、加载、统一管理和向上游组件提供技能内容。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `discover_skills(paths)` | 技能搜索路径列表 | 已发现技能的元数据列表 | 启动时 / S12(会话管理) |
| `get_skill_index()` | — | 所有已安装技能的轻量索引（名称+描述） | S7(上下文引擎) |
| `load_skill(skill_id)` | 技能标识符 | 完整技能内容（SKILL.md 主体） | S11(编排循环)、S7(上下文引擎) |
| `load_skill_file(skill_id, filename)` | 技能标识符、文件名 | 附属文件内容 | S11(编排循环) |
| `register_skill(skill_def)` | 技能定义 | — | 启动时 / 运行时 |
| `unregister_skill(skill_id)` | 技能标识符 | — | 运行时 |
| `list_skill_tools(skill_id)` | 技能标识符 | 技能附带的可执行脚本列表 | S5(工具系统) |
| `evaluate_relevance(task_desc, skill_index)` | 当前任务描述、技能索引 | 相关技能排序列表 | S11(编排循环) |

#### 功能需求

**F14.1 技能格式与结构**

技能采用 **Agent Skills 开放标准**（agentskills.io），格式简单、跨平台可移植：

- 每个技能是一个目录，包含一个 `SKILL.md` 文件
- `SKILL.md` 以 YAML frontmatter 开头，包含必要元数据：

  ```yaml
  ---
  name: pdf-processor
  description: 处理 PDF 文件：表单填写、内容提取、格式转换
  ---
  ```

- `SKILL.md` 正文包含详细的指令、工作流步骤和最佳实践
- 技能目录可包含附属文件（参考文档、脚本、模板），由 `SKILL.md` 引用
- 技能可包含可执行代码（Python 脚本），Agent 可根据需要执行

**F14.2 渐进式披露（Progressive Disclosure）**

技能系统的核心设计原则——像组织良好的手册一样，按需逐层加载信息：

- **第一层（始终加载）**：所有已安装技能的 `name` + `description`，约 150 字符/条目，预加载到系统提示中
- **第二层（按需触发）**：当 Agent 判断技能相关时，加载完整 `SKILL.md` 正文到上下文
- **第三层（深度探索）**：`SKILL.md` 引用的附属文件，仅在 Agent 需要具体细节时加载
- 渐进式披露使技能系统可扩展——可打包的上下文量实质上不受限制
- S7（上下文引擎）负责将技能内容注入到上下文的最优位置

**F14.3 技能发现与安装**

- **本地技能目录**：从配置路径（如 `~/.praxis/skills/`、项目 `.praxis/skills/`）扫描
- **技能市场/插件**：通过插件系统从官方和社区仓库安装技能
- **版本控制共享**：团队通过 Git 仓库共享项目级技能
- **MCP 技能发现**：通过 S5 的 MCP 集成从 MCP Server 发现可用技能（Skills Over MCP）
- 技能安装时进行安全审计：检查代码依赖、外部网络连接、敏感操作指令

**F14.4 技能触发与激活**

- **自动触发**：Agent 根据当前任务上下文和技能索引自动判断是否激活技能
- **手动触发**：用户通过显式命令加载特定技能
- 触发逻辑基于技能的 `name` 和 `description` 与当前任务的语义匹配
- 同时可激活多个技能，Agent 自动协调多技能组合使用
- 技能激活事件通过 S2（遥测）记录

**F14.5 技能与工具系统协同**

技能和工具是互补关系，共同构成 Agent 的完整能力：

| 维度 | 技能（Skill） | 工具（Tool） |
|------|------|------|
| **本质** | 程序化知识（如何做） | 执行能力（做什么） |
| **格式** | Markdown 指令 + 附属文件 + 脚本 | JSON Schema 定义 + 处理函数 |
| **控制** | Agent 自主发现和加载 | 模型通过函数调用触发 |
| **作用** | 教 Agent 工作流和最佳实践 | 提供具体操作的执行入口 |
| **示例** | "如何使用 PDF 工具处理表单" | `read_pdf`、`fill_form` |

- 技能可声明依赖的工具集：激活技能时自动确保相关工具已注册
- 技能附带的可执行脚本通过 S5 工具系统执行（确定性代码执行）
- 技能可引用 MCP Resources 作为参考材料（通过 S5 的 MCP 集成读取）
- 技能可引用 MCP Prompts 作为工作流模板

**F14.6 技能与 MCP 融合**

技能系统与 MCP 生态深度集成，互为补充：

- **Skills 提供知识，MCP 提供能力**：技能教 Agent 复杂工作流（如"如何使用 Sentry MCP 工具调试生产问题"），MCP 提供具体工具（`sentry_get_issue`、`sentry_resolve`）
- **MCP Prompts 与 Skills 分层**：MCP Prompts 是服务器级的结构化工作流模板（用户显式调用），Skills 是客户端级的程序化知识（Agent 自主发现和加载）
- **Skills Over MCP**：技能可通过 MCP Server 分发，MCP Server 暴露技能目录作为 Resources，Praxis 通过 `resources/read` 下载和安装
- **技能引导 MCP 使用**：复杂的 MCP Server 可附带技能包，教 Agent 如何有效组合使用该 Server 的 Tools、Resources 和 Prompts
- **跨平台可移植**：技能遵循 Agent Skills 开放标准，可在支持该标准的任何 Agent 平台间共享

**F14.7 技能统一管理**

- 启动时扫描所有技能路径，建立技能索引
- 运行时支持热加载新技能和卸载已有技能
- 技能版本管理：支持多版本共存、升级和回退
- 技能使用统计：通过 S2（遥测）记录每个技能的触发次数、成功率、上下文消耗
- 技能持久化：索引和缓存通过 S3（持久化）存储，避免重复扫描

**F14.8 安全考量**

- 仅从受信任来源安装技能
- 安装前审计技能内容：检查文件、代码依赖、网络连接指令
- 技能中的可执行代码通过 S5（工具系统）的沙箱环境执行
- 技能指令中的工具调用受 S8（护栏系统）权限控制
- 防止恶意技能导致数据泄露或非预期操作

---

## 6. Layer 3 · 控制层

### 6.1 S8：护栏系统（Guardrail System）

**包路径**：`praxis.guardrails`
**依赖**：S1（配置）、S2（遥测）
**被依赖**：S11（编排循环）

#### 职责边界

护栏系统是纯策略引擎——它接收操作描述，返回允许/拒绝/需确认的裁决。权限执行与模型推理在架构上分离：模型决定"要做什么"，护栏决定"是否允许做"。护栏系统不执行任何工具，也不修改任何上下文。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `check_input(user_message)` | 用户输入消息 | `GuardrailVerdict`（pass/block + 原因） | S11(编排循环) |
| `check_tool_call(tool_name, arguments, metadata)` | 工具名、参数、工具元数据 | `GuardrailVerdict`（auto_approve/confirm/deny + 原因） | S11(编排循环) |
| `check_output(assistant_response)` | 助手最终响应 | `GuardrailVerdict`（pass/block + 原因） | S11(编排循环) |
| `register_rule(rule)` | 护栏规则定义 | — | 启动时 / 运行时 |

#### 功能需求

**F8.1 三层护栏架构**

- **输入护栏**：在 Agent 处理前运行，检测提示注入、恶意指令、越界请求
  - 规则引擎：基于关键词、模式匹配的快速检测
  - 可选模型检测：调用 S4（模型网关）进行语义级注入检测（高级模式）
- **工具护栏**：在每次工具执行前运行，验证操作安全性
  - 基于工具元数据（只读/写、风险级别）和参数内容做裁决
  - 与 S5（工具系统）的元数据协同：S5 提供工具元数据，S8 基于元数据执行策略
- **输出护栏**：在最终响应返回前运行，评估内容安全性和合规性
  - 敏感信息检测（密钥、密码、个人数据泄露）
  - 内容安全性检查
- **绊线机制（Tripwire）**：任一层护栏触发绊线时，立即终止当前 Agent 循环

**F8.2 权限分层系统**

- 三阶段权限流程：
  1. **信任建立**（项目加载时）：加载项目级权限配置
  2. **实时检查**（每次工具调用前）：根据规则和上下文做裁决
  3. **人工确认**（高风险操作）：通过 S11 通知用户，等待显式确认
- 权限级别：
  - `auto_approve`：自动批准（只读操作、安全查询）
  - `confirm`：需用户确认（文件写入、命令执行、网络请求）
  - `deny`：直接拒绝（违反安全策略的操作）
- 默认策略：**限制性优先**，危险操作（文件写入、Shell 执行、文件编辑）默认 `confirm`
- 权限配置声明式管理（YAML），支持按工具名、按类别、按路径模式配置
- 支持运行时临时授权：用户可在会话中动态提升工具权限

**F8.3 护栏规则引擎**

- 规则定义格式统一（Pydantic 模型）
- 内置规则集覆盖常见风险场景
- 支持用户自定义规则扩展
- 规则评估有序执行，短路逻辑：首个 deny 即终止评估
- 每次裁决通过 S2（遥测）记录审计日志

---

### 6.2 S9：错误恢复（Error Recovery）

**包路径**：`praxis.recovery`
**依赖**：S1（配置）、S2（遥测）
**被依赖**：S11（编排循环）

#### 职责边界

错误恢复是纯策略组件——它接收错误信息，返回恢复策略（重试、降级、中止等）。它维护重试计数和熔断器状态，但不执行具体的恢复动作（由 S11 编排循环执行）。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `classify_error(error)` | 异常对象或错误信息 | `ErrorClassification`（类别 + 建议策略） | S11(编排循环) |
| `get_retry_decision(tool_name, attempt_count)` | 工具名、已重试次数 | `RetryDecision`（retry/stop + 等待时间） | S11(编排循环) |
| `check_circuit(tool_name)` | 工具名 | `CircuitState`（closed/open/half_open） | S11(编排循环) |
| `record_outcome(tool_name, success)` | 工具名、是否成功 | — | S11(编排循环) |
| `get_fallback(tool_name)` | 工具名 | 备选工具名或 None | S11(编排循环) |

#### 功能需求

**F9.1 四类错误分类**

- **瞬态错误（Transient）**：网络超时、API 限流、临时不可用
  - 策略：指数退避重试（初始 1s，最大 30s，抖动 ±25%）
- **LLM 可恢复错误（Model-Recoverable）**：参数格式错误、工具名错误、Schema 不匹配
  - 策略：将错误信息作为工具结果返回给 LLM，让模型自行修正
- **用户可修复错误（User-Fixable）**：权限不足、歧义指令、缺少必要信息
  - 策略：中断循环，通过 S11 请求用户输入
- **未预期错误（Unexpected）**：未分类的运行时异常
  - 策略：记录完整上下文到 S2（遥测），冒泡到调用方

**F9.2 重试策略**

- 每个工具的重试次数上限可配置（默认 2 次）
- 指数退避 + 随机抖动，避免重试风暴
- 重试前检查熔断器状态
- 累计重试次数跨轮次追踪

**F9.3 熔断器（Circuit Breaker）**

- 每个工具独立维护熔断器实例
- 三态模型：
  - **闭合（Closed）**：正常运行，记录失败计数
  - **断开（Open）**：连续失败超过阈值（默认 3 次），拒绝新请求，返回快速失败
  - **半开（Half-Open）**：断开持续配置时间后（默认 60s），允许一次探测请求
- 状态转换通过 S2（遥测）记录

**F9.4 优雅降级**

- 维护工具降级映射：当首选工具不可用时，自动切换到降级替代方案
- LLM Provider 降级：主 Provider 故障时通知 S4（模型网关）执行故障转移
- 降级事件可观测（日志 + 指标）

---

### 6.3 S10：验证引擎（Verification Engine）

**包路径**：`praxis.verification`
**依赖**：S1（配置）、S2（遥测）、S4（模型网关）、S5（工具系统）
**被依赖**：S11（编排循环）

#### 职责边界

验证引擎负责评估 Agent 产出的质量。它是将演示级 Agent 提升为生产级 Agent 的关键组件。验证引擎提供三种验证能力（计算型、推理型、视觉），由 S11（编排循环）在适当时机调用。

> *研究表明：给予模型验证自身工作的能力可提升质量 2~3 倍。*

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `run_computational(verifiers, target)` | 验证器列表、待验证目标 | `VerificationResult`（pass/fail + 详情列表） | S11(编排循环) |
| `run_inferential(criteria, content)` | 评估标准、待评估内容 | `VerificationResult`（评分 + 判定 + 反馈） | S11(编排循环) |
| `run_visual(url, expectations)` | 页面 URL、预期描述 | `VerificationResult`（截图 + 判定 + 差异） | S11(编排循环) |
| `register_verifier(verifier)` | 验证器定义 | — | 启动时 |

#### 功能需求

**F10.1 计算型验证（Computational Verification）**

- 确定性的、基于 CPU 的快速验证
- 支持注册验证器（Verifier Protocol）：
  - **测试套件验证器**：运行项目测试并解析结果（通过 S5 工具系统执行）
  - **类型检查验证器**：运行类型检查工具
  - **Lint 验证器**：运行代码风格检查
  - **Schema 验证器**：校验结构化输出的 Schema 合规性
- 毫秒到秒级执行，结果可靠
- 验证失败时返回结构化的失败详情（文件、行号、错误消息）

**F10.2 推理型验证（Inferential Verification / LLM-as-Judge）**

- 调用 S4（模型网关）的 `judge` 接口，使用独立 LLM 评估执行结果
- 评估代理与执行代理分离（不同上下文，避免自我偏见）
- 评估标准可自定义：维度列表（正确性、完整性、代码质量等）+ 阈值
- 评估结果包含：数值评分、通过/失败判定、文字反馈（可回馈给执行 Agent）
- 捕获计算型验证无法发现的语义问题：过度工程化、冗余实现、误诊

**F10.3 视觉验证**

- 通过浏览器自动化（Playwright）截取页面截图
- 截图提交给 S4（模型网关）的多模态能力进行视觉比对
- 验证结果包含：截图、视觉判定、差异描述
- 适用于 UI 任务的端到端验证

**F10.4 Gather → Act → Verify 循环支持**

- 验证引擎作为 GAV 循环中 Verify 阶段的执行者
- 标准流程：
  - **Gather**：Agent 通过工具收集上下文（S11 驱动）
  - **Act**：Agent 执行操作（S11 驱动）
  - **Verify**：S11 调用验证引擎检查结果
- 验证失败时，S11 将验证结果注入上下文，重新进入 Gather 阶段
- 验证引擎不控制循环流程（由 S11 控制），只提供验证能力

**F10.5 前馈与反馈控制矩阵**

验证引擎的能力覆盖 Harness 工程控制框架的四象限：

| | 计算型（CPU） | 推理型（LLM） |
|------|------|------|
| **前馈（行动前）** | Schema 预校验、类型检查 | LLM 预审计划合理性 |
| **反馈（行动后）** | 测试套件、Lint、结构分析 | LLM-as-Judge 语义评估 |

**F10.6 质量左移**

- **集成前**（每次变更伴随运行）：Lint、快速测试、基本代码审查
- **集成后**（流水线中运行）：变异测试、更广泛的代码审查、全局评估
- **持续漂移监控**（变更生命周期之外）：死代码检测、测试覆盖率分析、依赖扫描
- **运行时反馈**：SLO 退化检测、AI 评审持续采样质量、日志异常标记

---

## 7. Layer 4 · 编排层

### 7.1 S11：编排循环（Orchestration Loop）

**包路径**：`praxis.orchestrator`
**依赖**：S1（配置）、S2（遥测）、S4（模型网关）、S5（工具系统）、S6（记忆系统）、S7（上下文引擎）、S8（护栏系统）、S9（错误恢复）、S10（验证引擎）
**被依赖**：S12（会话管理）、S13（子代理协调）

#### 职责边界

编排循环是 Harness 的心跳——实现 **Thought → Action → Observation（TAO）** 循环。它是所有组件的协调中枢，负责按正确顺序调用各组件完成一个完整的 Agent 轮次。编排循环本身是"哑循环"，所有智能由模型提供，Harness 仅管理轮次流程和组件协调。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `run(user_message, config?)` | 用户消息、可选运行配置 | `AgentResponse`（最终响应 + 执行元数据） | S12(会话管理)、外部调用者 |
| `run_stream(user_message, config?)` | 同上 | `AsyncIterator[AgentEvent]`（事件流） | S12(会话管理)、外部调用者 |
| `abort()` | — | — | S12(会话管理)、用户中断 |
| `get_state()` | — | 当前循环状态（轮次数、阶段、Token 用量） | S12(会话管理) |

#### 功能需求

**F11.1 核心循环引擎**

- 实现完整的 TAO 循环：
  ```
  S6.append_message(user_message)      → 记录用户消息
  while not terminated:
      prompt      = S7.assemble_prompt(turn_context)
      response    = S4.chat(prompt, tools)
      tool_calls  = parse_tool_calls(response)
      if no tool_calls:
          S8.check_output(response)  → 输出护栏
          S6.append_message(response)  → 记录助手响应
          return response               → 自然终止
      for each tool_call:
          S8.check_tool_call(...)    → 工具护栏
          S9.check_circuit(...)      → 熔断检查
          result = S5.execute_tool(...)  → 工具执行
          S9.record_outcome(...)     → 记录结果
          if failed:
              strategy = S9.classify_error(...)
              handle_by_strategy(...)
      S6.append_message(response)    → 记录助手响应（含工具调用）
      S7.update_with_result(results) → 更新上下文
      check_termination_conditions() → 终止检查
  ```
- 支持三种运行模式：同步（阻塞等待）、异步（await）、流式（逐事件输出）
- 单次循环框架开销 < 50ms（不含 LLM 推理和工具执行时间）

**F11.2 输出解析**

- 依赖 LLM 原生 `tool_calls` 结构化输出，而非自由文本正则解析
- 判断逻辑：有 tool_calls → 进入工具执行 → 继续循环；无 tool_calls → 最终响应 → 退出循环
- 支持多个并行工具调用的批量解析
- 支持 Handoff 请求检测：模型请求转交给特定子代理时，通知 S13（子代理协调）
- 结构化输出约束：支持基于 Pydantic 模型的 Schema 约束响应

**F11.3 终止条件管理**

- 多层终止条件，按优先级评估：
  1. 护栏绊线触发（S8 返回 block）→ 立即终止
  2. 用户主动中断（`abort()` 调用）→ 立即终止
  3. 安全拒绝（模型返回 safety refusal）→ 终止并报告
  4. 模型返回最终响应（无工具调用）→ 自然终止
  5. 达到最大轮次上限 → 强制终止并返回中间状态
  6. 耗尽 Token 预算（S7 报告）→ 强制终止
- 所有终止条件的阈值可通过 S1（配置）配置

**F11.4 循环策略选择**

- **ReAct 模式**（默认）：每步交叉推理与行动，灵活适合探索性任务
- **Plan-and-Execute 模式**：先规划步骤列表，再批量执行，适合结构化任务
- 策略可在运行时根据任务特征切换

**F11.5 工具调用协调流程**

编排循环在调用工具时的完整协调流程（涉及多组件）：

```
S11.编排循环
    │
    ├─1→ S8.check_tool_call()        ─── 护栏裁决
    │    ├─ auto_approve → 继续
    │    ├─ confirm → 暂停，请求用户确认
    │    └─ deny → 跳过该工具调用，返回拒绝信息给 LLM
    │
    ├─2→ S9.check_circuit()          ─── 熔断检查
    │    ├─ closed → 继续
    │    ├─ open → 返回快速失败给 LLM
    │    └─ half_open → 允许探测
    │
    ├─3→ S5.execute_tool()           ─── 工具执行
    │    ├─ 参数验证 → 沙箱执行 → 结果捕获 → 格式化
    │    └─ 异常 → 返回错误 ToolResult
    │
    ├─4→ S9.record_outcome()         ─── 记录成功/失败
    │
    ├─5→ [如果失败] S9.classify_error() → 决定恢复策略
    │    ├─ Transient → S9.get_retry_decision() → 重试或放弃
    │    ├─ Model-Recoverable → 将错误返回 LLM
    │    ├─ User-Fixable → 暂停，请求用户输入
    │    └─ Unexpected → 记录日志，终止或继续（配置决定）
    │
    └─6→ [可选] S10.run_computational() ─── 变更后验证
```

**F11.6 事件发射**

- 编排循环在关键节点发射结构化事件，供外部消费：
  - `turn_start`：新轮次开始
  - `llm_request`：向 LLM 发送请求
  - `llm_response`：收到 LLM 响应
  - `tool_call_start`：开始工具调用
  - `tool_call_end`：工具调用完成
  - `verification_result`：验证结果
  - `turn_end`：轮次结束
  - `termination`：循环终止（含原因）
- 事件通过 S2（遥测）记录，同时通过流式接口实时推送

---

### 7.2 S12：会话管理（Session Manager）

**包路径**：`praxis.session`
**依赖**：S1（配置）、S2（遥测）、S3（持久化）、S6（记忆）、S7（上下文引擎）、S11（编排循环）
**被依赖**：S13（子代理协调）、外部调用者

#### 职责边界

会话管理负责 Agent 会话的"诞生"到"终结"全过程——初始化、运行、暂停、恢复、终止。它管理跨上下文窗口的状态续接，使 Agent 能在长期项目中存活。S12 是外部调用者与 Harness 交互的主要入口。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `create_session(project_config)` | 项目配置 | `Session` 对象 | 外部调用者 |
| `resume_session(session_id)` | 会话 ID | `Session` 对象（恢复状态） | 外部调用者 |
| `run_turn(session, user_message)` | 会话、用户消息 | `AgentResponse` | 外部调用者 |
| `run_turn_stream(session, user_message)` | 会话、用户消息 | `AsyncIterator[AgentEvent]` | 外部调用者 |
| `save_checkpoint(session)` | 会话 | 检查点 ID | 自动/手动 |
| `list_checkpoints(session_id)` | 会话 ID | 检查点列表（ID + 时间 + 摘要） | 外部调用者 |
| `rollback(session, checkpoint_id)` | 会话、检查点 ID | 恢复后的 Session | 外部调用者 |
| `terminate_session(session)` | 会话 | — | 外部调用者 |

#### 功能需求

**F12.1 会话初始化**

- 创建新会话时：
  1. 初始化所有组件实例（S4~S11）并注入各自配置
  2. 加载项目级记忆索引（S6）
  3. 注册项目工具集（S5）
  4. 加载权限配置（S8）
  5. 生成会话 ID 和初始检查点
- 恢复会话时：
  1. 从 S3（持久化）加载最新检查点
  2. 恢复 S6（记忆）状态（内部自动从游标位置继续后台处理）
  3. 恢复 S7（上下文引擎）状态
  4. 验证恢复状态的完整性

**F12.2 自动检查点**

- 在每次 S11 循环终止后（自然终止或强制终止），自动保存检查点
- 检查点内容：
  - 会话元数据（ID、创建时间、轮次计数、累计 Token）
  - S6 记忆状态快照（工作记忆消息序列、记忆索引；内含后台游标，由 S6 自行管理）
  - S7 上下文引擎状态快照（压缩历史、遮蔽状态）
  - S11 编排循环状态（轮次号、最后终止原因）
- 检查点通过 S3（持久化）写入
- 写入延迟 < 200ms

**F12.3 跨上下文窗口续接**

- 采纳 Anthropic 验证的两阶段模式：
  - **初始化阶段**：首次会话使用专用系统 Prompt，引导 Agent 建立环境（初始化脚本、进度文件、功能列表、初始 Git 提交）
  - **增量进度阶段**：后续会话从 S6（记忆）读取工作记忆，定位当前状态，继续未完成工作
- **热身序列标准化**：每次会话恢复时自动执行：
  1. 检查工作目录状态
  2. 读取进度文件和待办列表（S6 工作记忆）
  3. 验证基础功能（S10 计算型验证）
  4. 开始新工作

**F12.4 长时间运行任务支持**

- **一次一个功能**：每个 Agent 轮次仅完成一个功能，杜绝"一步到位"
- **干净状态原则**：每次会话结束时，工作产出应处于可合并状态
- **增量前进**：每步留下清晰工件（代码提交、进度文件更新），即使中断也不丢失进度
- **JSON 功能列表**：使用 JSON 格式追踪功能状态（比 Markdown 更不容易被模型意外修改）

**F12.5 时间旅行调试**

- 通过 `rollback` 接口回退到任意历史检查点
- 回退后 S6、S7、S11 状态全部恢复到该检查点时刻
- 支持从回退点重新运行，探索不同执行路径

---

### 7.3 S13：子代理协调（Subagent Coordinator）

**包路径**：`praxis.subagent`
**依赖**：S1（配置）、S2（遥测）、S11（编排循环）、S12（会话管理）
**被依赖**：S7（上下文引擎，通过子代理委托）

#### 职责边界

子代理协调管理多 Agent 场景——创建子代理实例、分配任务、隔离上下文、聚合结果。当单个 Agent 的工具集超过 ~10 个重叠工具，或存在明确分离的任务域时，启用子代理协作。

> 架构决策：**单 Agent 优先，按需分裂**。Anthropic 和 OpenAI 均建议最大化单 Agent 能力，仅在必要时才拆分为多 Agent。

#### 公开接口契约

| 接口 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `spawn_agent_as_tool(task, tools, context_summary)` | 子任务描述、工具子集、上下文摘要 | `SubagentResult`（精炼摘要） | S11(编排循环) 通过工具 |
| `handoff(target_agent_type, context_summary)` | 目标代理类型、上下文摘要 | 控制权转移到目标代理 | S11(编排循环) |
| `fork(parent_session, task)` | 父会话、子任务 | 子代理 Session（继承父上下文只读副本） | S11(编排循环) |
| `collect_results(subagent_ids)` | 子代理 ID 列表 | 结果列表 | S11(编排循环) |

#### 功能需求

**F13.1 三种执行模型**

- **Agent-as-Tool（代理即工具）**：
  - 专家子代理处理有界子任务，返回结构化结果给主代理
  - 子代理拥有独立的 S11 循环实例和独立上下文窗口
  - 子代理工具集为主代理工具集的子集（按任务需求过滤）
  - 子代理可消耗数万 Token 进行深度工作，仅返回 1,000~2,000 Token 精炼摘要
- **Handoff（移交）**：
  - 专家代理接管完全控制权，主代理暂停
  - 传递精炼的上下文摘要而非完整历史
  - 移交完成后控制权返回主代理
- **Fork（分叉）**：
  - 子代理获得父代理上下文的只读副本
  - 并行执行，独立工作
  - 适用于需要并行探索多条路径的场景

**F13.2 上下文隔离**

- 每个子代理拥有独立的组件实例集（S4~S11），通过 S12 创建
- Handoff 时传递精炼摘要，非完整历史，避免上下文污染
- 子代理间不共享可变状态
- 子代理的遥测数据（S2）关联到父 Agent 的追踪链路

**F13.3 结果聚合**

- 子代理返回结构化的 `SubagentResult`（摘要、关键发现、执行元数据）
- 主代理综合分析多个子代理输出
- 冲突结果的协调：多个子代理对同一问题给出不同答案时，通知主代理裁决

**F13.4 资源管控**

- 子代理并发数上限可配置（默认 5）
- 每个子代理独立的 Token 预算和轮次上限
- 子代理超时控制：超过配置时间未完成，强制终止并返回部分结果

---

## 8. 跨组件调用链

以下场景走查展示组件间的完整协作流程，验证依赖关系和接口契约的完整性。

### 8.1 场景一：单轮次完整执行

**触发**：用户发送一条消息，Agent 需要调用工具后返回最终响应。

```
用户消息
    │
    ▼
S12.run_turn(session, message)               ─── 会话入口
    │
    ├─→ S8.check_input(user_msg)              ─── 输入护栏
    │    └─ pass → 继续
    │
    ├─→ S11.run(message)                      ─── 启动编排循环
    │    │
    │    ├─→ S6.append_message(user_msg)      ─── 记录用户消息到工作记忆
    │    │
    │    ├─ [Turn 1] ──────────────────────────
    │    │   ├─→ S7.assemble_prompt()         ─── 组装 Prompt
    │    │   │    ├─→ S5.get_tool_schemas()   ─── 获取工具定义
    │    │   │    ├─→ S6.get_message_history() ── 获取对话历史
    │    │   │    ├─→ S6.get_memory_index()   ─── 获取记忆索引
    │    │   │    └─→ S6.search_memory(task)  ─── 语义检索相关记忆
    │    │   │
    │    │   ├─→ S4.chat(prompt, tools)       ─── LLM 推理
    │    │   │    └─→ S2.emit_metric(...)     ─── Token 计量
    │    │   │
    │    │   ├─→ parse_tool_calls(response)   ─── 解析输出
    │    │   │    └─ 检测到 tool_call: edit_file
    │    │   │
    │    │   ├─→ S8.check_tool_call(edit_file, args, meta) ── 工具护栏
    │    │   │    └─ confirm → 请求用户确认 → 用户批准
    │    │   │
    │    │   ├─→ S9.check_circuit(edit_file)  ─── 熔断检查
    │    │   │    └─ closed → 继续
    │    │   │
    │    │   ├─→ S5.execute_tool(edit_file, args) ── 工具执行
    │    │   │    └─ 返回 ToolResult(success)
    │    │   │
    │    │   ├─→ S9.record_outcome(edit_file, true) ── 记录成功
    │    │   │
    │    │   ├─→ S10.run_computational([lint], target) ── 变更后验证
    │    │   │    └─ 返回 pass
    │    │   │
    │    │   ├─→ S6.append_message(response)  ─── 记录助手响应（S6 内部自动触发后台提取/整合）
    │    │   └─→ S7.update_with_result(results) ── 更新上下文
    │    │
    │    ├─ [Turn 2] ──────────────────────────
    │    │   ├─→ S7.assemble_prompt()         ─── 重新组装
    │    │   ├─→ S4.chat(prompt, tools)       ─── LLM 推理
    │    │   ├─→ parse_tool_calls(response)
    │    │   │    └─ 无工具调用 → 最终响应
    │    │   ├─→ S8.check_output(response)    ─── 输出护栏
    │    │   │    └─ pass → 返回
    │    │   └─→ S6.append_message(response)  ─── 记录最终响应
    │    │
    │    └─→ return AgentResponse
    │
    ├─→ S12.save_checkpoint(session)          ─── 自动检查点
    │    └─→ S3.save_checkpoint(...)          ─── 持久化
    │
    └─→ return AgentResponse 给用户
    （S6 后台异步任务默默完成记忆提取与整合，对此流程完全透明）
```

### 8.2 场景二：工具执行失败与错误恢复

**触发**：工具执行失败，触发错误恢复流程。

```
S11 编排循环中
    │
    ├─→ S5.execute_tool(web_fetch, args)      ─── 工具执行
    │    └─ 抛出 ConnectionTimeout 异常
    │
    ├─→ S9.classify_error(ConnectionTimeout)  ─── 错误分类
    │    └─ 返回 Transient（瞬态错误）
    │
    ├─→ S9.get_retry_decision(web_fetch, attempt=1) ── 重试决策
    │    └─ retry, wait=2s
    │
    ├─→ [等待 2s]
    │
    ├─→ S5.execute_tool(web_fetch, args)      ─── 重试执行
    │    └─ 再次失败
    │
    ├─→ S9.get_retry_decision(web_fetch, attempt=2) ── 第二次重试决策
    │    └─ stop（达到最大重试次数）
    │
    ├─→ S9.record_outcome(web_fetch, false)   ─── 记录失败
    │    └─ 熔断器失败计数 +1
    │
    ├─→ S9.get_fallback(web_fetch)            ─── 查找降级方案
    │    └─ 返回 None（无降级方案）
    │
    └─→ 将错误信息作为 ToolResult(error) 返回 LLM
         └─ LLM 根据错误信息调整策略（Model-Recoverable 路径）
```

### 8.3 场景三：上下文窗口溢出处理

**触发**：Token 用量接近窗口限制。

```
S11 编排循环中
    │
    ├─→ S7.assemble_prompt()
    │    │
    │    ├─→ S4.get_token_count(messages)     ─── Token 计数
    │    │    └─ 返回 180,000 tokens（窗口 200K 的 90%）
    │    │
    │    ├─→ [超过 80% 阈值] 自动触发压缩
    │    │
    │    ├─→ S7.trigger_compaction()          ─── 上下文压缩
    │    │    ├─→ S4.summarize(history, instruction) ── LLM 摘要
    │    │    ├─→ S6.get_message_history()    ─── 获取完整历史
    │    │    ├─→ 替换历史消息为摘要
    │    │    ├─→ 保留最近 5 个关键文件引用
    │    │    └─→ S2.emit_metric(compaction_ratio, 0.4) ── 记录压缩率
    │    │
    │    └─→ 返回压缩后的 Prompt（~72,000 tokens）
    │
    └─→ S4.chat(compressed_prompt, tools)     ─── 正常继续
```

### 8.4 场景四：会话恢复（跨窗口续接）

**触发**：用户恢复一个之前中断的长时间运行会话。

```
用户请求恢复会话 session_abc
    │
    ▼
S12.resume_session("session_abc")              ─── 恢复入口
    │
    ├─→ S3.load_checkpoint(latest_checkpoint)  ─── 加载检查点
    │    └─ 返回完整状态快照
    │
    ├─→ S6.import_state(snapshot.memory)       ─── 恢复记忆
    │    ├─ 恢复工作记忆（消息历史、进度文件、待办列表）
    │    └─ 内部自动从游标位置继续后台提取/整合
    │
    ├─→ S7.import_state(snapshot.context)      ─── 恢复上下文引擎
    │
    ├─→ 重新初始化 S4, S5, S8, S9, S10, S11, S14 ── 无状态组件重建
    │
    └─→ 返回恢复后的 Session
         │
         ▼
S12.run_turn(session, user_message)
    │
    └─→ S11.run(message)
         │
         ├─→ S7.assemble_prompt()
         │    ├─ 包含恢复的压缩历史
         │    ├─ 包含热身序列指令
         │    └─ 包含工作记忆内容（进度、待办）
         │
         └─→ Agent 自动执行热身：
              ├─ 读取 progress.json（通过 S5 工具）
              ├─ 读取 todos.json（通过 S5 工具）
              ├─ S10.run_computational() 验证基础功能
              └─ 选择最高优先级未完成功能继续工作
```

### 8.5 场景五：子代理任务委托

**触发**：主 Agent 需要深度研究一个子问题。

```
S11 主编排循环
    │
    ├─→ LLM 返回 tool_call: spawn_research_agent
    │
    ├─→ S8.check_tool_call(spawn_research_agent, ...) ── 护栏
    │    └─ auto_approve
    │
    ├─→ S13.spawn_agent_as_tool(                      ── 创建子代理
    │        task="分析 X 模块的性能瓶颈",
    │        tools=[read_file, grep_search, run_command],
    │        context_summary="项目使用 Python + asyncio..."
    │    )
    │    │
    │    ├─→ S12.create_session(sub_config)           ─── 创建子会话
    │    │    ├─→ 初始化独立的 S4~S11 实例集
    │    │    └─→ 注入工具子集到子 S5
    │    │
    │    ├─→ 子 S11.run(task)                         ─── 子代理执行
    │    │    ├─ [Turn 1~N] 子代理自主执行多轮次
    │    │    ├─ 消耗 ~30,000 tokens 深度分析
    │    │    └─ 返回详细结果
    │    │
    │    ├─→ 压缩结果为 ~1,500 tokens 的精炼摘要
    │    │
    │    └─→ 返回 SubagentResult 给主 S11
    │
    ├─→ S7.update_with_result([subagent_result])      ─── 更新主上下文
    │
    └─→ 主 LLM 基于子代理摘要继续工作
```

### 8.6 场景六：护栏绊线触发

**触发**：模型尝试执行危险操作，护栏绊线触发。

```
S11 编排循环中
    │
    ├─→ LLM 返回 tool_call: run_command(args={cmd: "rm -rf /"})
    │
    ├─→ S8.check_tool_call(run_command, {cmd: "rm -rf /"}, meta)
    │    │
    │    ├─ 规则评估：匹配"破坏性命令模式"规则
    │    ├─ 裁决：deny + tripwire
    │    └─→ S2.record_audit(tripwire_triggered, ...)  ─── 审计记录
    │
    ├─→ S11 检测到 tripwire → 立即终止循环
    │
    ├─→ S12.save_checkpoint(session)                    ─── 保存状态
    │
    └─→ 返回 AgentResponse(
              terminated=true,
              reason="tripwire: 危险命令被护栏拦截",
              last_state=...
         )
```

### 8.7 场景七：技能触发的工具工作流

**触发**：用户请求处理 PDF 表单，Agent 自动发现并激活 PDF 技能，在技能指导下使用工具完成任务。

```
用户消息："请帮我填写这份 PDF 表单"
    │
    ▼
S12.run_turn(session, message)
    │
    ├─→ S11.run(message)                              ─── 启动编排循环
    │    │
    │    ├─ [Turn 1] ────────────────────────────
    │    │   ├─→ S7.assemble_prompt()                 ─── 组装 Prompt
    │    │   │    ├─→ S14.get_skill_index()            ─── 获取技能索引
    │    │   │    │    └─ 返回: [{name: "pdf-processor",
    │    │   │    │         description: "处理 PDF..."},
    │    │   │    │         {name: "excel-handler", ...}, ...]
    │    │   │    ├─→ 技能索引注入系统提示（第一层披露）
    │    │   │    ├─→ S5.get_tool_schemas()             ─── 获取工具定义
    │    │   │    └─→ S6.get_message_history()
    │    │   │
    │    │   ├─→ S4.chat(prompt, tools)                ─── LLM 推理
    │    │   │    └─ LLM 判断需要 pdf-processor 技能
    │    │   │
    │    │   ├─→ 检测到 tool_call: read_file("pdf/SKILL.md")
    │    │   │    └─→ S14.load_skill("pdf-processor")  ─── 第二层披露
    │    │   │         └─ 返回完整 SKILL.md 内容（指令+工作流）
    │    │   │
    │    │   └─→ S7.update_with_result([skill_content])
    │    │
    │    ├─ [Turn 2] ────────────────────────────
    │    │   ├─→ S7.assemble_prompt()                  ─── 含技能指令
    │    │   ├─→ S4.chat(prompt, tools)                ─── LLM 按技能指导行动
    │    │   │    └─ LLM 依据技能中的步骤指引，调用 read_pdf
    │    │   │
    │    │   ├─→ S14.load_skill_file("pdf-processor", "forms.md")
    │    │   │    └─ 第三层披露：加载表单填写专项指引
    │    │   │
    │    │   ├─→ S14.list_skill_tools("pdf-processor")
    │    │   │    └─ 返回技能附带的 Python 脚本列表
    │    │   │
    │    │   ├─→ S5.execute_tool(extract_form_fields, {pdf: "..."})
    │    │   │    └─ 执行技能附带的确定性 Python 脚本
    │    │   │         └─ 返回 ToolResult(form_fields=[...])
    │    │   │
    │    │   └─→ S7.update_with_result([form_fields])
    │    │
    │    ├─ [Turn 3~N] ─────────────────────────
    │    │   └─ LLM 在技能指导下依次填写表单字段、生成输出 PDF
    │    │
    │    └─→ return AgentResponse
    │
    └─→ S12.save_checkpoint(session)
```

### 8.8 场景八：MCP 服务器交互（Elicitation + Sampling）

**触发**：Agent 通过 MCP Server 执行外部任务，Server 需要用户确认和 LLM 分析。

```
S11 编排循环中
    │
    ├─→ LLM 返回 tool_call: mcp_book_flight(origin="NYC", dest="BCN")
    │    └─ 该工具来自 MCP Server（travel-booking-server）
    │
    ├─→ S8.check_tool_call(mcp_book_flight, args, meta) ─── 护栏
    │    └─ confirm → 用户确认
    │
    ├─→ S5.execute_tool(mcp_book_flight, args)      ─── 通过 MCP 执行
    │    │
    │    ├─→ MCP Client → tools/call → Travel Server
    │    │
    │    │  [Server 需要 LLM 分析航班]
    │    ├─← Travel Server → sampling/createMessage   ─── Sampling 请求
    │    │    │  "分析 47 个航班选项，推荐最优选择"
    │    │    │  modelPreferences: {intelligencePriority: 0.9}
    │    │    │
    │    │    ├─→ S11 → 用户审核采样请求 → 批准
    │    │    ├─→ S4.chat(sampling_messages, config)  ─── 独立 LLM 调用
    │    │    │    └─ 返回航班分析和推荐
    │    │    ├─→ S11 → 用户审核响应 → 批准
    │    │    └─→ 返回采样结果给 Travel Server
    │    │
    │    │  [Server 需要用户确认预订]
    │    ├─← Travel Server → elicitation/create       ─── Elicitation 请求
    │    │    │  message: "确认预订 BCN 航班"
    │    │    │  schema: {confirmBooking: boolean, seatPref: enum}
    │    │    │
    │    │    ├─→ S11 → 转发给用户界面（CLI/API）
    │    │    ├─→ 用户填写表单：确认预订 + 选择靠窗座位
    │    │    └─→ 返回用户响应给 Travel Server
    │    │
    │    └─→ Travel Server 完成预订 → 返回 ToolResult
    │
    ├─→ S9.record_outcome(mcp_book_flight, true)    ─── 记录成功
    │
    └─→ S7.update_with_result([booking_result])     ─── 更新上下文
```

---

## 9. 非功能需求

### 9.1 性能

| 指标 | 目标 | 测量方式 |
|------|------|---------|
| Harness 循环开销 | < 50ms/轮次（不含 LLM 推理） | S2 指标：`turn_overhead_ms` |
| 工具调用延迟 | < 100ms（不含外部 API） | S2 指标：`tool_execution_ms` |
| 状态检查点写入 | < 200ms | S2 指标：`checkpoint_write_ms` |
| 上下文压缩 | < 5s | S2 指标：`compaction_duration_ms` |
| Prompt 组装 | < 20ms | S2 指标：`prompt_assembly_ms` |
| 护栏裁决 | < 10ms（规则引擎）/ < 2s（模型检测） | S2 指标：`guardrail_latency_ms` |

### 9.2 可靠性

| 指标 | 目标 |
|------|------|
| 单步成功率 | ≥ 99.5% |
| 10 步端到端成功率 | ≥ 95%（通过 S9 错误恢复提升） |
| 状态恢复成功率 | 100%（从 S3 检查点恢复） |
| 宕机后续接准确率 | ≥ 95% |
| 审计日志完整率 | 100%（S2 零丢失） |

### 9.3 可扩展性

| 维度 | 目标 |
|------|------|
| 工具注册表容量 | 100+ 工具（通过 S5 分组和懒加载管理） |
| 单 Agent 最大轮次 | 1000+ 轮次（通过 S7 压缩和 S3 检查点管理） |
| 子代理并发数 | 可配置，默认上限 5（S13 管控） |
| 长期记忆条目 | 10,000+ 条（通过 S6 三层层级和 S3 索引管理） |
| MCP 服务器数 | 10+ 并发连接（S5 管理） |
| 技能库容量 | 200+ 技能（通过 S14 渐进式披露管理上下文开销） |

### 9.4 安全性

| 要求 | 负责组件 |
|------|-----------|
| 所有工具在沙箱中执行 | S5（沙箱执行环境） |
| 敏感数据不离开配置的安全边界 | S8（输出护栏检测） |
| 完整审计日志覆盖每次 Agent 操作 | S2（审计日志） |
| 速率限制和成本上限 | S4（Token 预算）+ S11（轮次上限） |
| 权限最小化原则 | S8（默认 confirm） |
| 提示注入防护 | S8（输入护栏） |

### 9.5 可观测性

| 维度 | 实现 | 负责组件 |
|------|------|-----------|
| 结构化日志 | 每轮次完整日志（组件名、操作、耗时） | S2 |
| 分布式追踪 | Agent → 子代理 → 工具的完整链路 | S2 |
| 指标仪表板 | 成功率、平均轮次数、Token 效率、成本 | S2 |
| 审计追踪 | 工具调用 + 权限决策 + LLM 调用全记录 | S2 |
| 健康检查 | 各组件运行状态、熔断器状态 | S9 + S2 |

---

## 10. 架构决策记录

| # | 决策维度 | 选择 | 理由 | 涉及组件 |
|---|---------|------|------|-----------|
| 1 | 单 Agent vs 多 Agent | 单 Agent 优先，按需分裂 | Anthropic/OpenAI 建议最大化单 Agent；多 Agent 仅在工具集 >10 重叠或任务域分离时启用 | S11, S13 |
| 2 | ReAct vs Plan-and-Execute | 默认 ReAct，可切换 | ReAct 灵活适合探索；Plan-and-Execute 3.6x 加速但仅适合结构化任务 | S11 |
| 3 | 上下文窗口管理 | 压缩 + 笔记 + 子代理三管齐下 | 压缩保持对话流，笔记适合里程碑，子代理处理深度探索 | S7, S6, S13 |
| 4 | 验证策略 | 计算型优先，推理型补充 | 计算型确定性成本低；推理型捕获语义问题但增加延迟 | S10 |
| 5 | 权限安全 | 默认限制性，可配置放宽 | 生产安全 > 开发速度；允许用户配置放宽 | S8 |
| 6 | 工具集范围 | 最小暴露 + 懒加载 | 工具越多性能越差；仅暴露当前步骤所需最小集合 | S5, S7 |
| 7 | Harness 厚度 | 薄 Harness，信任模型 | 随模型进步持续削减 Harness 复杂度 | 全局 |
| 8 | 组件通信 | 直接函数调用，非消息队列 | 单进程部署，避免分布式复杂度；未来可替换为消息总线 | 全局 |
| 9 | 状态持久化 | 检查点式，非事件溯源 | 检查点简单可靠；事件溯源在当前规模下过度工程化 | S3, S12 |
| 10 | 技能知识 vs 工具能力 | Skills 与 Tools 分离，独立组件 | Skills 提供程序化知识，Tools 提供执行能力；分离后可独立演进、组合复用 | S5, S14 |
| 11 | MCP 集成深度 | 完整协议实现，非仅 Tools | 三大原语 + 三大客户端特性确保与 MCP 生态全面兼容 | S5 |
| 12 | 技能格式 | Agent Skills 开放标准 | 跨平台可移植、社区生态兼容、渐进式披露节约上下文 | S14 |
| 13 | LLM 接入方式 | LiteLLM SDK 薄封装，不二次抽象 | 100+ Provider 开箱即用、Router 内置负载均衡/重试/fallback/cooldown；避免重复造轮子；社区维护的模型价格表自动成本追踪 | S4 |

---

## 11. 技术栈约束

| 维度 | 选择 | 约束来源 |
|------|------|---------|
| 编程语言 | Python 3.12+ | 项目规范 |
| 包管理 | uv | 项目规范，禁止 pip/poetry/pipenv |
| 数据模型 | Pydantic BaseModel | 项目规范，禁止 dataclass |
| 配置管理 | Pydantic Settings（pydantic-settings） | S1 配置系统 |
| 异步框架 | asyncio（原生） | S11 编排循环 |
| LLM 交互 | LiteLLM（Python SDK + Router） | S4 模型网关；100+ Provider 统一接入、负载均衡、故障转移、Token/成本追踪 |
| 工具协议 | MCP（Model Context Protocol） | S5 工具系统 |
| 技能标准 | Agent Skills 开放标准（agentskills.io） | S14 技能系统 |
| MCP SDK | mcp (Python SDK) | S5 MCP 集成 |
| 进程执行 | asyncio.create_subprocess_exec | S5 沙箱执行 |
| HTTP 客户端 | httpx | S5 网络交互 |
| 持久化 | SQLite（默认）/ Redis / 文件系统 | S3 持久化引擎 |
| 代码组织 | 每组件独立子包，单文件单职责 | 项目规范 |
| 模型定义 | 集中于 `praxis.models` 包 | 项目规范 |

---

## 12. 术语表

| 术语 | 定义 | 关联组件 |
|------|------|-----------|
| **Agent Harness** | 包裹 LLM 的完整软件基础设施，由 14 个组件组合而成 | 全部 |
| **TAO / ReAct Loop** | Thought → Action → Observation 循环，Agent 的基本执行模式 | S11 |
| **Context Rot** | 随上下文 Token 增加，模型准确召回信息的能力下降 | S7 |
| **Compaction** | 将接近窗口限制的对话历史压缩为高保真摘要 | S7, S4 |
| **Tripwire** | 护栏绊线，触发后立即停止 Agent | S8, S11 |
| **Circuit Breaker** | 熔断器，防止对持续失败的工具反复重试 | S9 |
| **HITL** | Human-in-the-Loop，人在回路中的交互模式 | S8, S11 |
| **Feedforward / Guide** | 前馈控制，在 Agent 行动前预防性引导 | S10 |
| **Feedback / Sensor** | 反馈控制，在 Agent 行动后观察并纠正 | S10 |
| **Computational Control** | 确定性控制（测试、Lint），CPU 执行，快速可靠 | S10 |
| **Inferential Control** | 推理型控制（LLM-as-Judge），GPU 执行，语义分析 | S10 |
| **MCP** | Model Context Protocol，LLM 与外部工具/数据源的标准化协议，基于 JSON-RPC 2.0 | S5 |
| **MCP Host** | MCP 架构中的主机应用，管理多个 MCP Client，Praxis 扮演此角色 | S5 |
| **MCP Client** | 与单个 MCP Server 维持有状态连接的协议客户端 | S5 |
| **MCP Server** | 提供 Tools、Resources、Prompts 三大原语的外部服务 | S5 |
| **MCP Tools** | 模型控制的函数，LLM 通过函数调用触发，具有类型化输入输出 | S5 |
| **MCP Resources** | 应用控制的只读数据源，通过 URI 标识 | S5, S7 |
| **MCP Prompts** | 用户控制的预定义提示模板，支持参数化 | S5, S14 |
| **Elicitation** | MCP Server 向用户请求额外信息的结构化机制 | S5 |
| **Sampling** | MCP Server 通过 Host 请求 LLM 推理的能力 | S5, S4 |
| **Agent Skill** | 封装程序化知识的可发现、可组合、可复用的技能单元 | S14 |
| **Progressive Disclosure** | 渐进式披露，技能信息按三层按需加载以节约上下文 | S14, S7 |
| **SKILL.md** | 技能的核心文件，YAML frontmatter + Markdown 指令 | S14 |
| **Skills Over MCP** | 通过 MCP 协议分发和发现技能的机制 | S5, S14 |
| **Sub-agent** | 主代理派生的专门子代理，处理深度子任务后返回精炼摘要 | S13 |
| **GAV** | Gather → Act → Verify 循环，标准化的三阶段工作模式 | S10, S11 |
| **Checkpoint** | Agent 状态的完整快照，支持恢复和时间旅行 | S3, S12 |
| **Observation Masking** | 隐藏旧工具输出同时保留调用记录的上下文优化技术 | S7 |
| **Scratchpad** | Agent 主动维护的结构化笔记，持久化但不占上下文预算 | S6 |
| **Semantic Memory** | 语义记忆，存储事实、知识和偏好等持久化信息 | S6 |
| **Episodic Memory** | 情景记忆，捕获具体交互经历和决策过程 | S6 |
| **Procedural Memory** | 程序记忆，编码工作流、工具使用模式和行为准则 | S6 |
| **CoALA** | Cognitive Architectures for Language Agents，认知架构框架，定义 Agent 记忆分类模型 | S6 |
| **Memory Extraction** | 模型辅助记忆提取，LLM 从对话中识别并提取值得长期存储的信息 | S6, S4 |
| **Memory Consolidation** | 记忆整合，LLM 驱动的 ADD/UPDATE/NOOP 决策，智能去重和冲突解决 | S6, S4 |
| **Dream Consolidation** | 梦境整理，后台定期清理过时/矛盾/模糊记忆，借鉴 REM 睡眠概念 | S6 |
| **Memory-as-Hint** | 记忆即提示原则，记忆作为参考而非事实，使用前需实际验证 | S6 |
| **Hot Path / Background Path** | 记忆双路径处理：实时主动存取 vs 异步自动提取整合 | S6 |
| **LiteLLM** | 开源 LLM 统一接入库，提供 100+ Provider 的 OpenAI 兼容调用格式、Token 计量、成本追踪 | S4 |
| **LiteLLM Router** | LiteLLM 的路由引擎，内置负载均衡、重试、故障转移、cooldown 机制 | S4 |

---

## 13. 参考来源

| 来源 | 标题 | 关键贡献 |
|------|------|---------|
| Anthropic | Effective harnesses for long-running agents (2025) | 两阶段模式、进度文件、增量前进 |
| Anthropic | Effective context engineering for AI agents (2025) | 上下文即稀缺资源、压缩策略、子代理架构 |
| OpenAI | Harness engineering (2026) | 环境设计 > 代码实现、反馈循环、结构化测试 |
| Martin Fowler / Thoughtworks | Harness engineering for coding agent users (2026) | 前馈/反馈控制框架、计算/推理二维、质量左移 |
| Salesforce | Agent Harness: The Infrastructure for Reliable AI (2026) | 核心组件定义、架构模式、企业级部署 |
| Daily Dose of DS (Avi Chawla) | The Anatomy of an Agent Harness (2026) | 11 组件综合、七大决策、步进式工作流 |
| Parallel.ai | What is an agent harness (2026) | Harness vs Framework vs Orchestrator 辨析 |
| LangChain | Agent frameworks, runtimes, and harnesses | DeepAgents 实践、状态图建模 |
| Anthropic | Equipping agents for the real world with agent skills (2025) | Agent Skills 架构、渐进式披露、代码执行、可组合性 |
| Anthropic | Claude Skills (2025) | Skills 产品化实现、API 支持、跨平台兼容 |
| agentskills.io | Agent Skills Open Standard | 技能格式规范、SKILL.md 定义、well-known URI 发现 |
| modelcontextprotocol.io | MCP Specification 2025-11-25 | MCP 完整协议规范：传输、原语、客户端特性 |
| modelcontextprotocol.io | MCP Architecture & Concepts | Host-Client-Server 架构、Sampling、Elicitation |
| modelcontextprotocol.io | MCP 2026 Roadmap | 传输演进、Agent 通信、Skills Over MCP |
| Mem0 / ECAI 2025 | Building Production-Ready AI Agents with Scalable Long-Term Memory | 选择性记忆提取、向量+图混合存储、多作用域模型、LOCOMO 基准 |
| LangChain | LangMem：Memory for LLM Applications | 语义/情景/程序记忆、热路径+后台处理、ReflectionExecutor |
| Supermemory Research | State-of-the-Art Agent Memory (LongMemEval) | 关系版本链、时间锤定、混合检索、原子记忆+原始块双层架构 |
| AWS | AgentCore Long-Term Memory Deep Dive | 提取/整合管线、LLM 驱动 ADD/UPDATE/NOOP、自定义策略 |
| Milvus / Anthropic | Claude Code Memory System (Leaked Source Analysis) | 四层记忆架构、Auto Memory、Auto Dream、KAIROS、Memory-as-Hint |
| LiteLLM | LiteLLM Documentation (docs.litellm.ai) | 100+ Provider 统一接入、Router 负载均衡/故障转移、Token 计量与成本追踪、异常标准化、可观测性回调 |

---

*本文档基于截至 2026 年 4 月的 AI Agent Harness 领域最新研究和工程实践综合编写。*
*Praxis Harness 由 14 个组件分 5 层组合构建，每个组件可独立开发、测试和替换。*
