# Praxis 使用指南

## 目录

- [快速开始](#快速开始)
- [架构概览](#架构概览)
- [会话管理](#会话管理)
- [工具注册](#工具注册)
- [护栏配置](#护栏配置)
- [MCP 集成](#mcp-集成)
- [技能编写](#技能编写)
- [子代理委托](#子代理委托)
- [记忆系统](#记忆系统)
- [检查点与恢复](#检查点与恢复)
- [遥测与监控](#遥测与监控)

---

## 快速开始

### 1. 安装

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目并安装依赖
git clone <repo_url> praxis
cd praxis
uv sync
```

### 2. 配置

创建 `config.yaml`（可选，所有字段均有默认值）：

```yaml
gateway:
  model_list:
    - model_name: "default"
      litellm_params:
        model: "openai/gpt-4o"
        api_key: "sk-..."
  default_model: "default"

persistence:
  backend: "sqlite"
  sqlite_path: "data/praxis.db"
```

### 3. 基本使用

```python
import asyncio
from praxis.config.schemas import (
    ContextConfig, OrchestratorConfig, PersistenceConfig, SessionConfig,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.persistence.store import create_store
from praxis.session.core import SessionFactory

async def main():
    # 初始化持久化
    store = await create_store(PersistenceConfig())

    # 创建护栏引擎
    rule_engine = RuleEngine()
    rule_engine.register_builtin_rules()
    guardrails = GuardrailEngine(rule_engine, PermissionManager())

    # 创建会话工厂
    factory = SessionFactory(
        store=store,
        session_config=SessionConfig(),
        orchestrator_config=OrchestratorConfig(),
        context_config=ContextConfig(),
    )

    # 创建会话并运行
    session = factory.create_session(guardrails=guardrails)
    response = await session.run_turn("你好，帮我分析一下这段代码")
    print(response.content)

    await store.close()

asyncio.run(main())
```

### 4. 运行测试

```bash
uv run pytest                           # 全部测试
uv run pytest tests/e2e/                # 端到端场景测试
uv run pytest tests/integration/        # 跨组件集成测试
uv run pytest tests/benchmarks/         # 性能基准测试
uv run pytest tests/nfr/                # 非功能需求验证
```

---

## 架构概览

Praxis 由 **14 个独立组件**组合而成，分布在五层架构中：

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 4 · 编排层    S11 编排循环 │ S12 会话管理 │ S13 子代理协调 │
├──────────────────────────────────────────────────────────────┤
│ Layer 3 · 控制层    S8 护栏系统  │ S9 错误恢复  │ S10 验证引擎  │
├──────────────────────────────────────────────────────────────┤
│ Layer 2 · 核心能力  S5 工具 │ S6 记忆 │ S7 上下文 │ S14 技能       │
├──────────────────────────────────────────────────────────────┤
│ Layer 1 · 模型接入              S4 模型网关                   │
├──────────────────────────────────────────────────────────────┤
│ Layer 0 · 基础设施  S1 配置系统  │ S2 遥测系统  │ S3 持久化引擎 │
└──────────────────────────────────────────────────────────────┘
```

每层仅允许依赖同层或更低层的组件，禁止向上依赖。

**单轮数据流**: 用户消息 → S8 输入护栏 → S7 Prompt 组装 → S4 LLM 调用 → S11 输出解析 → S5 工具执行 → S8 输出护栏 → 返回响应

---

## 会话管理

### 创建会话

```python
session = factory.create_session(
    guardrails=guardrails,
    registry=tool_registry,
    model="openai/gpt-4o",
)
```

### 多轮对话

```python
r1 = await session.run_turn("列出当前目录文件")
r2 = await session.run_turn("打开其中的 main.py")
r3 = await session.run_turn("修复第 42 行的 bug")
```

### 会话生命周期

`INITIALIZING` → `ACTIVE` → `TERMINATED`

---

## 工具注册

### 注册自定义工具

```python
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.registry import ToolRegistry

registry = ToolRegistry()

async def read_file_handler(arguments: dict) -> str:
    path = arguments["path"]
    return open(path).read()

registry.register(
    ToolDefinition(
        name="read_file",
        description="读取文件内容",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
            },
            "required": ["path"],
        },
        metadata=ToolMetadata(
            category="file_ops",
            readonly=True,
            permission_level="auto_approve",
        ),
    ),
    read_file_handler,
)
```

### 工具元数据

| 字段 | 说明 | 默认值 |
|---|---|---|
| `category` | 工具类别（file_ops/search/shell/system/mcp） | `"general"` |
| `permission_level` | 权限级别（auto_approve/confirm/deny） | `"confirm"` |
| `readonly` | 是否只读 | `False` |
| `timeout_seconds` | 执行超时 | `30.0` |
| `tags` | 标签列表 | `[]` |

---

## 护栏配置

### 内置规则

调用 `rule_engine.register_builtin_rules()` 自动注册：
- **输入护栏**: 提示注入检测（ignore instructions, jailbreak 等）
- **输出护栏**: 敏感信息泄露检测（API Key, 密码等）

### 自定义规则

```python
from praxis.guardrails.rules import GuardrailRule, RuleTarget
from praxis.models.guardrails import VerdictType

rule_engine.register_rule(GuardrailRule(
    name="block_sql_injection",
    description="阻止 SQL 注入",
    target=RuleTarget.INPUT,
    patterns=[r"(?i)(drop\s+table|delete\s+from|union\s+select)"],
    verdict=VerdictType.BLOCK,
    tripwire=True,  # 触发后立即终止会话
))
```

### 权限管理

```python
from praxis.guardrails.permissions import PermissionManager

manager = PermissionManager.from_config_dict({
    "default_permission": "confirm",
    "rules": [
        {"tool_name": "read_file", "permission": "auto_approve"},
        {"category": "shell", "permission": "confirm"},
    ],
})
```

---

## MCP 集成

### 连接 MCP Server

```python
from praxis.tools.mcp.tools import MCPToolsBridge
from praxis.tools.mcp.connection import MCPConnectionManager
from mcp import ClientSession

bridge = MCPToolsBridge(registry)

# 发现并注册 MCP Server 工具
tools = await bridge.discover_tools("server_name", client_session)
# 工具自动注册为 mcp_{server_name}_{tool_name}
```

### Elicitation（用户确认）

```python
from praxis.tools.mcp.elicitation import ElicitationManager

elicitation = ElicitationManager()

async def user_handler(request):
    # 展示给用户并获取确认
    return MCPElicitationResponse(accepted=True)

elicitation.set_handler(user_handler)
```

### Sampling（LLM 采样）

```python
from praxis.tools.mcp.sampling import SamplingManager

sampling = SamplingManager(gateway_router)
sampling.set_review_handler(human_review_fn)  # 可选
```

---

## 技能编写

### 技能目录结构

```
.praxis/skills/
└── my-skill/
    ├── SKILL.md           # 技能定义文件（必需）
    ├── template.py        # 附属文件
    └── config.yaml        # 附属配置
```

### SKILL.md 格式

```markdown
---
name: my-skill
version: "1.0.0"
description: 简短描述
tags: [python, testing]
tools:
  - name: run_tests
    command: "pytest {path}"
---

# 技能完整内容

详细的技能指南和使用说明...
```

### 三层渐进式披露

1. **第一层（索引）**: LLM 看到技能名称和描述列表
2. **第二层（内容）**: LLM 调用 `load_skill` 获取完整 SKILL.md
3. **第三层（文件）**: LLM 调用 `load_skill_file` 获取附属文件

---

## 子代理委托

### 创建子代理

```python
from praxis.subagent.isolation import IsolatedContext
from praxis.models.subagent import SubagentSpec

isolated = IsolatedContext(store, orchestrator_config, context_config)

spec = SubagentSpec(
    task="分析 utils/ 目录的代码质量",
    tool_names=["read_file", "grep_search"],
    max_turns=10,
)

sub_session = isolated.create_isolated_session(
    spec=spec,
    guardrails=guardrails,
    parent_registry=main_registry,
    model="openai/gpt-4o-mini",
)

result = await sub_session.run_turn(spec.task)
```

### 特点

- 独立工具注册表（仅包含指定工具子集）
- 独立对话历史
- 独立轮次限制
- 共享护栏引擎（只读）

---

## 记忆系统

### 工作记忆

每个会话维护独立的工作记忆，自动追踪对话消息。

### 长期记忆

四种认知类型：
- **语义记忆** (SEMANTIC) — 事实与知识
- **情景记忆** (EPISODIC) — 交互片段
- **程序记忆** (PROCEDURAL) — 操作序列
- **工作记忆** (WORKING) — 当前会话上下文

### 记忆作用域

格式 `<type>/<id>`：
- `session/abc123` — 会话级
- `project/praxis` — 项目级
- `user/u1` — 用户级
- `global` — 全局

---

## 检查点与恢复

### 自动检查点

默认开启，每轮对话后自动保存。通过配置控制：

```python
SessionConfig(
    auto_checkpoint=True,
    max_checkpoints_per_session=50,
)
```

### 手动恢复

```python
from praxis.session.checkpoint import CheckpointManager
from praxis.session.resume import SessionResumer

resumer = SessionResumer(factory, CheckpointManager(store))
restored_session = await resumer.resume_session(session_id, guardrails)
```

---

## 遥测与监控

### 结构化日志

```python
from praxis.telemetry.logger import get_logger
log = get_logger("my_module")
log.info("操作完成", duration_ms=42, items=10)
```

按组件配置日志级别：

```yaml
telemetry:
  log_level: "INFO"
  log_levels:
    gateway: "DEBUG"
    guardrails: "WARNING"
```

### 指标

```python
from praxis.telemetry.metrics import emit_metric
emit_metric("request_duration", 0.042, {"model": "gpt-4o"}, "histogram")
```

### 审计

所有护栏裁决自动记录到审计通道，包含：
- 操作类型（check_input/check_tool_call/check_output）
- 裁决结果
- 触发规则
- 时间戳
