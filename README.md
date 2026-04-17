# Praxis — AI Agent Harness

生产级 AI Agent 框架，采用五层十四组件架构，支持 100+ LLM Provider 统一接入、MCP 工具集成、认知记忆系统和多 Agent 协调。

## 特性

- **统一模型网关**: 通过 LiteLLM 接入 100+ LLM Provider（OpenAI、Anthropic、Google 等），内置路由、负载均衡和成本预算
- **MCP 工具集成**: 支持 Model Context Protocol 工具发现/执行、Elicitation 用户确认、Sampling LLM 采样代理
- **三层护栏**: 输入护栏（提示注入检测）、工具护栏（权限管理）、输出护栏（敏感信息防护），支持绊线机制
- **认知记忆**: 四型记忆模型（语义/情景/程序/工作），向量检索，后台自动提取与整合
- **编排引擎**: 支持 ReAct 和 Plan-and-Execute 策略，自动终止、事件流和错误恢复
- **子代理委托**: 隔离上下文、独立工具子集和轮次限制
- **技能系统**: 三层渐进式披露，技能发现/注册/激活/版本管理
- **检查点恢复**: 自动检查点保存，跨窗口会话恢复
- **沙箱安全**: 文件系统白名单、网络出站控制、Shell 超时限制
- **全面可观测**: 结构化日志、OpenTelemetry 指标/追踪、100% 审计覆盖

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 4 · 编排层    S11 编排循环 │ S12 会话管理 │ S13 子代理协调 │
├──────────────────────────────────────────────────────────────┤
│ Layer 3 · 控制层    S8 护栏系统  │ S9 错误恢复  │ S10 验证引擎  │
├──────────────────────────────────────────────────────────────┤
│ Layer 2 · 核心能力  S5 工具系统 │ S6 记忆系统 │ S7 上下文 │ S14 技能 │
├──────────────────────────────────────────────────────────────┤
│ Layer 1 · 模型接入              S4 模型网关                   │
├──────────────────────────────────────────────────────────────┤
│ Layer 0 · 基础设施  S1 配置系统  │ S2 遥测系统  │ S3 持久化引擎 │
└──────────────────────────────────────────────────────────────┘
```

每层仅允许依赖同层或更低层的组件，禁止向上依赖。

## 技术栈

- **Python** 3.12+
- **包管理** uv
- **数据模型** Pydantic v2
- **LLM 接入** LiteLLM
- **工具协议** MCP (Model Context Protocol)
- **持久化** SQLAlchemy（SQLite/Redis/文件系统）
- **可观测** OpenTelemetry + structlog

## 快速开始

```bash
# 安装依赖
uv sync

# 创建配置（可选，所有字段均有默认值）
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入 API Key

# 运行测试
uv run pytest
```

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
    store = await create_store(PersistenceConfig())
    rule_engine = RuleEngine()
    rule_engine.register_builtin_rules()
    guardrails = GuardrailEngine(rule_engine, PermissionManager())

    factory = SessionFactory(
        store=store,
        session_config=SessionConfig(),
        orchestrator_config=OrchestratorConfig(),
        context_config=ContextConfig(),
    )

    session = factory.create_session(guardrails=guardrails)
    response = await session.run_turn("你好，请帮我分析这段代码")
    print(response.content)
    await store.close()

asyncio.run(main())
```

## 测试

```bash
uv run pytest tests/e2e/          # 8 个端到端场景（38 测试）
uv run pytest tests/integration/  # 跨组件集成（25 测试）
uv run pytest tests/benchmarks/   # 性能基准（9 测试）
uv run pytest tests/nfr/          # 非功能需求（41 测试）
```

## 文档

- [API 参考](docs/API.md)
- [使用指南](docs/GUIDE.md)
- [配置参考](docs/CONFIG_REFERENCE.md)
- [产品需求](docs/PRD.md)
- [开发路线图](docs/ROADMAP.md)

## 许可证

MIT
