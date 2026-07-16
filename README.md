# Praxis

Praxis 是可嵌入的异步 Python Agent SDK/CLI，可作为 Windows 或 Linux 长期运行 Agent
服务的核心运行时。它提供严格配置、模型网关、并发安全会话、工具审批、检查点、审计、
MCP、记忆、技能、子代理和健康检查，但不绑定任何 Web 框架。

支持 Python `>=3.12,<3.15`。

## 安装

```shell
pip install praxis
```

按需安装可选能力：

```shell
pip install "praxis[mcp]"
pip install "praxis[redis]"
pip install "praxis[visual]"
pip install "praxis[otlp]"
```

## 快速开始

复制 [config.example.yaml](config.example.yaml) 为 `config.yaml`。模型密钥只能通过环境变量注入：

```powershell
$env:PRAXIS_MODEL_API_KEY = "<your-key>"
```

```bash
export PRAXIS_MODEL_API_KEY='<your-key>'
```

```python
import asyncio

from praxis import PraxisRuntime, load_config


async def main() -> None:
    config = load_config("config.yaml")
    async with PraxisRuntime(config) as runtime:
        async with runtime.session() as session:
            response = await session.run("请说明当前可用能力")
            print(response.content)


asyncio.run(main())
```

Runtime 可以被多个会话安全共享；同一个 `AgentSession` 会拒绝并发执行两个轮次。流式事件、
健康检查和关闭流程均为异步接口，便于嵌入 ASGI、任务队列或自定义守护进程。

```python
async with PraxisRuntime(load_config("config.yaml")) as runtime:
    health = await runtime.health()
    async with runtime.session() as session:
        async for event in session.run_stream("分析这个问题"):
            print(event.event_type, event.data)
```

## CLI

```shell
praxis version
praxis config validate config.yaml
praxis config show config.yaml
praxis doctor config.yaml
praxis chat config.yaml
```

`config show` 默认脱敏。`doctor` 不输出密钥，也不会发起计费模型请求。

## 安全默认值

- 模型固定通过配置中的默认别名调用；示例部署为 `openai/glm-5.1-openai`。
- `PRAXIS_MODEL_API_KEY` 不进入配置树、日志、审计、检查点或 CLI 输出。
- 文件工具在授权根目录为空时拒绝访问；Shell 和自主网络默认禁用。
- 网络工具默认阻止 URL 凭据、回环、私网、链路本地地址和越界重定向。
- 未配置审批处理器、审批超时或处理器异常时，写操作一律拒绝。
- 不可信 Shell 必须运行在容器或操作系统沙箱中；`ToolPolicy` 不是安全沙箱。

## 文档

- [架构与生命周期](docs/ARCHITECTURE.md)
- [配置参考](docs/CONFIGURATION.md)
- [生产部署](docs/DEPLOYMENT.md)
- [扩展 SDK](docs/EXTENDING.md)
- [安全模型](docs/SECURITY.md)
- [故障排查](docs/TROUBLESHOOTING.md)
- [贡献指南](CONTRIBUTING.md)
- [安全报告](SECURITY.md)

## 开发与发布门禁

```shell
uv lock
uv sync --all-extras --frozen
uv run ruff check .
uv run pyright
uv run pytest -W error --strict-config --strict-markers --cov=praxis --cov-branch --cov-fail-under=90
uv run pip-audit
uv build
```

MIT License。
