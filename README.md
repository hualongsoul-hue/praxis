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

### 实时交互控制台

仓库内的 [examples/interactive_console.py](examples/interactive_console.py) 是一个完整的
长生命周期控制台示例。它只创建一个 `PraxisRuntime` 和一个 `AgentSession`，持续复用会话上下文，
并实时渲染 `content_delta`、工具执行和终止事件：

```shell
copy config.example.yaml config.yaml  # Windows
cp config.example.yaml config.yaml     # Linux
uv run python examples/interactive_console.py --config config.yaml
```

模型密钥仍然只从 `PRAXIS_MODEL_API_KEY` 环境变量读取。控制台命令包括：

- `/health`：查看 Runtime、模型、存储和后台任务健康状态；
- `/attach <image|audio|video|file> <path>`：为下一轮排队一个本地附件；
- `/attachments`、`/clear`：查看或清除待发送附件；
- `/help`、`/quit`：查看帮助或优雅退出。

附件路径仍由 `inputs.allowed_paths`、模型能力和输入大小策略共同校验；控制台不会绕过 SDK
的路径、网络、护栏、审批、检查点或审计流程。流式输出期间按 `Ctrl+C` 只中止当前轮次，
不会直接跳过 Runtime 的关闭流程。

## 文本与多模态输入

`AgentSession.run()` 和 `run_stream()` 接受普通字符串，也接受由图片、音频、视频和文件组成的
`UserInput`。下面的本地路径必须位于 `inputs.allowed_paths` 的授权根目录内：

```python
from pathlib import Path

from praxis import FileInput, ImageInput, UserInput


request = UserInput(
    text="概括图片，并结合附件给出结论",
    parts=(
        ImageInput.from_path(Path("attachments/chart.png")),
        FileInput.from_path(Path("attachments/context.txt")),
    ),
)
response = await session.run(request)
print(response.content)
```

SDK 能安全解析和传输这四类输入，不代表每个模型端点都接受它们。部署的
`gateway.deployments[].capabilities` 是失败关闭的端点事实声明：只有经过至少三个独立健康窗口
稳定验证的模态才能设为 `true`；`config.example.yaml` 中的 `false` 也可能表示端点尚未产生足够稳定的成功证据。
未声明能力会在读取文件或发起网络请求之前本地失败。

当前指定端点的三窗口实测矩阵为：图片和视频 `true`，音频和文件 `false`（稳定映射为
`GatewayError/BadRequestError`）。这只是该端点的事实快照，不代表 SDK 的传输层限制。

单附件和单轮总大小默认分别限制为 20 MB、50 MB。`InputConfig` 的 schema 默认使用空授权根目录，
因此拒绝所有本地路径；快速开始使用的 `config.example.yaml` 为了让上例可运行，显式授权了
`./attachments`。远程 URL 默认禁用，必须设置 `inputs.remote_enabled: true`，并仍会执行协议、
DNS、重定向、私网地址和响应大小检查。

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
