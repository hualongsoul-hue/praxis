# 2026-09-15 依赖更新记录

版本通过 PyPI 官方 JSON 元数据及项目迁移说明核对，直接依赖只选择未撤回的稳定版。
`pyproject.toml` 记录经过验证的最低版本与兼容上限，`uv.lock` 固定完整依赖图及发行文件哈希。
锁文件保留 Python 3.12–3.14、Windows/Linux 的平台分支。本次更新 56 个已有锁定包，新增
5 个包，移除 `httpx-sse`，合计锁定 119 个包（含项目本身和平台专用包）。

## 直接依赖和构建工具

| 范围 | 本次版本 |
| --- | --- |
| 运行 | Pydantic 2.13.5、LiteLLM 1.100.1、HTTPX 0.28.1、aiosqlite 0.22.1、SQLAlchemy 2.0.53 |
| 运行 | OpenTelemetry API/SDK 1.44.0、PyYAML 6.0.3、jsonschema 4.26.0、json-repair 0.63.4 |
| 可选 | MCP 2.2.0、HTTPX2 2.13.0、Redis 8.1.0、Playwright 1.62.0、OTLP gRPC exporter 1.44.0 |
| 开发 | Hypothesis 6.168.0、pytest 9.1.1、pytest-asyncio 1.4.0、pytest-cov 7.1.0 |
| 开发 | pip-audit 2.10.1、pip 26.2.1、Pyright 1.1.414、Ruff 0.16.7 |
| 构建 | Hatchling 1.32.0 |

除下述 LiteLLM 例外，以上均为核对日的最新稳定版。构建后端由构建隔离环境按
`build-system.requires` 解析，不属于项目的运行环境锁定包。

## 未强行升级的版本及原因

| 包 | 本次版本 | PyPI 最新稳定版 | 原因 |
| --- | --- | --- | --- |
| litellm | 1.100.1 | 1.101.0 | 严格警告模式下，响应模型构建触发 Pydantic 的 ReadOnly TypedDict 警告，模型请求发出前失败 |
| openai | 2.54.0 | 3.14.0 | LiteLLM 要求 `openai<3.0.0` |
| importlib-metadata | 8.9.0 | 9.0.1 | LiteLLM 要求 `<9.0` |
| pydantic-core | 2.46.5 | 2.49.0 | Pydantic 2.13.5 精确要求 `==2.46.5` |
| pyee | 13.0.1 | 14.0.0 | Playwright 1.62.0 要求 `<14` |

不添加依赖 override 来绕过上游约束。`uv pip list --outdated` 可能还列出 PyPI 上同名的
`praxis`，它不是本仓库的可升级依赖，不应安装来替换当前项目。

传递依赖中，`opentelemetry-semantic-conventions==0.65b0` 采用 beta 版本命名；这是稳定版
OpenTelemetry SDK 1.44.0 的精确要求，不代表项目主动启用了预发行版本解析。

LiteLLM 1.101.0 的最小复现（仅在独立诊断环境安装该版本后运行，无密钥或模型请求）：

```console
python -W error -c "from litellm.types.utils import ModelResponse; ModelResponse()"
```

其 `ChatCompletionReasoningItem.summary` 的 `ReadOnly` 注解触发 Pydantic 警告。本地真实
HTTP 契约测试和指定模型端点测试均复现这一根因；1.100.1 的同一最小复现通过。
项目没有关闭警告门禁、修改第三方源码或修改宿主的全局 warning filter。
未来解除 `!=1.101.0` 或采用新版本前，必须重新运行模型构建及真实传输测试。

## 适配与边界

- MCP 使用 v2 的字段、错误类型、能力属性和 HTTPX2；SDK 升级不等于自动启用新的协议版本，
  协商范围及原因见 [扩展文档](EXTENDING.md#mcp)。不保留 v1 Python API 兼容层。
- MCP 测试使用真实协议数据类型，并覆盖 stdio 子进程和真实 TCP HTTP 的工具、资源、Prompt、
  Sampling、Elicitation、认证头转发及会话关闭。
- Redis 8 使用其默认 RESP3，存储适配器保留字节数据和空值语义，接受客户端已解码的字符串。
- Playwright 升级后，启用视觉功能的部署需重新执行 `uv run playwright install chromium`，
  或用对应版本的受管浏览器镜像；Python 包升级不会自动安装浏览器。

## 验证命令

```console
uv sync --all-extras --frozen
uv run ruff check .
uv run pyright
uv run pytest -W error --strict-config --strict-markers --cov=praxis --cov-branch --cov-fail-under=90
uv run pip-audit
uv pip check
uv build
uv run python scripts/verify_wheel.py
```

真实端点测试仅从进程环境读取 `PRAXIS_MODEL_API_KEY`，不将密钥写入仓库或测试记录。
Linux/Windows 六项 CI 矩阵继续使用同一个 frozen 锁文件；本地 Windows 验证不替代 Linux CI。

已完成的本地验证（Windows）：

| 检查 | 结果 |
| --- | --- |
| Python 3.12.12 全量严格测试 | 1019 通过，14 项环境跳过，分支覆盖率 90.88% |
| Python 3.13.9 独立环境重点回归 | MCP、模型 HTTP 契约、Redis 边界共 68 项通过 |
| Python 3.14.4 独立环境重点回归 | 同一组 68 项通过 |
| Linux x86_64 / Python 3.12、3.13、3.14 | 全 extras 的 wheel-only dry-run 依赖解析通过；非 Linux 运行测试 |
| 指定真实模型端点 | 13 项通过；普通/流式/工具调用、取消、超时、错误映射及多模态能力分类 |
| Ruff / Pyright strict | 通过 / 0 个诊断 |
| pip-audit / uv pip check | 未发现已知漏洞 / 无依赖冲突 |
| uv lock --check / frozen 同步 | 通过 |
| wheel/sdist 构建与干净安装 | 通过；验证 CLI、SDK、py.typed、内置技能资源与缺失 extra 提示 |

真实端点的多模态测试通过表示“不支持”的错误分类仍准确，并不表示模型新增了图片、音频、
视频或文件理解能力。Redis 本轮验证为适配器边界回归，未连接外部 Redis 服务。

## 官方来源

- [PyPI 项目元数据](https://docs.pypi.org/api/json/)
- [LiteLLM 发布版本](https://pypi.org/project/litellm/)
- [MCP v2 迁移说明](https://py.sdk.modelcontextprotocol.io/migration/)
- [Redis 发布说明](https://github.com/redis/redis-py/releases)
- [Playwright 浏览器安装](https://playwright.dev/python/docs/browsers)
