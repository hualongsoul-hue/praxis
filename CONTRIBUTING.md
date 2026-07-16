# 贡献指南

## 开发环境

```powershell
uv lock
uv sync --all-extras --frozen
```

## 提交要求

使用 Conventional Commits，例如 `fix(config): 拒绝未知配置字段`。每个行为变更先添加失败测试，再实施最小修复。

提交前运行：

```powershell
uv lock
uv sync --all-extras --frozen
uv run ruff check .
uv run pyright
uv run pytest -W error --strict-config --strict-markers --cov=praxis --cov-branch --cov-fail-under=90
uv run pip-audit
uv build
uv run python scripts/verify_wheel.py
```

真实模型验收默认因缺少凭据而跳过。需要验证指定 OpenAI 兼容端点时，先在当前进程设置
`PRAXIS_MODEL_API_KEY`，再执行：

```powershell
uv run pytest -m live_model tests/integration/test_gateway_live.py
```

该套件覆盖普通响应、流式响应、强制工具调用、取消、超时和错误映射；不要把密钥写进命令、
配置文件、测试数据或终端输出。

不要创建或提交包含真实凭据的配置、日志、测试数据或文档。模型密钥只通过
`PRAXIS_MODEL_API_KEY` 注入。
