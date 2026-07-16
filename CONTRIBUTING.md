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

不要创建或提交包含真实凭据的配置、日志、测试数据或文档。模型密钥只通过
`PRAXIS_MODEL_API_KEY` 注入。
