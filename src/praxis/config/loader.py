"""配置加载与分发。

提供 load_config / get_subsystem_config / reload_config 三个核心接口。
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from praxis.config.settings import PraxisConfig, set_yaml_data
from praxis.config.validation import notify_change, validate_config
from praxis.exceptions import ConfigError

_current_config: PraxisConfig | None = None
_config_path: Path | None = None


def load_config(
    config_path: Path | str | None = None,
    **overrides: Any,
) -> PraxisConfig:
    """加载 Praxis 配置。

    配置来源优先级（从低到高）：默认值 → YAML 文件 → 环境变量 → overrides 参数。

    Args:
        config_path: YAML 配置文件路径，为 None 时仅使用默认值和环境变量。
        **overrides: 最高优先级的配置覆盖，传递给 PraxisConfig 构造函数。

    Returns:
        加载并验证后的 PraxisConfig 实例。

    Raises:
        ConfigError: 配置文件不存在或验证失败。
    """
    global _current_config, _config_path

    yaml_data: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path).resolve()
        if not path.exists():
            raise ConfigError(f"配置文件不存在: {path}")
        raw = path.read_text(encoding="utf-8")
        yaml_data = yaml.safe_load(raw) or {}
        _config_path = path
    else:
        _config_path = None

    set_yaml_data(yaml_data)
    _current_config = PraxisConfig(**overrides)
    validate_config(_current_config)

    return _current_config


def get_subsystem_config(name: str) -> BaseModel:
    """获取指定子系统的配置切片。

    子系统仅能通过此接口获取自身配置，实现命名空间隔离。

    Args:
        name: 子系统配置字段名（如 ``"gateway"``、``"telemetry"``）。

    Returns:
        对应子系统的配置模型实例。

    Raises:
        ConfigError: 配置未加载或子系统名称无效。
    """
    if _current_config is None:
        raise ConfigError("配置未加载，请先调用 load_config()")
    if not hasattr(_current_config, name):
        raise ConfigError(
            f"未知的子系统配置: {name}",
            details={"available": list(PraxisConfig.model_fields.keys())},
        )
    return getattr(_current_config, name)


def reload_config(**overrides: Any) -> PraxisConfig:
    """热更新配置：重新读取配置文件和环境变量，验证后替换当前配置。

    仅当配置发生变化时通知已注册的变更监听器。

    Returns:
        更新后的 PraxisConfig 实例。
    """
    old_config = _current_config
    new_config = load_config(_config_path, **overrides)

    if old_config is not None:
        old_data = old_config.model_dump()
        new_data = new_config.model_dump()
        if old_data != new_data:
            changes = _diff_config(old_data, new_data)
            notify_change(changes)

    return new_config


def _diff_config(
    old: dict[str, Any],
    new: dict[str, Any],
    prefix: str = "",
) -> dict[str, tuple[Any, Any]]:
    """检测两个配置字典之间的差异。

    Returns:
        ``{key_path: (old_value, new_value)}`` 字典。
    """
    changes: dict[str, tuple[Any, Any]] = {}
    all_keys = set(old.keys()) | set(new.keys())
    for key in all_keys:
        path = f"{prefix}.{key}" if prefix else key
        old_val = old.get(key)
        new_val = new.get(key)
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            changes.update(_diff_config(old_val, new_val, path))
        elif old_val != new_val:
            changes[path] = (old_val, new_val)
    return changes
