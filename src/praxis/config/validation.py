"""配置验证与变更事件。

支持自定义验证器注册和配置变更监听。
"""

from collections.abc import Callable
from typing import Any

from praxis.config.settings import PraxisConfig
from praxis.exceptions import ConfigError

ConfigValidator = Callable[[PraxisConfig], None]
ConfigChangeListener = Callable[[dict[str, tuple[Any, Any]]], None]

_validators: list[ConfigValidator] = []
_change_listeners: list[ConfigChangeListener] = []


def register_validator(validator: ConfigValidator) -> None:
    """注册自定义配置验证器。

    验证器接收完整 PraxisConfig，校验失败时应抛出 ConfigError。
    在 load_config / reload_config 时自动执行所有已注册验证器。
    """
    _validators.append(validator)


def validate_config(config: PraxisConfig) -> None:
    """执行所有已注册验证器。校验失败时抛出 ConfigError。"""
    for validator in _validators:
        try:
            validator(config)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"自定义验证器 {validator.__name__} 执行失败: {exc}",
                details={"validator": validator.__name__, "error": str(exc)},
            ) from exc


def on_config_change(listener: ConfigChangeListener) -> None:
    """注册配置变更监听器。

    监听器接收变更字典 ``{key_path: (old_value, new_value)}``，
    在 reload_config 检测到配置变化时调用。
    """
    _change_listeners.append(listener)


def notify_change(changes: dict[str, tuple[Any, Any]]) -> None:
    """通知所有已注册的变更监听器。"""
    for listener in _change_listeners:
        listener(changes)
