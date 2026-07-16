"""纯函数配置加载器：默认值 → YAML → 环境变量 → 显式覆盖。"""

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

from praxis.config.settings import PraxisConfig
from praxis.exceptions import ConfigError

OBJECT_MAP = TypeAdapter(dict[str, object])


def merge_config(base: dict[str, object], update: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(base)
    for key, value in update.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            current_map = cast(dict[str, object], current)
            result[key] = merge_config(current_map, cast(Mapping[str, object], value))
        else:
            result[key] = deepcopy(value)
    return result


def environment_config(environ: Mapping[str, str]) -> dict[str, object]:
    """将 PRAXIS_<SECTION>__<FIELD> 映射为嵌套配置。

    ``PRAXIS_MODEL_API_KEY`` 被明确排除；它只由模型适配器在运行时读取。
    """
    result: dict[str, object] = {}
    for env_name, raw_value in environ.items():
        if not env_name.startswith("PRAXIS_") or env_name == "PRAXIS_MODEL_API_KEY":
            continue
        path = env_name.removeprefix("PRAXIS_").lower().split("__")
        if len(path) < 2:
            continue
        parsed = yaml.safe_load(raw_value)
        value: object = raw_value if parsed is None else parsed
        cursor = result
        for segment in path[:-1]:
            child = cursor.setdefault(segment, {})
            if not isinstance(child, dict):
                child = {}
                cursor[segment] = child
            cursor = cast(dict[str, object], child)
        cursor[path[-1]] = value
    return result


def load_yaml_config(path: Path) -> dict[str, object]:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"读取配置文件失败: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件不是有效 YAML: {path}") from exc
    if raw is None:
        return {}
    try:
        return OBJECT_MAP.validate_python(raw)
    except ValidationError as exc:
        raise ConfigError("配置文件根节点必须是映射") from exc


def load_config(
    config_path: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    **overrides: object,
) -> PraxisConfig:
    """加载一个独立、不可变的配置快照，不修改任何进程级状态。"""
    data: dict[str, object] = {}
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"配置文件不存在: {path}")
        data = load_yaml_config(path)

    env_data = environment_config(os.environ if environ is None else environ)
    merged = merge_config(data, env_data)
    merged = merge_config(merged, overrides)
    return PraxisConfig.model_validate(merged)


def get_component_config[ModelT: BaseModel](
    config: PraxisConfig,
    name: str,
    expected_type: type[ModelT] | None = None,
) -> BaseModel | ModelT:
    """从显式配置快照提取组件切片，不依赖全局“当前配置”。"""
    component = getattr(config, name, None)
    if not isinstance(component, BaseModel):
        raise ConfigError(
            f"未知的组件配置: {name}",
            details={"available": list(PraxisConfig.model_fields)},
        )
    if expected_type is not None and not isinstance(component, expected_type):
        raise ConfigError(f"组件配置类型不匹配: {name}")
    return component
