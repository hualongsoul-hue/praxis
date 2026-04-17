"""LiteLLM Router 集成。

从 S1 加载 model_list 配置初始化 litellm.Router，
支持多 model_name 别名和同名负载均衡，运行时动态注册新模型。
"""

from typing import Any

from litellm import Router

from praxis.config.schemas import GatewayConfig
from praxis.exceptions import GatewayError
from praxis.gateway.callbacks import register_callbacks


class GatewayRouter:
    """LiteLLM Router 封装。

    薄封装策略：直接使用 litellm.Router 作为核心调用引擎，
    在其上叠加 Praxis 特有的配置驱动和遥测集成。
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._router = self._build_router(config)
        register_callbacks()

    @staticmethod
    def _build_router(config: GatewayConfig) -> Router:
        """根据配置构建 litellm.Router 实例。"""
        if not config.model_list:
            raise GatewayError(
                "model_list 为空，至少需要配置一个模型部署",
                details={"config_field": "gateway.model_list"},
            )

        return Router(
            model_list=config.model_list,
            routing_strategy=config.routing_strategy,
            num_retries=config.num_retries,
            timeout=config.timeout,
        )

    @property
    def router(self) -> Router:
        """获取底层 litellm.Router 实例。"""
        return self._router

    @property
    def config(self) -> GatewayConfig:
        """获取当前网关配置。"""
        return self._config

    def get_model_names(self) -> list[str]:
        """获取所有已注册的模型别名列表。"""
        seen: set[str] = set()
        names: list[str] = []
        for deployment in self._router.model_list:
            name = deployment.get("model_name", "")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    def get_model_list(self) -> list[dict[str, Any]]:
        """获取完整模型部署列表。"""
        return list(self._router.model_list)

    def register_model(self, deployment: dict[str, Any]) -> None:
        """运行时动态注册新模型部署。

        Args:
            deployment: LiteLLM 模型部署配置，包含 model_name 和 litellm_params。
        """
        if "model_name" not in deployment:
            raise GatewayError(
                "模型部署配置缺少 model_name",
                details={"deployment": deployment},
            )
        self._router.set_model_list(self._router.model_list + [deployment])

    def unregister_model(self, model_name: str) -> int:
        """移除指定 model_name 的所有部署。返回移除数量。"""
        original_count = len(self._router.model_list)
        self._router.model_list = [
            d for d in self._router.model_list
            if d.get("model_name") != model_name
        ]
        return original_count - len(self._router.model_list)
