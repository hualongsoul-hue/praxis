"""Praxis 测试配置和公共 fixture。"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from praxis.config.schemas import ModelCapabilities
from praxis.gateway.router import GatewayRouter


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """提供临时项目目录用于测试。"""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    return project_dir


def build_mock_gateway(default_model: str = "default") -> MagicMock:
    """构造一个 MagicMock(spec=GatewayRouter)，含最低可用配置。"""
    gw = MagicMock(spec=GatewayRouter)
    gw.config = MagicMock()
    gw.config.max_budget = None
    gw.config.default_model = default_model
    gw.capabilities.return_value = ModelCapabilities()
    return gw


@pytest.fixture
def mock_gateway() -> MagicMock:
    """提供共享的 Mock GatewayRouter，适用于不真实调用 LLM 的测试。"""
    return build_mock_gateway()
