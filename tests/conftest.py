"""Praxis 测试配置和公共 fixture。"""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """提供临时项目目录用于测试。"""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    return project_dir
