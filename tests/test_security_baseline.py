"""仓库级安全基线测试。"""

import re
from pathlib import Path


def test_examples_do_not_contain_hardcoded_api_keys() -> None:
    root = Path(__file__).resolve().parents[1]
    secret_pattern = re.compile("s" + r"k-[A-Za-z0-9_-]{20,}")
    candidates = [
        *root.joinpath("examples").rglob("*.py"),
        root / "config.example.yaml",
        root / "README.md",
    ]

    offenders = [
        path.relative_to(root).as_posix()
        for path in candidates
        if path.is_file() and secret_pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == [], f"发现硬编码 API Key: {offenders}"
