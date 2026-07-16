"""Repository-wide secret scanning with value-free findings."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

SECRET_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("openai_key", re.compile(rb"\bsk-(?!example|test|fake)[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google_api_key", re.compile(rb"\bAIza[A-Za-z0-9_-]{35}\b")),
    (
        "jwt",
        re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "bearer_token",
        re.compile(
            rb"\bBearer\s+(?!\{|\$|<|example|test|fake)[A-Za-z0-9._-]{24,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "generic_api_secret",
        re.compile(
            rb"\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)\b\s*[:=]\s*"
            rb"[\"'](?!PRAXIS_|<|example|test|fake|sk-(?:example|test|fake)|\{|\$)"
            rb"[A-Za-z0-9._-]{20,}[\"']",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """A safe finding that deliberately omits the matched value."""

    path: str
    rule_name: str


def tracked_secret_candidates(root: Path) -> list[Path]:
    """Return tracked files plus untracked source/docs task surfaces."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    candidates = {
        root / relative_path
        for relative_path in result.stdout.decode("utf-8").split("\0")
        if relative_path
    }
    text_suffixes = frozenset(
        {".cfg", ".ini", ".json", ".lock", ".md", ".py", ".rst", ".toml", ".txt", ".yaml", ".yml"}
    )
    for directory_name in ("docs", "examples", "scripts", "src", "tests"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        candidates.update(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in text_suffixes
            and "__pycache__" not in path.parts
        )
    task_report = root / ".superpowers" / "sdd" / "task-5-report.md"
    if task_report.is_file():
        candidates.add(task_report)
    return sorted(candidates)


def scan_secret_files(
    candidates: Iterable[Path],
    *,
    root: Path,
) -> list[SecretFinding]:
    """Scan bytes and return only relative paths and rule names."""

    findings: list[SecretFinding] = []
    for path in candidates:
        if not path.is_file():
            continue
        content = path.read_bytes()
        relative_path = path.relative_to(root).as_posix()
        for rule_name, pattern in SECRET_RULES:
            if pattern.search(content) is not None:
                findings.append(
                    SecretFinding(path=relative_path, rule_name=rule_name)
                )
    return sorted(findings, key=lambda finding: (finding.path, finding.rule_name))


def format_secret_findings(findings: Iterable[SecretFinding]) -> str:
    """Format findings without serializing any matched value."""

    return ", ".join(
        f"{finding.path}:{finding.rule_name}"
        for finding in findings
    )


def test_repository_tracked_files_do_not_contain_secrets() -> None:
    root = Path(__file__).resolve().parents[1]
    findings = scan_secret_files(tracked_secret_candidates(root), root=root)

    assert findings == [], f"secret rules matched: {format_secret_findings(findings)}"


@pytest.mark.parametrize(
    ("rule_name", "secret"),
    [
        ("openai_key", "s" + "k-" + "A" * 32),
        ("github_token", "gh" + "p_" + "B" * 36),
        ("aws_access_key", "AK" + "IA" + "C" * 16),
        ("google_api_key", "AI" + "za" + "D" * 35),
        ("jwt", "ey" + "J" + "E" * 16 + "." + "F" * 16 + "." + "G" * 16),
        ("bearer_token", "Bearer " + "H" * 32),
    ],
    ids=[
        "openai_key",
        "github_token",
        "aws_access_key",
        "google_api_key",
        "jwt",
        "bearer_token",
    ],
)
def test_secret_scanner_detects_multiple_provider_shapes(
    tmp_path: Path,
    rule_name: str,
    secret: str,
) -> None:
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(secret, encoding="utf-8")

    findings = scan_secret_files([candidate], root=tmp_path)

    assert [(finding.path, finding.rule_name) for finding in findings] == [
        ("candidate.txt", rule_name)
    ]


def test_secret_failure_summary_never_contains_matched_value(tmp_path: Path) -> None:
    secret = "s" + "k-" + "Z" * 32
    candidate = tmp_path / "docs.md"
    candidate.write_text(secret, encoding="utf-8")
    findings = scan_secret_files([candidate], root=tmp_path)

    summary = format_secret_findings(findings)

    assert summary == "docs.md:openai_key"
    assert secret not in summary


def test_tracked_file_inventory_includes_task_surfaces() -> None:
    root = Path(__file__).resolve().parents[1]
    relative_paths = {
        path.relative_to(root).as_posix()
        for path in tracked_secret_candidates(root)
    }

    assert "README.md" in relative_paths
    assert "config.example.yaml" in relative_paths
    assert "docs/SECURITY.md" in relative_paths
    assert "examples/console.py" in relative_paths
    assert "tests/integration/test_gateway_live.py" in relative_paths
    assert "tests/integration/live_model_evidence.py" in relative_paths
    assert ".superpowers/sdd/task-5-report.md" in relative_paths
