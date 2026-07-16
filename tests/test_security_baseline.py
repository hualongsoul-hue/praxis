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
)

API_SECRET_KEY_LABEL = (
    rb"(?:[A-Za-z0-9]+[_-]+)*(?:api[_-]?key|access[_-]?token|client[_-]?secret)"
)
API_SECRET_QUOTED_LITERAL = (
    rb'''(?:"[A-Za-z0-9._${}<>{}\[\]%/+~=-]*"|'''
    rb'''\'[A-Za-z0-9._${}<>{}\[\]%/+~=-]*\')'''
)
API_SECRET_BARE_LITERAL = (
    rb"(?:[A-Za-z0-9._${}<>%/+~-]+={0,2})"
    rb"(?![A-Za-z0-9_.$/+%<>{}~=()\[\]-])"
)
API_SECRET_ASSIGNMENT = re.compile(
    rb"(?ix)(?:"
    rb"(?<![A-Za-z0-9_'\"?&/])"
    + API_SECRET_KEY_LABEL
    + rb"[ \t]{0,8}(?:=|:)[ \t]{0,8}"
    rb"(?P<direct_value>"
    + API_SECRET_QUOTED_LITERAL
    + rb"|"
    + API_SECRET_BARE_LITERAL
    + rb")|"
    rb"(?:os\.)?environ[ \t]{0,8}\[[ \t]{0,8}"
    rb"(?P<subscript_quote>[\"'])"
    + API_SECRET_KEY_LABEL
    + rb"(?P=subscript_quote)[ \t]{0,8}\][ \t]{0,8}=[ \t]{0,8}"
    rb"(?P<subscript_value>"
    + API_SECRET_QUOTED_LITERAL
    + rb")|"
    rb"(?:\{|,)[ \t]{0,8}(?P<label_quote>[\"'])"
    + API_SECRET_KEY_LABEL
    + rb"(?P=label_quote)[ \t]{0,8}:[ \t]{0,8}"
    rb"(?P<mapped_value>"
    + API_SECRET_QUOTED_LITERAL
    + rb")"
    rb")"
)
API_SECRET_ASSIGNMENT_ALLOWLIST = frozenset(
    {
        b"",
        b"<your-key>",
        b"[REDACTED]",
        b"None",
        b"abc123",
        b"expired",
        b"fake-contract-credential",
        b"intentionally-invalid-live-credential",
        b"null",
        b"secret-value-that-must-not-leak",
        b"sk-example0123456789abcdef",
        b"str",
        b"test-key",
        b"unit-secret",
        b"unit-test-model-key",
    }
)
ENVIRONMENT_REFERENCE = re.compile(
    rb"(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*|%[A-Za-z_][A-Za-z0-9_]*%)"
)


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """A safe finding that deliberately omits the matched value."""

    path: str
    rule_name: str


def tracked_secret_candidates(root: Path) -> list[Path]:
    """Return tracked files plus untracked source, docs, and SDD task artifacts."""

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
    task_artifacts = root / ".superpowers" / "sdd"
    if task_artifacts.is_dir():
        candidates.update(
            path
            for path in task_artifacts.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    return sorted(candidates)


def assignment_value_is_allowed(value: bytes) -> bool:
    """Return whether a complete assignment uses an explicit safe placeholder."""

    value_is_quoted = (
        len(value) >= 2
        and value[:1] == value[-1:]
        and value[:1] in {b"'", b'"'}
    )
    if value_is_quoted:
        value = value[1:-1]
    return (
        value in API_SECRET_ASSIGNMENT_ALLOWLIST
        or ENVIRONMENT_REFERENCE.fullmatch(value) is not None
        or (
            not value_is_quoted
            and re.fullmatch(
                rb"[a-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
                value,
            )
            is not None
        )
    )


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
        for assignment in API_SECRET_ASSIGNMENT.finditer(content):
            value = (
                assignment.group("direct_value")
                or assignment.group("subscript_value")
                or assignment.group("mapped_value")
            )
            if not assignment_value_is_allowed(value):
                findings.append(
                    SecretFinding(
                        path=relative_path,
                        rule_name="api_secret_assignment",
                    )
                )
                break
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


@pytest.mark.parametrize(
    ("key", "assignment_template"),
    [
        ("PRAXIS_MODEL_API_KEY", '$env:{key} = "{token}"'),
        ("PRAXIS_MODEL_API_KEY", "export {key}={token}"),
        ("PRAXIS_MODEL_API_KEY", "{key}: '{token}'"),
        ("PRAXIS_MODEL_API_KEY", '{key} = "{token}"'),
        ("INTERNAL_PROVIDER_API_KEY", "{key}={token}"),
    ],
    ids=["powershell", "bash", "yaml", "python", "prefixed-generic"],
)
def test_secret_scanner_detects_target_and_generic_assignments(
    tmp_path: Path,
    key: str,
    assignment_template: str,
) -> None:
    candidate = tmp_path / "assignment.txt"
    candidate.write_text(
        assignment_template.format(key=key, token="Q" * 32),
        encoding="utf-8",
    )

    findings = scan_secret_files([candidate], root=tmp_path)

    assert [(finding.path, finding.rule_name) for finding in findings] == [
        ("assignment.txt", "api_secret_assignment")
    ]


@pytest.mark.parametrize(
    "assignment_template",
    [
        'os.environ["{key}"] = "{token}"',
        'environ={{"{key}": "{token}"}}',
        'dict({key}="{token}")',
        "{key}={token}==",
    ],
    ids=["environ-subscript", "dict-literal", "dict-keyword", "base64-padding"],
)
def test_secret_scanner_detects_embedded_literal_assignments(
    tmp_path: Path,
    assignment_template: str,
) -> None:
    key = "PRAXIS_MODEL_API_KEY"
    token = "W9xY2z" * 6
    candidate = tmp_path / "embedded.py"
    candidate.write_text(
        assignment_template.format(key=key, token=token),
        encoding="utf-8",
    )

    findings = scan_secret_files([candidate], root=tmp_path)

    assert [(finding.path, finding.rule_name) for finding in findings] == [
        ("embedded.py", "api_secret_assignment")
    ]


@pytest.mark.parametrize("marker", ["+", "-", " "], ids=["added", "removed", "context"])
def test_secret_scanner_ignores_diff_markers_before_assignment(
    tmp_path: Path,
    marker: str,
) -> None:
    key = "PRAXIS_MODEL_API_KEY"
    token = "D7fF3e" * 6
    candidate = tmp_path / "review.diff"
    candidate.write_text(f'{marker}{key}="{token}"', encoding="utf-8")

    findings = scan_secret_files([candidate], root=tmp_path)

    assert [(finding.path, finding.rule_name) for finding in findings] == [
        ("review.diff", "api_secret_assignment")
    ]


@pytest.mark.parametrize(
    "assignment",
    [
        'PRAXIS_MODEL_API_KEY = "<your-key>"',
        "export PRAXIS_MODEL_API_KEY=${MODEL_KEY}",
        "$env:PRAXIS_MODEL_API_KEY = 'fake-contract-credential'",
        'PRAXIS_MODEL_API_KEY = "intentionally-invalid-live-credential"',
        'API_KEY = "sk-example0123456789abcdef"',
    ],
    ids=["angle-placeholder", "shell-variable", "contract-fake", "invalid-live", "example-key"],
)
def test_secret_assignment_allowlist_is_narrow_and_explicit(
    tmp_path: Path,
    assignment: str,
) -> None:
    candidate = tmp_path / "allowlisted.txt"
    candidate.write_text(assignment, encoding="utf-8")

    assert scan_secret_files([candidate], root=tmp_path) == []


@pytest.mark.parametrize(
    "assignment",
    [
        "API_KEY=abc123-production",
        "API_KEY=fake-contract-credential-production",
        "PRAXIS_MODEL_API_KEY=${MODEL_KEY}-leaked-suffix",
        "PRAXIS_MODEL_API_KEY=<your-key>-production-secret",
    ],
    ids=["extended-short-fake", "extended-fake", "extended-variable", "extended-placeholder"],
)
def test_secret_assignment_allowlist_rejects_extended_values(
    tmp_path: Path,
    assignment: str,
) -> None:
    candidate = tmp_path / "not-allowlisted.txt"
    candidate.write_text(assignment, encoding="utf-8")

    findings = scan_secret_files([candidate], root=tmp_path)

    assert [(finding.path, finding.rule_name) for finding in findings] == [
        ("not-allowlisted.txt", "api_secret_assignment")
    ]


@pytest.mark.parametrize(
    "source_template",
    [
        'environ={{"{key}": api_key}}',
        'environ={{"{key}": ""}}',
        'environ={{"{key}": None}}',
        'os.environ.get("{key}", "")',
        "{key}=",
    ],
    ids=["variable-reference", "empty-literal", "none-default", "get-default", "empty-shell"],
)
def test_secret_scanner_ignores_non_secret_mapping_values(
    tmp_path: Path,
    source_template: str,
) -> None:
    candidate = tmp_path / "safe.py"
    candidate.write_text(
        source_template.format(key="PRAXIS_MODEL_API_KEY"),
        encoding="utf-8",
    )

    assert scan_secret_files([candidate], root=tmp_path) == []


def test_assignment_failure_summary_never_contains_literal_value(tmp_path: Path) -> None:
    key = "PRAXIS_MODEL_API_KEY"
    token = "L8mN4p" * 6
    candidate = tmp_path / "mapping.py"
    candidate.write_text(
        f'environ={{"{key}": "{token}"}}',
        encoding="utf-8",
    )

    findings = scan_secret_files([candidate], root=tmp_path)
    summary = format_secret_findings(findings)

    assert summary == "mapping.py:api_secret_assignment"
    assert token not in summary


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
    assert ".superpowers/sdd/task-5-brief.md" in relative_paths
