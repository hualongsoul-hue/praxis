"""Bounded filesystem traversal primitives for search tools."""

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from praxis.exceptions import ToolPolicyViolationError
from praxis.tools.policy import ToolPolicy

UNSAFE_NESTED_QUANTIFIER = re.compile(r"\([^)]*[*+][^)]*\)\s*(?:[*+]|{)")
BACKREFERENCE = re.compile(r"\\[1-9]")


@dataclass(slots=True)
class SearchBudget:
    """Mutable counters confined to one worker-thread search invocation."""

    max_files: int
    max_bytes: int
    max_matches: int
    deadline: float
    files: int = 0
    bytes: int = 0
    matches: int = 0
    truncated: bool = False

    def check_deadline(self) -> bool:
        if time.monotonic() <= self.deadline:
            return True
        self.truncated = True
        return False

    def accept_file(self) -> bool:
        if self.files >= self.max_files or not self.check_deadline():
            self.truncated = True
            return False
        self.files += 1
        return True

    def accept_match(self) -> bool:
        if self.matches >= self.max_matches or not self.check_deadline():
            self.truncated = True
            return False
        self.matches += 1
        return True


def create_search_budget(policy: ToolPolicy) -> SearchBudget:
    """Create counters and deadline from one immutable tool policy."""
    return SearchBudget(
        max_files=policy.search_max_files,
        max_bytes=policy.search_max_bytes,
        max_matches=policy.search_max_matches,
        deadline=time.monotonic() + policy.search_timeout,
    )


def validate_regex(pattern: str, policy: ToolPolicy) -> re.Pattern[str]:
    """Reject constructs that can cause unbounded backtracking in Python re."""
    if len(pattern) > policy.search_max_pattern_length:
        raise ToolPolicyViolationError("正则表达式超过长度上限")
    if "(?" in pattern or BACKREFERENCE.search(pattern):
        raise ToolPolicyViolationError("正则表达式包含不允许的高级结构")
    if UNSAFE_NESTED_QUANTIFIER.search(pattern):
        raise ToolPolicyViolationError("正则表达式包含不安全的嵌套量词")
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ToolPolicyViolationError(f"无效的正则表达式: {exc}") from None


def collect_search_files(
    search_path: Path,
    policy: ToolPolicy,
    budget: SearchBudget,
) -> list[Path]:
    """Collect at most the configured number of regular files without following links."""
    checked = policy.check_path(search_path)
    if checked.is_file():
        return [checked] if budget.accept_file() else []
    if not checked.exists():
        return []

    files: list[Path] = []
    for root, directories, names in os.walk(checked, followlinks=False):
        root_path = policy.check_path(root)
        directories[:] = sorted(
            name
            for name in directories
            if not (root_path / name).is_symlink()
            and not (
                hasattr(Path, "is_junction")
                and (root_path / name).is_junction()
            )
        )
        for name in sorted(names):
            candidate = root_path / name
            try:
                resolved = policy.check_path(candidate)
            except ToolPolicyViolationError:
                continue
            if resolved.is_symlink() or not resolved.is_file():
                continue
            if not budget.accept_file():
                return files
            files.append(resolved)
    return files


def read_search_text(path: Path, budget: SearchBudget) -> str | None:
    """Read only bytes remaining in the aggregate search budget."""
    if not budget.check_deadline():
        return None
    remaining = budget.max_bytes - budget.bytes
    if remaining <= 0:
        budget.truncated = True
        return None
    try:
        with path.open("rb") as stream:
            data = stream.read(remaining + 1)
    except (OSError, PermissionError):
        return None
    if len(data) > remaining:
        data = data[:remaining]
        budget.truncated = True
    budget.bytes += len(data)
    return data.decode("utf-8", errors="ignore")


def append_truncation(results: list[str], budget: SearchBudget) -> None:
    """Append one stable truncation marker when any hard bound was reached."""
    if budget.truncated:
        results.append("... 搜索范围已截断")
