"""Structural lower-bound checks for test observability.

This report detects tests with no explicit oracle. Passing it does not imply a
test is behaviorally rigorous; mock-only and implementation-coupled assertions
still require the focused audits in their owning test modules.
"""

import ast
from pathlib import Path


def call_name(node: ast.Call) -> str:
    """Return a stable dotted name for a direct call expression."""
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


class OracleVisitor(ast.NodeVisitor):
    """Find an oracle without counting assertions hidden in nested functions."""

    def __init__(self) -> None:
        self.found = False

    def visit_Assert(self, node: ast.Assert) -> None:
        self.found = True

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node)
        final_name = name.rsplit(".", maxsplit=1)[-1]
        if name in {"pytest.raises", "pytest.warns", "pytest.fail"}:
            self.found = True
        elif final_name.startswith("assert_"):
            self.found = True
        if not self.found:
            self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def has_fixture_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function is explicitly a pytest fixture."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
    return False


def has_observable_oracle(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Apply the structural lower bound to one test function."""
    visitor = OracleVisitor()
    for statement in node.body:
        visitor.visit(statement)
        if visitor.found:
            return True
    return False


def collect_test_functions(source: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return non-fixture test functions and methods from source."""
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        and not has_fixture_decorator(node)
    ]


def missing_test_oracles(root: Path) -> list[str]:
    """Report repository tests that lack the minimum explicit oracle."""
    missing: list[str] = []
    for path in sorted(root.rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        for node in collect_test_functions(source):
            if not has_observable_oracle(node):
                missing.append(f"{path.relative_to(root).as_posix()}::{node.name}")
    return missing


def test_oracle_detector_accepts_direct_pytest_contexts_and_failure() -> None:
    nodes = collect_test_functions(
        """
def test_raises():
    with pytest.raises(ValueError):
        action()

def test_warns():
    with pytest.warns(UserWarning):
        action()

def test_failure():
    pytest.fail('reason')
"""
    )
    assert all(has_observable_oracle(node) for node in nodes)


def test_oracle_detector_accepts_parametrized_delegated_assertion_helper() -> None:
    nodes = collect_test_functions(
        """
@pytest.mark.parametrize('value', [1, 2])
def test_values(value):
    assert_result(value)
"""
    )
    assert len(nodes) == 1
    assert has_observable_oracle(nodes[0]) is True


def test_oracle_detector_rejects_nonasserting_and_uncalled_nested_helpers() -> None:
    nodes = collect_test_functions(
        """
def test_missing():
    def nested():
        assert operation()
    arrange_and_act()
"""
    )
    assert len(nodes) == 1
    assert has_observable_oracle(nodes[0]) is False


def test_oracle_detector_exempts_fixture_functions() -> None:
    nodes = collect_test_functions(
        """
@pytest.fixture
def test_data():
    return object()
"""
    )
    assert nodes == []


def test_repository_tests_have_an_explicit_observable_oracle() -> None:
    root = Path(__file__).resolve().parent

    assert missing_test_oracles(root) == []
