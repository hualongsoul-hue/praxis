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
    """Return tests that pytest can collect from module and top-level classes."""
    tree = ast.parse(source)
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and not has_fixture_decorator(node):
                functions.append(node)
            continue
        if not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"):
            continue
        has_constructor = any(
            isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name in {"__init__", "__new__"}
            for member in node.body
        )
        if has_constructor:
            continue
        functions.extend(
            member
            for member in node.body
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name.startswith("test_")
            and not has_fixture_decorator(member)
        )
    return functions


def missing_test_oracles(root: Path) -> list[str]:
    """Report repository tests that lack the minimum explicit oracle."""
    missing: list[str] = []
    paths = set(root.rglob("test_*.py")) | set(root.rglob("*_test.py"))
    for path in sorted(paths):
        source = path.read_text(encoding="utf-8")
        for node in collect_test_functions(source):
            if not has_observable_oracle(node):
                missing.append(f"{path.relative_to(root).as_posix()}::{node.name}")
    return missing


def implementation_coupled_gateway_mocks(root: Path) -> list[str]:
    """Find orchestration tests that bypass the public ModelGateway protocol."""

    allowed = {
        "integration/test_gateway_litellm.py",
        "test_gateway.py",
    }
    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "acompletion"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "router"
            ):
                findings.append(f"{relative}:{node.lineno}")
    return findings


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


def test_oracle_detector_collects_only_pytest_module_and_test_class_nodes() -> None:
    nodes = collect_test_functions(
        """
def test_module_level():
    def test_nested():
        assert operation()
    assert operation()

class HelperClass:
    def test_not_collectable(self):
        assert operation()

class TestCollectable:
    def test_direct_method(self):
        assert operation()

    def helper(self):
        def test_nested_method():
            assert operation()
"""
    )
    assert [node.name for node in nodes] == [
        "test_module_level",
        "test_direct_method",
    ]


def test_missing_oracles_scans_both_pytest_filename_patterns_once(
    tmp_path: Path,
) -> None:
    (tmp_path / "quality_test.py").write_text(
        "def test_suffix_only():\n    action()\n",
        encoding="utf-8",
    )
    (tmp_path / "test_overlap_test.py").write_text(
        "def test_both_patterns():\n    action()\n",
        encoding="utf-8",
    )

    assert missing_test_oracles(tmp_path) == [
        "quality_test.py::test_suffix_only",
        "test_overlap_test.py::test_both_patterns",
    ]


def test_implementation_coupling_detector_rejects_gateway_adapter_bypass(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "test_orchestration.py"
    candidate.write_text(
        "def test_turn(mock_gateway):\n"
        "    mock_gateway.router.acompletion.return_value = object()\n"
        "    assert mock_gateway is not None\n",
        encoding="utf-8",
    )

    assert implementation_coupled_gateway_mocks(tmp_path) == [
        "test_orchestration.py:2"
    ]


def test_repository_tests_have_an_explicit_observable_oracle() -> None:
    root = Path(__file__).resolve().parent

    assert missing_test_oracles(root) == []


def test_orchestration_tests_mock_only_public_gateway_protocol() -> None:
    root = Path(__file__).resolve().parent

    assert implementation_coupled_gateway_mocks(root) == []
