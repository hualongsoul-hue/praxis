"""Repository policy for project-owned Python symbol visibility."""

import ast
from pathlib import Path

PYTHON_ROOTS = ("src/praxis", "tests", "scripts", "examples")


def is_single_leading_underscore(name: str) -> bool:
    """Return whether ``name`` is private without being a complete dunder name."""
    return name.startswith("_") and not (
        name.startswith("__") and name.endswith("__")
    )


class PrivateSymbolVisitor(ast.NodeVisitor):
    """Collect forbidden private symbol definitions and accesses from one module."""

    def __init__(self, path: Path, root: Path) -> None:
        self.path = path.relative_to(root).as_posix()
        self.violations: list[str] = []
        self.reported: set[tuple[int, int, str, str]] = set()

    def record(self, node: ast.AST, kind: str, name: str) -> None:
        if is_single_leading_underscore(name):
            key = (node.lineno, node.col_offset, kind, name)
            if key in self.reported:
                return
            self.reported.add(key)
            self.violations.append(
                f"{self.path}:{node.lineno}:{node.col_offset + 1}: {kind} {name}"
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.record(node, "function", node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.record(node, "async function", node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.record(node, "class", node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.record(node, "identifier", node.id)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self.record(node, "parameter", node.arg)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg is not None:
            self.record(node, "keyword argument", node.arg)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name.partition(".")[0]
            self.record(node, "import alias", bound_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name
            self.record(node, "import alias", bound_name)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.record(node, "attribute access", node.attr)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.record(node, "exception binding", node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self.record(node, "global declaration", name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self.record(node, "nonlocal declaration", name)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self.record(node, "match binding", node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self.record(node, "match binding", node.name)
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self.record(node, "match binding", node.rest)
        self.generic_visit(node)

    def visit_MatchClass(self, node: ast.MatchClass) -> None:
        for name in node.kwd_attrs:
            self.record(node, "match attribute", name)
        self.generic_visit(node)

    def visit_TypeVar(self, node: ast.TypeVar) -> None:
        self.record(node, "type parameter", node.name)
        self.generic_visit(node)

    def visit_ParamSpec(self, node: ast.ParamSpec) -> None:
        self.record(node, "type parameter", node.name)
        self.generic_visit(node)

    def visit_TypeVarTuple(self, node: ast.TypeVarTuple) -> None:
        self.record(node, "type parameter", node.name)
        self.generic_visit(node)


def collect_private_symbol_violations(root: Path) -> list[str]:
    """Collect project-owned single-leading-underscore Python symbols and accesses."""
    violations: list[str] = []
    for python_root in PYTHON_ROOTS:
        for path in sorted(root.joinpath(python_root).rglob("*.py")):
            visitor = PrivateSymbolVisitor(path, root)
            visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
            violations.extend(visitor.violations)
    return sorted(violations)


def collect_source_violations(root: Path, source: str) -> list[str]:
    """Scan one synthetic project module and return its policy violations."""
    path = root / "src" / "praxis" / "sample.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return collect_private_symbol_violations(root)


def violation_names(violations: list[str]) -> list[str]:
    """Extract reported identifier names from stable violation messages."""
    return sorted(violation.rsplit(maxsplit=1)[-1] for violation in violations)


def test_scanner_rejects_incomplete_dunders_and_allows_complete_dunders(
    tmp_path: Path,
) -> None:
    violations = collect_source_violations(
        tmp_path,
        """
class Public:
    def __init__(self):
        self.__class__

def __public_dunder__():
    pass

def public(obj):
    __private = 1
    obj.__private
""",
    )

    assert violation_names(violations) == ["__private", "__private"]


def test_scanner_rejects_parameters_and_local_assignments(tmp_path: Path) -> None:
    violations = collect_source_violations(
        tmp_path,
        """
def public(_arg, /, regular, *_args, _keyword, **_kwargs):
    _local = regular
    _annotated: int = 1
""",
    )

    assert violation_names(violations) == [
        "_annotated",
        "_arg",
        "_args",
        "_keyword",
        "_kwargs",
        "_local",
    ]


def test_scanner_rejects_compound_statement_bindings(tmp_path: Path) -> None:
    violations = collect_source_violations(
        tmp_path,
        """
def public():
    for _ in ():
        pass
    generated = [1 for _entry in ()]
    with manager() as _resource:
        pass
    try:
        pass
    except Exception as _error:
        pass
    if (_captured := True):
        pass
    match generated:
        case {"value": _matched, **_remaining}:
            pass
        case [*_tail]:
            pass
""",
    )

    assert violation_names(violations) == [
        "_",
        "_captured",
        "_entry",
        "_error",
        "_matched",
        "_remaining",
        "_resource",
        "_tail",
    ]


def test_scanner_rejects_private_import_bindings(tmp_path: Path) -> None:
    violations = collect_source_violations(
        tmp_path,
        """
import os as _operating_system
from pathlib import Path as _Path
""",
    )

    assert violation_names(violations) == ["_Path", "_operating_system"]


def test_project_python_uses_only_public_owned_symbols() -> None:
    root = Path(__file__).resolve().parents[1]

    violations = collect_private_symbol_violations(root)

    assert violations == [], "Forbidden project-owned private symbols:\n" + "\n".join(violations)
