"""Repository policy for project-owned Python symbol visibility."""

import ast
from pathlib import Path

PYTHON_ROOTS = ("src/praxis", "tests", "scripts", "examples")


def is_single_leading_underscore(name: str) -> bool:
    """Return whether ``name`` is private without being a Python dunder name."""
    return name.startswith("_") and not name.startswith("__")


class PrivateSymbolVisitor(ast.NodeVisitor):
    """Collect forbidden private symbol definitions and accesses from one module."""

    def __init__(self, path: Path, root: Path) -> None:
        self.path = path.relative_to(root).as_posix()
        self.violations: list[str] = []
        self.scopes: list[str] = []

    def record(self, node: ast.AST, kind: str, name: str) -> None:
        if is_single_leading_underscore(name):
            self.violations.append(
                f"{self.path}:{node.lineno}:{node.col_offset + 1}: {kind} {name}"
            )

    def visit_Module(self, node: ast.Module) -> None:
        self.scopes.append("module")
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.record(node, "function", node.name)
        self.scopes.append("function")
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.record(node, "async function", node.name)
        self.scopes.append("function")
        self.generic_visit(node)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.record(node, "class", node.name)
        self.scopes.append("class")
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.scopes.append("function")
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.scopes[-1] in {"module", "class"}:
            for target in node.targets:
                for name_node in assignment_names(target):
                    self.record(name_node, "assignment", name_node.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.scopes[-1] in {"module", "class"}:
            for name_node in assignment_names(node.target):
                self.record(name_node, "annotated assignment", name_node.id)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name.partition(".")[0]
            self.record(node, "import alias", bound_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name
            self.record(node, "import alias", bound_name)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.record(node, "attribute access", node.attr)
        self.generic_visit(node)


def assignment_names(target: ast.expr) -> list[ast.Name]:
    """Return names bound by an assignment target, including unpacking targets."""
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, ast.Starred):
        return assignment_names(target.value)
    if isinstance(target, ast.Tuple | ast.List):
        return [name for element in target.elts for name in assignment_names(element)]
    return []


def collect_private_symbol_violations(root: Path) -> list[str]:
    """Collect project-owned single-leading-underscore Python symbols and accesses."""
    violations: list[str] = []
    for python_root in PYTHON_ROOTS:
        for path in sorted(root.joinpath(python_root).rglob("*.py")):
            visitor = PrivateSymbolVisitor(path, root)
            visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
            violations.extend(visitor.violations)
    return sorted(violations)


def test_project_python_uses_only_public_owned_symbols() -> None:
    root = Path(__file__).resolve().parents[1]

    violations = collect_private_symbol_violations(root)

    assert violations == [], "Forbidden project-owned private symbols:\n" + "\n".join(violations)
