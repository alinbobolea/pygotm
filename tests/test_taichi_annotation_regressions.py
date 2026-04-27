from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    PROJECT_ROOT / "src" / "pygotm",
    PROJECT_ROOT / "tests",
)
TAICHI_KERNEL_DECORATORS = {"ti.kernel", "ti_kernel"}
TAICHI_FUNC_DECORATORS = {"ti.func", "ti_func"}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        files.extend(sorted(root.rglob("*.py")))
    return files


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _decorator_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _function_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for decorator in node.decorator_list:
        name = _decorator_name(decorator)
        if name is not None:
            names.add(name)
    return names


def _has_future_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(alias.name == "annotations" for alias in node.names):
                return True
    return False


def _is_none_annotation(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant):
        return annotation.value is None
    if isinstance(annotation, ast.Name):
        return annotation.id == "None"
    return False


def test_no_future_annotations_in_taichi_modules() -> None:
    offenders: list[str] = []

    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        has_taichi_callable = any(
            _function_decorators(node)
            & (TAICHI_KERNEL_DECORATORS | TAICHI_FUNC_DECORATORS)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if has_taichi_callable and _has_future_annotations(tree):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert not offenders, (
        "Modules containing Taichi kernels/functions must not use "
        "`from __future__ import annotations` on the current stack:\n"
        + "\n".join(offenders)
    )


def test_no_none_return_annotations_on_taichi_kernels() -> None:
    offenders: list[str] = []

    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not (_function_decorators(node) & TAICHI_KERNEL_DECORATORS):
                continue
            if _is_none_annotation(node.returns):
                relative_path = path.relative_to(PROJECT_ROOT)
                offenders.append(f"{relative_path}:{node.lineno} {node.name}")

    assert not offenders, (
        "Taichi kernels must omit `-> None` because Taichi 1.7.4 miscompiles "
        "void kernels with explicit None returns:\n"
        + "\n".join(offenders)
    )
