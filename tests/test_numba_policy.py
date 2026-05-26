"""Repository-wide Numba compatibility policy checks."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "pygotm"


def _module_uses_numba(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "numba" or alias.name.startswith("numba.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "numba" or (node.module or "").startswith("numba."):
                return True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                decorator_source = ast.unparse(decorator)
                if "numba" in decorator_source or decorator_source.startswith(
                    ("njit", "jit")
                ):
                    return True
    return False


def test_numba_callable_modules_do_not_use_postponed_annotations() -> None:
    """Numba-callable modules must expose real annotations at import time."""

    offenders: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "from __future__ import annotations" not in text:
            continue
        if _module_uses_numba(ast.parse(text)):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert not offenders, "\n".join(offenders)
