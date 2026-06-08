"""Repository-wide Numba compatibility policy checks."""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
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


def _run_child_import(tmp_path: Path, env: dict[str, str] | None = None) -> str:
    code = (
        "import inspect; "
        "from pathlib import Path; "
        "from pygotm.icethm import _util; "
        "source = inspect.getsourcefile(_util.freezing_temperature.py_func); "
        "assert source is not None and Path(source).is_file(), source; "
        "print(_util.freezing_temperature(35.0))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "cannot cache function" not in result.stderr
    return result.stdout.strip()


def test_icethm_cached_helper_imports_from_foreign_subprocess(
    tmp_path: Path,
) -> None:
    """External interpreters must import cached icethm helpers from real files."""

    value = float(_run_child_import(tmp_path))

    assert math.isclose(value, -2.0125, rel_tol=0.0, abs_tol=1.0e-12)


def test_icethm_cached_helper_imports_with_explicit_numba_cache_dir(
    tmp_path: Path,
) -> None:
    """Consumer-managed Numba cache directories must not break helper imports."""

    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba-cache")

    value = float(_run_child_import(tmp_path, env=env))

    assert math.isclose(value, -2.0125, rel_tol=0.0, abs_tol=1.0e-12)
