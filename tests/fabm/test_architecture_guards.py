"""Architecture guards for the FABM Python bridge."""

from __future__ import annotations

from pathlib import Path


def test_fabm_bridge_does_not_introduce_forbidden_bindings_or_numba() -> None:
    root = Path("src/pygotm/fabm")
    checked = [
        root / "engine.py",
        root / "config.py",
        root / "coupling.py",
        root / "state.py",
    ]

    for path in checked:
        text = path.read_text(encoding="utf-8")
        assert "ctypes" not in text
        assert "f2py" not in text.lower()
        assert "import numba" not in text
        assert "@numba.njit" not in text
