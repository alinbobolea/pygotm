"""Tests for validation/warmup.py — Numba kernel warm-up."""

from __future__ import annotations

from pygotm.validation.warmup import trigger_numba_jit
from tests.fixtures import BUNDLED_CASES_ROOT


def test_trigger_numba_jit_returns_positive_float() -> None:
    elapsed = trigger_numba_jit(cases_root=BUNDLED_CASES_ROOT)
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_trigger_numba_jit_accepts_small_nlev() -> None:
    assert trigger_numba_jit(nlev=2, cases_root=BUNDLED_CASES_ROOT) >= 0.0
