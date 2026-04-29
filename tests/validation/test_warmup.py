"""Tests for validation/warmup.py — Numba kernel warm-up."""

from __future__ import annotations

from pygotm.validation.warmup import trigger_numba_jit


def test_trigger_numba_jit_returns_positive_float() -> None:
    elapsed = trigger_numba_jit()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_trigger_numba_jit_accepts_small_nlev() -> None:
    assert trigger_numba_jit(nlev=2) >= 0.0
