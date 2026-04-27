"""Tests for pygotm.turbulence.fk_craig."""

from __future__ import annotations

import math

import pytest

from pygotm.turbulence.fk_craig import fk_craig


def test_import() -> None:
    assert callable(fk_craig)


def test_matches_craig_banner_formula() -> None:
    assert fk_craig(0.02, 100.0) == pytest.approx(100.0 * 0.02**3)


def test_zero_friction_velocity_gives_zero_flux() -> None:
    assert fk_craig(0.0, 75.0) == 0.0


def test_cubic_scaling_is_preserved() -> None:
    base = fk_craig(0.03, 120.0)
    doubled = fk_craig(0.06, 120.0)

    assert doubled == pytest.approx(8.0 * base)
    assert math.isfinite(doubled)
