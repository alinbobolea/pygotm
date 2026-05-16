"""Tests for pygotm.airsea.albedo_water."""

from __future__ import annotations

import math

import pytest

from pygotm.airsea.airsea_variables import CONST, PAYNE
from pygotm.airsea.albedo_water import albedo_cogley, albedo_payne, albedo_water


def test_import_and_smoke() -> None:
    assert albedo_water(PAYNE, 60.0, 100) > 0.0


def test_const_method_returns_zero() -> None:
    assert albedo_water(CONST, 60.0, 100) == pytest.approx(0.0)


def test_payne_returns_exact_tabulated_end_points() -> None:
    assert albedo_payne(90.0) == pytest.approx(0.719)
    assert albedo_payne(0.0) == pytest.approx(0.034)


def test_payne_interpolates_linearly_between_neighbouring_angles() -> None:
    # Between 76 deg (0.193) and 74 deg (0.164).
    assert albedo_payne(75.0) == pytest.approx(0.1785)


def test_cogley_returns_exact_table_value_at_grid_point() -> None:
    # yday 1 is the first tabulated time anchor and zenith 50 deg is an exact
    # grid point.
    assert albedo_cogley(50.0, 1) == pytest.approx(0.220)


@pytest.mark.parametrize(
    ("zen", "yday"),
    [(10.0, 1), (35.0, 90), (70.0, 200), (85.0, 365)],
)
def test_cogley_output_is_bounded_and_finite(zen: float, yday: int) -> None:
    albedo = albedo_cogley(zen, yday)
    assert math.isfinite(albedo)
    assert 0.0 <= albedo <= 1.0


def test_invalid_albedo_method_raises_value_error() -> None:
    with pytest.raises(ValueError, match="A non-valide albedo method has been used"):
        albedo_water(99, 60.0, 100)
