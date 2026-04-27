"""Tests for pygotm.airsea.shortwave_radiation."""

from __future__ import annotations

import math

import pytest

from pygotm.airsea.shortwave_radiation import shortwave_radiation


def test_import_and_smoke() -> None:
    flux = shortwave_radiation(45.0, 100, 0.0, 45.0, 0.3)
    assert math.isfinite(flux)


def test_nighttime_flux_is_zero() -> None:
    assert shortwave_radiation(100.0, 81, 0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_clear_sky_equatorial_noon_matches_fortran_formula() -> None:
    assert shortwave_radiation(0.0, 81, 0.0, 0.0, 0.0) == pytest.approx(1086.75)


def test_clouds_reduce_shortwave_flux() -> None:
    clear = shortwave_radiation(30.0, 172, 0.0, 30.0, 0.0)
    cloudy = shortwave_radiation(30.0, 172, 0.0, 30.0, 0.8)
    assert cloudy < clear
    assert cloudy > 0.0


def test_flux_is_capped_by_clear_sky_total() -> None:
    flux = shortwave_radiation(0.0, 81, 0.0, 0.0, 0.0)
    assert flux <= 1086.75 + 1.0e-12


@pytest.mark.parametrize(
    ("zenith_angle", "yday", "dlon", "dlat", "cloud"),
    [
        (5.0, 81, 0.0, 0.0, 0.2),
        (45.0, 172, 10.0, 40.0, 0.5),
        (80.0, 355, -60.0, 70.0, 0.9),
    ],
)
def test_outputs_are_finite_for_representative_inputs(
    zenith_angle: float,
    yday: int,
    dlon: float,
    dlat: float,
    cloud: float,
) -> None:
    flux = shortwave_radiation(zenith_angle, yday, dlon, dlat, cloud)
    assert math.isfinite(flux)
    assert flux >= 0.0
