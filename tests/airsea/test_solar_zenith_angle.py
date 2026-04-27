"""Tests for pygotm.airsea.solar_zenith_angle."""

from __future__ import annotations

import math

import pytest

from pygotm.airsea.solar_zenith_angle import solar_zenith_angle


def test_import_and_smoke() -> None:
    angle = solar_zenith_angle(81, 12.0, 0.0, 0.0)
    assert angle >= 0.0


def test_equinox_noon_at_equator_is_nearly_overhead() -> None:
    angle = solar_zenith_angle(81, 12.0, 0.0, 0.0)
    assert angle == pytest.approx(0.7015827459508952, rel=1.0e-12)
    assert angle < 1.0


def test_midnight_is_clipped_to_ninety_degrees() -> None:
    assert solar_zenith_angle(81, 0.0, 0.0, 0.0) == pytest.approx(90.0)


def test_longitude_shift_preserves_local_solar_noon() -> None:
    reference = solar_zenith_angle(81, 12.0, 0.0, 0.0)
    shifted = solar_zenith_angle(81, 10.0, 30.0, 0.0)
    assert shifted == pytest.approx(reference, rel=1.0e-12)


@pytest.mark.parametrize(
    ("yday", "hh", "dlon", "dlat"),
    [
        (1, 12.0, 0.0, 60.0),
        (81, 6.0, -45.0, 30.0),
        (172, 12.0, 15.0, 40.0),
        (355, 18.0, -120.0, -20.0),
    ],
)
def test_zenith_angle_is_bounded_and_finite(
    yday: int,
    hh: float,
    dlon: float,
    dlat: float,
) -> None:
    angle = solar_zenith_angle(yday, hh, dlon, dlat)
    assert math.isfinite(angle)
    assert 0.0 <= angle <= 90.0
