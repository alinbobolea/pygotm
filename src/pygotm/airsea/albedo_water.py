# ruff: noqa: E501
"""
Sea-surface albedo — translation of ``albedo_water.F90``.

Returns the fractional albedo over water as a function of solar zenith angle
and year-day.  Three methods are available, selected by an integer constant:

* ``CONST`` (0) — fixed zero albedo; use when shortwave forcing is already net
  (albedo removed in the observational product).
* ``PAYNE`` (1) — Payne (1972) look-up table (20 entries, 30°–40° N Atlantic)
  interpolated linearly in zenith angle.
* ``COGLEY`` (2) — Cogley (1979) table with bilinear interpolation in zenith
  angle (10° bins) and month-of-year (12 monthly midpoints).

Original FORTRAN authors: Karsten Bolding, Hans Burchard.
"""

from __future__ import annotations

import math

from pygotm.airsea.airsea_variables import COGLEY, CONST, PAYNE

__all__ = ["albedo_water", "albedo_payne", "albedo_cogley"]

_PAYNE_ALBEDO = (
    0.719,
    0.656,
    0.603,
    0.480,
    0.385,
    0.300,
    0.250,
    0.193,
    0.164,
    0.131,
    0.103,
    0.084,
    0.071,
    0.061,
    0.054,
    0.043,
    0.039,
    0.036,
    0.034,
    0.034,
)
_PAYNE_ZA = (
    90.0,
    88.0,
    86.0,
    84.0,
    82.0,
    80.0,
    78.0,
    76.0,
    74.0,
    70.0,
    66.0,
    62.0,
    58.0,
    54.0,
    50.0,
    40.0,
    30.0,
    20.0,
    10.0,
    0.0,
)
_PAYNE_DZA = (2.0,) * 8 + (4.0,) * 6 + (10.0,) * 5

_COGLEY_COLUMNS = (
    (1.0, 1.0, 0.301, 0.293, 0.171, 0.148, 0.160, 0.246, 0.342, 1.0, 1.0, 1.0),
    (1.0, 0.301, 0.319, 0.225, 0.16, 0.131, 0.145, 0.206, 0.294, 0.305, 1.0, 1.0),
    (0.301, 0.338, 0.229, 0.148, 0.116, 0.112, 0.114, 0.134, 0.202, 0.313, 0.301, 1.0),
    (
        0.339,
        0.240,
        0.155,
        0.105,
        0.088,
        0.084,
        0.086,
        0.098,
        0.136,
        0.216,
        0.321,
        0.355,
    ),
    (0.220, 0.161, 0.108, 0.084, 0.075, 0.073, 0.074, 0.08, 0.099, 0.144, 0.210, 0.241),
    (0.145, 0.111, 0.085, 0.073, 0.068, 0.067, 0.068, 0.071, 0.08, 0.103, 0.138, 0.161),
    (
        0.103,
        0.086,
        0.073,
        0.067,
        0.065,
        0.064,
        0.064,
        0.066,
        0.071,
        0.082,
        0.100,
        0.111,
    ),
    (
        0.083,
        0.074,
        0.067,
        0.064,
        0.063,
        0.063,
        0.063,
        0.064,
        0.066,
        0.072,
        0.081,
        0.087,
    ),
    (
        0.072,
        0.067,
        0.064,
        0.063,
        0.064,
        0.064,
        0.064,
        0.063,
        0.063,
        0.066,
        0.071,
        0.074,
    ),
    (
        0.066,
        0.064,
        0.063,
        0.064,
        0.066,
        0.068,
        0.067,
        0.064,
        0.063,
        0.064,
        0.066,
        0.068,
    ),
)
_COGLEY_TABLE = tuple(
    tuple(column[month] for column in _COGLEY_COLUMNS) for month in range(12)
)
_COGLEY_ZA = (90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0)
_COGLEY_DZ = 10.0
_COGLEY_TIM = (
    1.0,
    32.0,
    60.0,
    91.0,
    121.0,
    152.0,
    182.0,
    213.0,
    244.0,
    274.0,
    305.0,
    335.0,
)
_COGLEY_DT = 365.25 / 12.0


def albedo_water(method: int, zenith_angle: float, yday: int) -> float:
    """Return the sea-surface albedo for the selected GOTM method."""

    if method == CONST:
        return 0.0
    if method == PAYNE:
        return albedo_payne(zenith_angle)
    if method == COGLEY:
        return albedo_cogley(zenith_angle, yday)

    msg = "A non-valide albedo method has been used"
    raise ValueError(msg)


def albedo_payne(zen: float) -> float:
    """Payne (1972) water albedo as a function of solar zenith angle."""

    if zen >= 74.0:
        jab = int(0.5 * (90.0 - zen) + 1.0)
    elif zen >= 50.0:
        jab = int(0.23 * (74.0 - zen) + 9.0)
    else:
        jab = int(0.10 * (50.0 - zen) + 15.0)

    if jab == 20:
        return _PAYNE_ALBEDO[jab - 1]

    dzen = (_PAYNE_ZA[jab - 1] - zen) / _PAYNE_DZA[jab - 1]
    return _PAYNE_ALBEDO[jab - 1] + dzen * (_PAYNE_ALBEDO[jab] - _PAYNE_ALBEDO[jab - 1])


def albedo_cogley(zen: float, yd: int) -> float:
    """Cogley (1979) water albedo using bilinear interpolation in angle and time."""

    jab = int(min(max((90.0 - zen) / _COGLEY_DZ + 1.0, 1.0), 10.0))
    tab = int(min(max(math.floor(yd / _COGLEY_DT) + 1.0, 1.0), 12.0))

    dzen = (_COGLEY_ZA[jab - 1] - zen) / _COGLEY_DZ
    dti = (yd - _COGLEY_TIM[tab - 1]) / _COGLEY_DT

    jab1 = min(jab + 1, 10)
    tab1 = 1 if tab == 12 else tab + 1

    r1 = _COGLEY_TABLE[tab - 1][jab - 1] + dzen * (
        _COGLEY_TABLE[tab - 1][jab1 - 1] - _COGLEY_TABLE[tab - 1][jab - 1]
    )
    r2 = _COGLEY_TABLE[tab1 - 1][jab - 1] + dzen * (
        _COGLEY_TABLE[tab1 - 1][jab1 - 1] - _COGLEY_TABLE[tab1 - 1][jab - 1]
    )

    value = r1 + dti * (r2 - r1)
    return max(min(value, 1.0), 0.0)
