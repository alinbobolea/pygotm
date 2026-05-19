# ruff: noqa: E501
"""
Solar zenith angle — translation of the corresponding function in ``shortwave_radiation.F90``.

Returns the solar zenith angle in degrees from day-of-year, decimal UTC hour,
longitude, and latitude.  Used by both
:func:`~pygotm.airsea.shortwave_radiation.shortwave_radiation` and
:func:`~pygotm.airsea.albedo_water.albedo_water`.

Solar declination is computed from a four-term Fourier series (Spencer 1971):

.. math::

   \\delta = 0.006918 - 0.399912\\cos\\theta_0 + 0.070257\\sin\\theta_0
           - 0.006758\\cos 2\\theta_0 + 0.000907\\sin 2\\theta_0
           - 0.002697\\cos 3\\theta_0 + 0.001480\\sin 3\\theta_0

where :math:`\\theta_0 = 2\\pi\\,d / 365.25`.  The hour angle accounts for
the observer's longitude.  :math:`\\cos\\zeta` is clamped to zero below the
horizon before the arc-cosine is taken.

Original author: Karsten Bolding.
"""

from __future__ import annotations

import math

from pygotm.constants import DEG_TO_RAD, PI, RAD_TO_DEG

__all__ = ["solar_zenith_angle"]


def solar_zenith_angle(yday: int, hh: float, dlon: float, dlat: float) -> float:
    """Return the solar zenith angle in degrees."""

    rlon = DEG_TO_RAD * dlon
    rlat = DEG_TO_RAD * dlat

    yrdays = 365.25
    th0 = 2.0 * PI * yday / yrdays
    th02 = 2.0 * th0
    th03 = 3.0 * th0

    sundec = (
        0.006918
        - 0.399912 * math.cos(th0)
        + 0.070257 * math.sin(th0)
        - 0.006758 * math.cos(th02)
        + 0.000907 * math.sin(th02)
        - 0.002697 * math.cos(th03)
        + 0.001480 * math.sin(th03)
    )
    thsun = (hh - 12.0) * 15.0 * DEG_TO_RAD + rlon

    coszen = math.sin(rlat) * math.sin(sundec) + math.cos(rlat) * math.cos(
        sundec
    ) * math.cos(thsun)
    if coszen < 0.0:
        coszen = 0.0

    return RAD_TO_DEG * math.acos(coszen)
