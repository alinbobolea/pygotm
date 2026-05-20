# ruff: noqa: E501
"""
Shortwave radiation at the sea surface — translation of ``shortwave_radiation.F90``.

Computes the net downwelling shortwave flux [W m⁻²] from solar zenith angle,
year-day, longitude, latitude, and fractional cloud cover.  No albedo
correction is included; the caller must subtract the reflected fraction via
:func:`~pygotm.airsea.albedo_water.albedo_water`.

The formula follows Reed (1977) and Rosati & Miyakoda (1988, eq. 3.8),
adapted from the MOM-I implementation at INGV:

.. math::

   Q_s = Q_{\\mathrm{tot}}\\,(1 - 0.62\\,C + 0.0019\\,\\beta)

where :math:`Q_{\\mathrm{tot}} = Q_{\\mathrm{dir}} + Q_{\\mathrm{diff}}` is
the clear-sky total radiation, :math:`C` is fractional cloud cover, and
:math:`\\beta` is the solar noon altitude in degrees.  Internal constants:
atmospheric transmittance :math:`\\tau = 0.7`, ozone absorption
:math:`A_{\\mathrm{oz}} = 0.09`.

Original FORTRAN authors: Karsten Bolding, Hans Burchard.
"""

from __future__ import annotations

import math

from pygotm.constants import DEG_TO_RAD, PI, RAD_TO_DEG, SOLAR_CONSTANT_W_M2

__all__ = ["shortwave_radiation"]

_ECLIPTIC_OBLIQUITY = 23.439 * DEG_TO_RAD
_TAU = 0.7
_AOZONE = 0.09


def shortwave_radiation(
    zenith_angle: float,
    yday: int,
    dlon: float,
    dlat: float,
    cloud: float,
) -> float:
    """Return short-wave radiation at the sea surface in W/m²."""

    coszen = math.cos(DEG_TO_RAD * zenith_angle)
    if coszen <= 0.0:
        coszen = 0.0
        qatten = 0.0
    else:
        qatten = _TAU ** (1.0 / coszen)

    qzer = coszen * SOLAR_CONSTANT_W_M2
    qdir = qzer * qatten
    qdiff = ((1.0 - _AOZONE) * qzer - qdir) * 0.5
    qtot = qdir + qdiff

    rlat = DEG_TO_RAD * dlat
    eqnx = (yday - 81.0) / 365.0 * 2.0 * PI
    sunbet = math.sin(rlat) * math.sin(_ECLIPTIC_OBLIQUITY * math.sin(eqnx)) + (
        math.cos(rlat) * math.cos(_ECLIPTIC_OBLIQUITY * math.sin(eqnx))
    )
    sunbet = math.asin(sunbet) * RAD_TO_DEG

    qshort = qtot * (1.0 - 0.62 * cloud + 0.0019 * sunbet)
    if qshort > qtot:
        qshort = qtot

    return qshort
