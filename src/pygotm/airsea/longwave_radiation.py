# ruff: noqa: E501
"""
Net longwave (back) radiation — translation of ``longwave_radiation.F90``.

Computes the net outgoing longwave radiation at the sea surface [W m⁻²] using
one of six parameterisation methods, selected by an integer constant:

* ``CLARK`` (3) — Clark et al. (1974)
* ``HASTENRATH_LAMB`` (4) — Hastenrath & Lamb (1978)
* ``BIGNAMI`` (5) — Bignami et al. (1995)
* ``BERLIAND_BERLIAND`` (6) — Berliand & Berliand (1952)
* ``JOSEY1`` (7) — Josey et al. (2003), equation J1 (eq. 9)
* ``JOSEY2`` (8) — Josey et al. (2003), equation J2 (eq. 14)

The latitude must be provided in degrees.  A 91-element cloud correction
factor table indexed by integer absolute latitude is used by the Clark,
Hastenrath–Lamb, and Berliand–Berliand methods.  The vapour pressure ``ea``
(or specific humidity ``qa``) must be set in the ``AirSeaState`` object by a
prior call to :func:`~pygotm.airsea.humidity.humidity`.

Original FORTRAN authors: Adolf Stips, Hans Burchard, Karsten Bolding.
"""

from __future__ import annotations

import math

from pygotm.airsea.airsea_variables import (
    BERLIAND_BERLIAND,
    BIGNAMI,
    CLARK,
    HASTENRATH_LAMB,
    JOSEY1,
    JOSEY2,
    AirSeaState,
    bolz,
    emiss,
)

__all__ = ["longwave_radiation"]

_CLOUD_CORRECTION_FACTOR = (
    0.497202,
    0.501885,
    0.506568,
    0.511250,
    0.515933,
    0.520616,
    0.525299,
    0.529982,
    0.534665,
    0.539348,
    0.544031,
    0.548714,
    0.553397,
    0.558080,
    0.562763,
    0.567446,
    0.572129,
    0.576812,
    0.581495,
    0.586178,
    0.590861,
    0.595544,
    0.600227,
    0.604910,
    0.609593,
    0.614276,
    0.618959,
    0.623641,
    0.628324,
    0.633007,
    0.637690,
    0.642373,
    0.647056,
    0.651739,
    0.656422,
    0.661105,
    0.665788,
    0.670471,
    0.675154,
    0.679837,
    0.684520,
    0.689203,
    0.693886,
    0.698569,
    0.703252,
    0.707935,
    0.712618,
    0.717301,
    0.721984,
    0.726667,
    0.731350,
    0.736032,
    0.740715,
    0.745398,
    0.750081,
    0.754764,
    0.759447,
    0.764130,
    0.768813,
    0.773496,
    0.778179,
    0.782862,
    0.787545,
    0.792228,
    0.796911,
    0.801594,
    0.806277,
    0.810960,
    0.815643,
    0.820326,
    0.825009,
    0.829692,
    0.834375,
    0.839058,
    0.843741,
    0.848423,
    0.853106,
    0.857789,
    0.862472,
    0.867155,
    0.871838,
    0.876521,
    0.881204,
    0.885887,
    0.890570,
    0.895253,
    0.899936,
    0.904619,
    0.909302,
    0.913985,
    0.918668,
)


def _fortran_nint(value: float) -> int:
    return int(math.floor(value + 0.5))


def longwave_radiation(
    state: AirSeaState,
    method: int,
    dlat: float,
    tw: float,
    ta: float,
    cloud: float,
) -> float:
    """Return net longwave radiation in W/m² using GOTM's selected method."""

    ccf = _CLOUD_CORRECTION_FACTOR[_fortran_nint(abs(dlat))]

    if method == CLARK:
        x1 = (1.0 - ccf * cloud * cloud) * (tw**4)
        x2 = 0.39 - 0.05 * math.sqrt(state.ea * 0.01)
        x3 = 4.0 * (tw**3) * (tw - ta)
        return -emiss * bolz * (x1 * x2 + x3)

    if method == HASTENRATH_LAMB:
        x1 = (1.0 - ccf * cloud * cloud) * (tw**4)
        x2 = 0.39 - 0.056 * math.sqrt(1000.0 * state.qa)
        x3 = 4.0 * (tw**3) * (tw - ta)
        return -emiss * bolz * (x1 * x2 + x3)

    if method == BIGNAMI:
        ccf = 0.1762
        x1 = (1.0 + ccf * cloud * cloud) * ta**4
        x2 = 0.653 + 0.00535 * (state.ea * 0.01)
        x3 = emiss * (tw**4)
        return -bolz * (-x1 * x2 + x3)

    if method == BERLIAND_BERLIAND:
        x1 = (1.0 - 0.6823 * cloud * cloud) * ta**4
        x2 = 0.39 - 0.05 * math.sqrt(0.01 * state.ea)
        x3 = 4.0 * ta**3 * (tw - ta)
        return -emiss * bolz * (x1 * x2 + x3)

    if method == JOSEY1:
        x1 = emiss * tw**4
        x2 = (10.77 * cloud + 2.34) * cloud - 18.44
        x3 = 0.955 * (ta + x2) ** 4
        return -bolz * (x1 - x3)

    if method == JOSEY2:
        x1 = emiss * tw**4
        if state.ea < 10.0:
            state.ea = 10.0
        x2 = 34.07 + 4157.0 / math.log(2.1718e10 / state.ea)
        x2 = (10.77 * cloud + 2.34) * cloud - 18.44 + 0.84 * (x2 - ta + 4.01)
        x3 = 0.955 * (ta + x2) ** 4
        return -bolz * (x1 - x3)

    msg = "longwave_radiation: illegal longwave_radiation_method"
    raise ValueError(msg)
