# ruff: noqa: E501
"""
Theory-wave Stokes drift — translation of ``stokes_drift_theory.F90``.

Calculates Stokes drift profiles from surface wind speed using the empirical
'theory-wave' approximation of Li et al. (2017).  Two Numba kernels are
provided:

* :func:`stokes_drift_theory_srf` — Stokes drift averaged over the surface
  layer at a single depth.
* :func:`stokes_drift_theory` — full vertical profile plus surface values
  ``us0``, ``vs0``, and Stokes depth ``ds``.

A parallel batch variant :func:`stokes_drift_theory_batch` is also provided
for ensemble use.  The constant :data:`US0_TO_U10` = 0.0162 is the ratio of
surface Stokes drift to 10-m wind speed.

Original FORTRAN authors: Qing Li; re-added to GOTM by Brandon Reichl.
"""

import math

import numba
import numpy as np

from pygotm.constants import PI

__all__ = [
    "US0_TO_U10",
    "stokes_drift_theory",
    "stokes_drift_theory_batch",
    "stokes_drift_theory_srf",
]

US0_TO_U10: float = 0.0162
_SMALL: float = 1.0e-12


@numba.njit(cache=True)
def stokes_drift_theory_srf(
    u10: float,
    z_srf: float,
    gravity: float,
    us0_to_u10: float = US0_TO_U10,
) -> float:
    """Return Stokes drift averaged over the surface layer."""

    stokes_srf = 0.0
    if u10 <= 0.0:
        return stokes_srf

    u19p5_to_u10 = 1.075
    fm_to_fp = 1.296
    r_loss = 0.667

    us0 = us0_to_u10 * u10
    hm0 = 0.0246 * u10**2
    tmp = 2.0 * PI * u19p5_to_u10 * u10
    fp = 0.877 * gravity / tmp
    fm = fm_to_fp * fp
    stokes_trans = 0.125 * PI * r_loss * fm * hm0**2
    kphil = 0.176 * us0 / stokes_trans
    kstar = kphil * 2.56

    z0 = abs(z_srf)
    if z0 <= 1.0e-4:
        return 0.0

    z0i = 1.0 / z0
    r1 = (0.151 / kphil * z0i - 0.84) * (1.0 - math.exp(-2.0 * kphil * z0))
    r2 = (
        -(0.84 + 0.0591 / kphil * z0i)
        * math.sqrt(2.0 * PI * kphil * z0)
        * math.erfc(math.sqrt(2.0 * kphil * z0))
    )
    r3 = (0.0632 / kstar * z0i + 0.125) * (1.0 - math.exp(-2.0 * kstar * z0))
    r4 = (
        (0.125 + 0.0946 / kstar * z0i)
        * math.sqrt(2.0 * PI * kstar * z0)
        * math.erfc(math.sqrt(2.0 * kstar * z0))
    )
    stokes_srf = us0 * (0.715 + r1 + r2 + r3 + r4)
    return stokes_srf


@numba.njit(cache=True)
def stokes_drift_theory(
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    u10: float,
    v10: float,
    gravity: float,
    stokes_srf: np.ndarray,
    usprof: np.ndarray,
    vsprof: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate a Li et al. (2017) empirical theory-wave Stokes profile."""

    for k in range(nlev + 1):
        stokes_srf[k] = 0.0
        usprof[k] = 0.0
        vsprof[k] = 0.0

    us0 = 0.0
    vs0 = 0.0
    ds = 0.0

    wind_speed = math.sqrt(u10 * u10 + v10 * v10)
    if wind_speed <= 0.0:
        return us0, vs0, ds

    xcomp = u10 / wind_speed
    ycomp = v10 / wind_speed

    for k in range(nlev):
        stokes_srf[k] = stokes_drift_theory_srf(
            wind_speed,
            zi[nlev] - zi[k],
            gravity,
            US0_TO_U10,
        )
    stokes_srf[nlev] = 0.0

    for k in range(1, nlev + 1):
        tmp = (
            stokes_srf[k - 1] * (zi[nlev] - zi[k - 1])
            - stokes_srf[k] * (zi[nlev] - zi[k])
        ) / (zi[k] - zi[k - 1])
        usprof[k] = tmp * xcomp
        vsprof[k] = tmp * ycomp

    us0 = US0_TO_U10 * u10
    vs0 = US0_TO_U10 * v10
    ds = stokes_srf[0] * abs(zi[0]) / max(_SMALL, math.sqrt(us0 * us0 + vs0 * vs0))
    return us0, vs0, ds


@numba.njit(parallel=True, cache=True)
def stokes_drift_theory_batch(
    batch_size: int,
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    gravity: float,
    stokes_srf: np.ndarray,
    usprof: np.ndarray,
    vsprof: np.ndarray,
    scalars: np.ndarray,
) -> None:
    """Batch empirical theory-wave Stokes drift profile."""

    for b in numba.prange(batch_size):
        us0, vs0, ds = stokes_drift_theory(
            nlev,
            z[b],
            zi[b],
            u10[b],
            v10[b],
            gravity,
            stokes_srf[b],
            usprof[b],
            vsprof[b],
        )
        scalars[b, 0] = us0
        scalars[b, 1] = vs0
        scalars[b, 2] = ds
