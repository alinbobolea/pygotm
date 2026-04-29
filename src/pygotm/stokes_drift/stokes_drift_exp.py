# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !IROUTINE: stokes_drift_exp
!
! !DESCRIPTION:
!  Calculate the Stokes drift profile from surface Stokes drift and the
!  Stokes penetration depth, assuming exponential profile.
!
! !REVISION HISTORY:
!  Original author(s): Qing Li
!
!EOP
!-----------------------------------------------------------------------
"""

import math

import numba
import numpy as np

__all__ = ["stokes_drift_exp", "stokes_drift_exp_batch"]


@numba.njit(cache=True)
def stokes_drift_exp(
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    us0: float,
    vs0: float,
    ds: float,
    usprof: np.ndarray,
    vsprof: np.ndarray,
) -> None:
    """Calculate a grid-cell-averaged exponential Stokes drift profile."""

    for k in range(nlev + 1):
        usprof[k] = 0.0
        vsprof[k] = 0.0

    if ds <= 0.0:
        raise ValueError("Stokes penetration depth ds must be positive")

    for k in range(1, nlev + 1):
        dz = zi[k] - zi[k - 1]
        kdz = 0.5 * dz / ds
        if kdz < 100.0:
            tmp = math.sinh(kdz) / kdz * math.exp((z[k] - zi[nlev]) / ds)
        else:
            tmp = math.exp((z[k] - zi[nlev]) / ds)
        usprof[k] = tmp * us0
        vsprof[k] = tmp * vs0


@numba.njit(parallel=True, cache=True)
def stokes_drift_exp_batch(
    batch_size: int,
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    us0: np.ndarray,
    vs0: np.ndarray,
    ds: np.ndarray,
    usprof: np.ndarray,
    vsprof: np.ndarray,
) -> None:
    """Batch exponential Stokes drift profile over independent columns."""

    for b in numba.prange(batch_size):
        stokes_drift_exp(
            nlev,
            z[b],
            zi[b],
            us0[b],
            vs0[b],
            ds[b],
            usprof[b],
            vsprof[b],
        )
