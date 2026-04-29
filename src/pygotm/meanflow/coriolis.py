r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The Coriolis rotation \label{sec:coriolis}
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import math

import numba
import numpy as np

from pygotm.meanflow.meanflow import MeanflowState

__all__ = [
    "coriolis",
    "step_coriolis_batch",
]


@numba.njit(cache=True)
def _coriolis_kernel(
    nlev: int,
    cosomega: float,
    sinomega: float,
    u: np.ndarray,
    v: np.ndarray,
    usprof: np.ndarray,
    vsprof: np.ndarray,
) -> None:
    for i in range(1, nlev + 1):
        ul = u[i] + usprof[i]
        vl = v[i] + vsprof[i]
        ua = ul
        ul = ul * cosomega + vl * sinomega
        vl = -ua * sinomega + vl * cosomega
        u[i] = ul - usprof[i]
        v[i] = vl - vsprof[i]


@numba.njit(parallel=True, cache=True)
def step_coriolis_batch(
    batch_size: int,
    nlev: int,
    cosomega: float,
    sinomega: float,
    u: np.ndarray,
    v: np.ndarray,
    usprof: np.ndarray,
    vsprof: np.ndarray,
) -> None:
    """Batch Coriolis rotation: process batch_size columns in parallel."""
    for b in numba.prange(batch_size):
        _coriolis_kernel(nlev, cosomega, sinomega, u[b], v[b], usprof[b], vsprof[b])


def coriolis(
    state: MeanflowState,
    nlev: int,
    dt: float,
    usprof: np.ndarray | None = None,
    vsprof: np.ndarray | None = None,
) -> None:
    """Apply the Coriolis rotation to horizontal velocities for one column."""
    assert state.u is not None
    assert state.v is not None

    _usprof = usprof if usprof is not None else np.zeros(nlev + 1)
    _vsprof = vsprof if vsprof is not None else np.zeros(nlev + 1)

    omega = state.cori * dt
    cosomega = math.cos(omega)
    sinomega = math.sin(omega)

    _coriolis_kernel(nlev, cosomega, sinomega, state.u, state.v, _usprof, _vsprof)
