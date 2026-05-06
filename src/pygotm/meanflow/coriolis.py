"""
Coriolis rotation of horizontal velocity.

Implements GOTM Section 3.2.4 — applies an exact rotation of the horizontal
velocity vector :math:`(U, V)` through angle :math:`\\omega = f \\Delta t` at
each vertical level, where :math:`f` is the Coriolis parameter and
:math:`\\Delta t` is the time step.  This exactly satisfies the homogeneous
Coriolis system

.. math::

   \\frac{\\partial U}{\\partial t} = fV, \\quad
   \\frac{\\partial V}{\\partial t} = -fU \\comma

by applying the rotation matrix

.. math::

   \\begin{pmatrix} U^{n+1} \\\\ V^{n+1} \\end{pmatrix} =
   \\begin{pmatrix} \\cos\\omega & \\sin\\omega \\\\
                    -\\sin\\omega & \\cos\\omega \\end{pmatrix}
   \\begin{pmatrix} U^n \\\\ V^n \\end{pmatrix} \\comma

where :math:`\\omega = f \\Delta t`.

When Stokes drift profiles :math:`(u_s, v_s)` are supplied, they are **added**
to the Eulerian velocity before rotation and **subtracted** after, so that only
the Eulerian part of the horizontal velocity is rotated:

.. math::

   \\tilde{U} = U + u_s, \\quad \\tilde{V} = V + v_s \\comma

and the rotation is applied to :math:`(\\tilde{U}, \\tilde{V})`.

Authors (original Fortran): Hans Burchard, Karsten Bolding.
"""

import math

import numba
import numpy as np

from pygotm.meanflow.meanflow import MeanflowState

__all__ = [
    "coriolis",
    "step_coriolis_batch",
    "step_coriolis_single",
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
    """Apply the Coriolis rotation to horizontal velocities for one column.

    Parameters
    ----------
    state:
        MeanflowState with u, v (horizontal velocities), cori (Coriolis parameter f [rad/s]).
    nlev:
        Number of model layers.
    dt:
        Time step [s].  Rotation angle is omega = state.cori * dt.
    usprof:
        Stokes drift profile in x, shape (nlev+1,) [m/s]. If None, zero.
    vsprof:
        Stokes drift profile in y, shape (nlev+1,) [m/s]. If None, zero.
    """
    assert state.u is not None
    assert state.v is not None

    _usprof = usprof if usprof is not None else np.zeros(nlev + 1)
    _vsprof = vsprof if vsprof is not None else np.zeros(nlev + 1)

    omega = state.cori * dt
    cosomega = math.cos(omega)
    sinomega = math.sin(omega)

    _coriolis_kernel(nlev, cosomega, sinomega, state.u, state.v, _usprof, _vsprof)


step_coriolis_single = _coriolis_kernel
