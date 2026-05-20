"""
Vertical diffusion for face-centred variables — translation of ``diff_face.F90``.

Solves the one-dimensional diffusion equation for variables defined at grid
faces (interfaces) rather than cell centres.  Uses the same Crank–Nicolson
approach as :mod:`~pygotm.util.diff_center` but with the stencil appropriate
for face-centred quantities.  Source terms ``l_sour`` (linear, implicit) and
``q_sour`` (constant, explicit) are supported; relaxation is not included.

Boundary conditions (``bc_up``, ``bc_down``) are Dirichlet (``Dirichlet = 0``)
or Neumann (``Neumann = 1``).  The solved range spans indices ``1`` to
``nlev - 1`` (interior faces only).

Includes a bug-fix for ``nlev == 2`` attributed to Georg Umgiesser: when only
two layers are present, boundary diffusivities and values are replicated from
the single interior face to stabilise the system.

The Thomas algorithm (:func:`~pygotm.util.tridiagonal.tridiagonal`) solves the
resulting banded system.

Original FORTRAN author: Lars Umlauf.
"""

import numba
import numpy as np

from pygotm.util.tridiagonal import tridiagonal
from pygotm.util.util import Dirichlet as DIRICHLET
from pygotm.util.util import Neumann as NEUMANN

__all__ = [
    "DIRICHLET",
    "NEUMANN",
    "diff_face",
    "diff_face_batch",
]


@numba.njit(cache=True)
def diff_face(
    nlev: int,
    dt: float,
    cnpar: float,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    y: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    """Solve diffusion for a face-centred variable using Crank–Nicolson.

    Advances ``y`` at interior faces (indices 1 to nlev−1) one time step.
    Implicitness is controlled by ``cnpar``; optional linear (``l_sour``) and
    explicit constant (``q_sour``) source terms are included.  Boundary
    conditions ``bc_up`` and ``bc_down`` are ``DIRICHLET`` (prescribe value) or
    ``NEUMANN`` (prescribe flux).
    """

    # Bug fix Georg Umgiesser: set boundary nu and y values for nlev==2
    if nlev == 2:
        nu_y[0] = nu_y[1]
        nu_y[nlev] = nu_y[1]
        y[0] = y[1]
        y[nlev] = y[1]

    for i in range(2, nlev - 1):
        c = dt * (nu_y[i + 1] + nu_y[i]) / (h[i] + h[i + 1]) / h[i + 1]
        a = dt * (nu_y[i] + nu_y[i - 1]) / (h[i] + h[i + 1]) / h[i]
        linear_source = dt * l_sour[i]

        cu[i] = -cnpar * c
        au[i] = -cnpar * a
        bu[i] = 1.0 + cnpar * (a + c) - linear_source
        du[i] = (1.0 - (1.0 - cnpar) * (a + c)) * y[i]
        du[i] += (1.0 - cnpar) * (a * y[i - 1] + c * y[i + 1])
        du[i] += dt * q_sour[i]

    if bc_up == NEUMANN:
        a = dt * (nu_y[nlev - 1] + nu_y[nlev - 2]) / (h[nlev - 1] + h[nlev])
        a /= h[nlev - 1]
        linear_source = dt * l_sour[nlev - 1]

        au[nlev - 1] = -cnpar * a
        bu[nlev - 1] = 1.0 + cnpar * a - linear_source
        du[nlev - 1] = (1.0 - (1.0 - cnpar) * a) * y[nlev - 1]
        du[nlev - 1] += (1.0 - cnpar) * a * y[nlev - 2]
        du[nlev - 1] += dt * q_sour[nlev - 1]
        du[nlev - 1] += 2.0 * dt * y_up / (h[nlev - 1] + h[nlev])
    else:
        au[nlev - 1] = 0.0
        bu[nlev - 1] = 1.0
        du[nlev - 1] = y_up

    if bc_down == NEUMANN:
        c = dt * (nu_y[2] + nu_y[1]) / (h[1] + h[2]) / h[2]
        linear_source = dt * l_sour[1]

        cu[1] = -cnpar * c
        bu[1] = 1.0 + cnpar * c - linear_source
        du[1] = (1.0 - (1.0 - cnpar) * c) * y[1]
        du[1] += (1.0 - cnpar) * c * y[2]
        du[1] += dt * q_sour[1]
        du[1] += 2.0 * dt * y_down / (h[1] + h[2])
    else:
        bu[1] = 1.0
        cu[1] = 0.0
        du[1] = y_down

    tridiagonal(au, bu, cu, du, ru, qu, y, 1, nlev - 1)


@numba.njit(parallel=True, cache=True)
def diff_face_batch(
    batch_size: int,
    nlev: int,
    dt: float,
    cnpar: float,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    y: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    """Batch variant: process batch_size columns in parallel with numba.prange."""
    for b in numba.prange(batch_size):
        diff_face(
            nlev,
            dt,
            cnpar,
            h[b],
            bc_up,
            bc_down,
            y_up,
            y_down,
            nu_y[b],
            l_sour[b],
            q_sour[b],
            y[b],
            au[b],
            bu[b],
            cu[b],
            du[b],
            ru[b],
            qu[b],
        )
