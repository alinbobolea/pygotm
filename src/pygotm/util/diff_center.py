"""
Vertical diffusion for cell-centred variables — translation of ``diff_center.F90``.

Solves the one-dimensional diffusion equation with optional source terms and
relaxation toward observed values:

.. math::

   \\frac{\\partial Y}{\\partial t}
   = \\frac{\\partial}{\\partial z}\\!\\left(\\nu_Y \\frac{\\partial Y}{\\partial z}\\right)
   - \\frac{Y - Y_{\\mathrm{obs}}}{\\tau_R}
   + Y\\,L_{\\mathrm{sour}} + Q_{\\mathrm{sour}}

The diffusivity :math:`\\nu_Y` is defined at cell faces.  The diffusion term,
linear source :math:`L_{\\mathrm{sour}}`, and relaxation term are treated
implicitly with Crank–Nicolson implicitness ``cnpar``; the constant source
:math:`Q_{\\mathrm{sour}}` is explicit.  Relaxation is only applied where
``tau_r[i] < 1e10``.

Boundary conditions (``bc_up``, ``bc_down``) are Dirichlet (``Dirichlet = 0``,
prescribes the value) or Neumann (``Neumann = 1``, prescribes the flux).
Fluxes *entering* a boundary cell are positive by convention.  For
non-negative concentrations (``posconc = 1``), negative Neumann boundary fluxes
are linearised following Patankar (1980) to preserve positivity.

The Thomas algorithm (:func:`~pygotm.util.tridiagonal.tridiagonal`) solves the
resulting banded system.

Original author: Lars Umlauf.
"""

import numba
import numpy as np

from pygotm.util.tridiagonal import tridiagonal
from pygotm.util.util import Dirichlet as DIRICHLET
from pygotm.util.util import Neumann as NEUMANN

__all__ = [
    "DIRICHLET",
    "NEUMANN",
    "diff_center",
    "diff_center_batch",
]


@numba.njit(cache=True)
def diff_center(
    nlev: int,
    dt: float,
    cnpar: float,
    posconc: int,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    tau_r: np.ndarray,
    y_obs: np.ndarray,
    y: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    """Solve diffusion for a cell-centred variable using Crank–Nicolson.

    Advances ``y`` at cell centres (indices 1 to nlev) one time step.  Solves
    the one-dimensional diffusion equation including linear and explicit source
    terms, optional relaxation toward observed values, and Dirichlet or Neumann
    boundary conditions.
    """

    for i in range(2, nlev):
        c = 2.0 * dt * nu_y[i] / (h[i] + h[i + 1]) / h[i]
        a = 2.0 * dt * nu_y[i - 1] / (h[i] + h[i - 1]) / h[i]
        linear_source = dt * l_sour[i]

        cu[i] = -cnpar * c
        au[i] = -cnpar * a
        bu[i] = 1.0 + cnpar * (a + c) - linear_source
        du[i] = (1.0 - (1.0 - cnpar) * (a + c)) * y[i]
        du[i] += (1.0 - cnpar) * (a * y[i - 1] + c * y[i + 1])
        du[i] += dt * q_sour[i]

    if bc_up == NEUMANN:
        a = 2.0 * dt * nu_y[nlev - 1] / (h[nlev] + h[nlev - 1]) / h[nlev]
        linear_source = dt * l_sour[nlev]

        au[nlev] = -cnpar * a
        if posconc == 1 and y_up < 0.0:
            # Patankar (1980): move negative flux to the implicit diagonal term.
            # Requires y[nlev] > 0 — same assumption as Fortran diff_center.F90.
            # Division by zero if y goes exactly to zero.
            # Guard: callers must ensure y > 0 when posconc=1 and y_up < 0.
            bu[nlev] = 1.0 - au[nlev] - linear_source - dt * y_up / y[nlev] / h[nlev]
            du[nlev] = y[nlev] + dt * q_sour[nlev]
            du[nlev] += (1.0 - cnpar) * a * (y[nlev - 1] - y[nlev])
        else:
            bu[nlev] = 1.0 - au[nlev] - linear_source
            du[nlev] = y[nlev] + dt * (q_sour[nlev] + y_up / h[nlev])
            du[nlev] += (1.0 - cnpar) * a * (y[nlev - 1] - y[nlev])
    else:
        au[nlev] = 0.0
        bu[nlev] = 1.0
        du[nlev] = y_up

    if bc_down == NEUMANN:
        c = 2.0 * dt * nu_y[1] / (h[1] + h[2]) / h[1]
        linear_source = dt * l_sour[1]

        cu[1] = -cnpar * c
        if posconc == 1 and y_down < 0.0:
            # Patankar (1980): move negative flux to the implicit diagonal term.
            # Requires y[1] > 0 — same assumption as Fortran diff_center.F90.
            # Division by zero if y goes exactly to zero.
            # Guard: callers must ensure y > 0 when posconc=1 and y_down < 0.
            bu[1] = 1.0 - cu[1] - linear_source - dt * y_down / y[1] / h[1]
            du[1] = y[1] + dt * q_sour[1]
            du[1] += (1.0 - cnpar) * c * (y[2] - y[1])
        else:
            bu[1] = 1.0 - cu[1] - linear_source
            du[1] = y[1] + dt * (q_sour[1] + y_down / h[1])
            du[1] += (1.0 - cnpar) * c * (y[2] - y[1])
    else:
        cu[1] = 0.0
        bu[1] = 1.0
        du[1] = y_down

    apply_relaxation = 0
    for i in range(1, nlev + 1):
        if tau_r[i] < 1.0e10:
            apply_relaxation = 1

    if apply_relaxation == 1:
        for i in range(1, nlev + 1):
            bu[i] += dt / tau_r[i]
            du[i] += dt / tau_r[i] * y_obs[i]

    tridiagonal(au, bu, cu, du, ru, qu, y, 1, nlev)


@numba.njit(parallel=True, cache=True)
def diff_center_batch(
    batch_size: int,
    nlev: int,
    dt: float,
    cnpar: float,
    posconc: int,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    tau_r: np.ndarray,
    y_obs: np.ndarray,
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
        diff_center(
            nlev,
            dt,
            cnpar,
            posconc,
            h[b],
            bc_up,
            bc_down,
            y_up,
            y_down,
            nu_y[b],
            l_sour[b],
            q_sour[b],
            tau_r[b],
            y_obs[b],
            y[b],
            au[b],
            bu[b],
            cu[b],
            du[b],
            ru[b],
            qu[b],
        )
