"""
Vertical advection for cell-centred tracers — translation of ``adv_center.F90``.

Solves the one-dimensional advection equation for variables defined at cell
centres.  Two formulations are supported:

* **Conservative** (``CONSERVATIVE = 1``):
  :math:`\\partial_t Y = -\\partial_z(wY)`.
  Used for settling or rising tracers (e.g. sediment, phytoplankton).
* **Non-conservative** (``NON_CONSERVATIVE = 0``):
  :math:`\\partial_t Y = -w\\partial_z Y`.
  Used when the water column has a prescribed net vertical velocity.

The advective face flux is reconstructed with one of seven slope-limiter
schemes selected by the ``method`` integer:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Constant
     - Value
     - Scheme
   * - ``UPSTREAM``
     - 1
     - First-order upwind
   * - ``P1``
     - 2
     - First-order upwind (P1 — same as UPSTREAM in this implementation)
   * - ``P2``
     - 3
     - Second-order unbounded (may produce over/undershoots)
   * - ``Superbee``
     - 4
     - Superbee limiter (Roe 1986)
   * - ``MUSCL``
     - 5
     - MUSCL (van Leer 1979)
   * - ``P2_PDM``
     - 6
     - P2 with Positive Definite Method limiter (Pietrzak 1998)
   * - ``SPLMAX13``
     - 13
     - SPLMAX13 (Pietrzak 1998)

Sub-stepping is applied when :math:`\\max(|w|\\Delta t / \\Delta z) > 1`, up to
``_ITMAX = 100`` sub-steps per timestep.

Original author: Lars Umlauf.
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace
from pygotm.util.util import (
    CENTRAL,
    MUSCL,
    P1,
    P2,
    P2_PDM,
    SPLMAX13,
    UPSTREAM,
)
from pygotm.util.util import (
    Superbee as SUPERBEE,
)
from pygotm.util.util import (
    flux as FLUX,
)
from pygotm.util.util import (
    oneSided as ONE_SIDED,
)
from pygotm.util.util import (
    value as VALUE,
)
from pygotm.util.util import (
    zeroDivergence as ZERO_DIVERGENCE,
)

__all__ = [
    "AdvectionBatchWorkspace",
    "AdvectionWorkspace",
    "CENTRAL",
    "CONSERVATIVE",
    "FLUX",
    "MUSCL",
    "NON_CONSERVATIVE",
    "ONE_SIDED",
    "P1",
    "P2",
    "P2_PDM",
    "SPLMAX13",
    "SUPERBEE",
    "UPSTREAM",
    "VALUE",
    "ZERO_DIVERGENCE",
    "adv_center",
    "adv_center_batch",
    "clean_adv_center",
    "init_adv_center",
]


NON_CONSERVATIVE = 0
CONSERVATIVE = 1

_HALF = 0.5
_ONE_THIRD = 1.0 / 3.0
_ONE_SIXTH = 1.0 / 6.0
_ITMAX = 100


class AdvectionWorkspace(ColumnWorkspace):
    """Single-column advection workspace — cu has shape (nlev+1,)."""

    def __init__(self, nlev: int) -> None:
        super().__init__(nlev)
        self.cu = np.zeros(nlev + 1, dtype=np.float64)


class AdvectionBatchWorkspace(ColumnWorkspace):
    """Batch advection workspace — cu has shape (batch_size, nlev+1)."""

    def __init__(self, nlev: int, batch_size: int) -> None:
        super().__init__(nlev, n_cols=batch_size)
        self.cu = np.zeros((batch_size, nlev + 1), dtype=np.float64)


def init_adv_center(nlev: int) -> AdvectionWorkspace:
    """Allocate the temporary face-flux array used by `adv_center`."""
    return AdvectionWorkspace(nlev)


def clean_adv_center(workspace: AdvectionWorkspace) -> None:
    """No-op — NumPy workspaces are garbage-collected normally."""


@numba.njit(cache=True)
def _adv_reconstruct(
    scheme: int,
    cfl: float,
    fuu: float,
    fu: float,
    fd: float,
) -> float:
    """Reconstruct the upstream-biased interface value with the GOTM limiter."""

    deltaf = fd - fu
    deltafu = fu - fuu
    result = fu
    limiter = 0.0
    x = 0.0

    if deltaf * deltafu > 0.0:
        ratio = deltafu / deltaf

        if scheme == SUPERBEE:
            limiter = max(
                min(2.0 * ratio, 1.0),
                min(ratio, 2.0),
            )
        elif scheme == P2_PDM:
            x = _ONE_SIXTH * (1.0 - 2.0 * cfl)
            limiter = (0.5 + x) + (0.5 - x) * ratio
            limiter = min(
                2.0 * ratio / (cfl + 1.0e-10),
                min(limiter, 2.0 / (1.0 - cfl)),
            )
        elif scheme == SPLMAX13:
            limiter = min(
                2.0 * ratio,
                min(
                    _ONE_THIRD * max(1.0 + 2.0 * ratio, 2.0 + ratio),
                    2.0,
                ),
            )
        elif scheme == MUSCL:
            limiter = min(
                2.0 * ratio,
                min(_HALF * (1.0 + ratio), 2.0),
            )
        elif scheme == P2:
            x = _ONE_SIXTH * (1.0 - 2.0 * cfl)
            limiter = (0.5 + x) + (0.5 - x) * ratio
        elif scheme == CENTRAL:
            limiter = 1.0 / (1.0 - cfl)
        else:
            # UPSTREAM and P1 (not yet implemented in GOTM) use first-order
            # upstream.
            limiter = 0.0

        result = fu + _HALF * limiter * (1.0 - cfl) * deltaf
    else:
        if scheme == P2:
            x = _ONE_SIXTH * (1.0 - 2.0 * cfl)
            result = fu + _HALF * (1.0 - cfl) * (
                (0.5 + x) * deltaf + (0.5 - x) * deltafu
            )
        elif scheme == CENTRAL:
            result = _HALF * (fu + fd)
        else:
            result = fu  # UPSTREAM and P1 fall back to upstream value

    return result


@numba.njit(cache=True)
def adv_center(
    nlev: int,
    dt: float,
    h: np.ndarray,
    ho: np.ndarray,
    ww: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    method: int,
    mode: int,
    y: np.ndarray,
    cu: np.ndarray,
) -> None:
    """Advance a single-column tracer on cell centers using GOTM advection."""

    cmax = 0.0
    for level in range(nlev + 1):
        cu[level] = 0.0

    for k in range(1, nlev):
        courant = abs(ww[k]) * dt / (0.5 * (h[k] + h[k + 1]))
        if courant > cmax:
            cmax = courant

    iterations = min(_ITMAX, int(cmax) + 1)
    iterations_f = float(iterations)

    for _ in range(iterations):
        for k in range(1, nlev):
            courant = 0.0
            y_upstream = 0.0
            y_central = 0.0
            y_downstream = 0.0
            if ww[k] > 0.0:
                courant = ww[k] / iterations_f * dt / (0.5 * (h[k] + h[k + 1]))
                if k > 1:
                    y_upstream = y[k - 1]
                else:
                    y_upstream = y[k]
                y_central = y[k]
                y_downstream = y[k + 1]
            else:
                courant = -ww[k] / iterations_f * dt / (0.5 * (h[k] + h[k + 1]))
                if k < nlev - 1:
                    y_upstream = y[k + 2]
                else:
                    y_upstream = y[k + 1]
                y_central = y[k + 1]
                y_downstream = y[k]

            reconstructed = _adv_reconstruct(
                method,
                courant,
                y_upstream,
                y_central,
                y_downstream,
            )
            cu[k] = ww[k] * reconstructed

        if bc_up == FLUX:
            cu[nlev] = -y_up
        elif bc_up == VALUE:
            cu[nlev] = ww[nlev] * y_up
        elif bc_up == ONE_SIDED:
            if ww[nlev] >= 0.0:
                cu[nlev] = ww[nlev] * y[nlev]
            else:
                cu[nlev] = 0.0
        else:
            cu[nlev] = cu[nlev - 1]

        if bc_down == FLUX:
            cu[0] = y_down
        elif bc_down == VALUE:
            cu[0] = ww[0] * y_down
        elif bc_down == ONE_SIDED:
            if ww[0] <= 0.0:
                cu[0] = ww[0] * y[1]
            else:
                cu[0] = 0.0
        else:
            cu[0] = cu[1]

        if mode == NON_CONSERVATIVE:
            for k in range(1, nlev + 1):
                y[k] = y[k] - dt / iterations_f * (
                    (cu[k] - cu[k - 1]) / h[k] - y[k] * (ww[k] - ww[k - 1]) / h[k]
                )
        else:
            for k in range(1, nlev + 1):
                y[k] = y[k] - dt / iterations_f * ((cu[k] - cu[k - 1]) / h[k])


@numba.njit(parallel=True, cache=True)
def adv_center_batch(
    batch_size: int,
    nlev: int,
    dt: float,
    h: np.ndarray,
    ho: np.ndarray,
    ww: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    method: int,
    mode: int,
    y: np.ndarray,
    cu: np.ndarray,
) -> None:
    """Batch variant: process batch_size columns in parallel with numba.prange."""
    for b in numba.prange(batch_size):
        adv_center(
            nlev,
            dt,
            h[b],
            ho[b],
            ww[b],
            bc_up,
            bc_down,
            y_up,
            y_down,
            method,
            mode,
            y[b],
            cu[b],
        )
