# ruff: noqa: E501
"""
Turbulence shear and buoyancy production.

Implements GOTM Sections 4.3 and 4.7.20 — computes the mechanical shear
production :math:`P`, buoyancy production :math:`G`, buoyancy-variance
production :math:`P_b`, and extra production :math:`P_x`.

Shear production (Eq. 146):

.. math::

   P = \\nu_t (M^2 + \\alpha_w N^2) + P_{\\mathrm{STK}} \\comma

where :math:`\\alpha_w` is the angle between the wave and current vectors
(internal wave model coupling, active only when ``iw_model == 1``), and
:math:`P_{\\mathrm{STK}} = \\nu_{\\mathrm{cl}} \\cdot \\mathrm{SSCSTK}` is the
Stokes-drift shear production cross-term.

Buoyancy production (Eq. 147):

.. math::

   G = -\\nu_t^B \\left(N^2 - \\tilde{\\Gamma}_B\\right) \\comma

where :math:`\\nu_t^B = \\kappa_t` is the turbulent scalar diffusivity and
:math:`\\tilde{\\Gamma}_B` is the counter-gradient buoyancy flux.
This implementation sets :math:`\\tilde{\\Gamma}_B = 0` (standard GOTM behaviour),
so in practice :math:`G = -\\kappa_t N^2`.
:math:`G` is negative (destruction) in stable stratification.

Buoyancy-variance production (Eq. 148):

.. math::

   P_b = -G N^2 = \\nu_t^B N^4 \\point

The total Stokes production (used in the turbulence transport equations) is:

.. math::

   P_{\\mathrm{STK}} = \\nu_t \\cdot \\mathrm{SSCSTK}
                     + \\nu_{\\mathrm{cl}} \\cdot \\mathrm{SSSTK} \\comma

where ``SSCSTK`` and ``SSSTK`` are the Stokes-drift cross-shear and
Stokes-shear-squared terms from :mod:`pygotm.meanflow.shear`.

Author (original Fortran): Lars Umlauf.
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "ProductionWorkspace",
    "step_production",
    "step_production_single",
]


class ProductionWorkspace(ColumnWorkspace):
    """Workspace arrays for production kernels."""

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.NN = make_column_array(nlev, n_cols=n_cols)
        self.SS = make_column_array(nlev, n_cols=n_cols)
        self.xP = make_column_array(nlev, n_cols=n_cols)
        self.SSCSTK = make_column_array(nlev, n_cols=n_cols)
        self.SSSTK = make_column_array(nlev, n_cols=n_cols)
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.nuh = make_column_array(nlev, n_cols=n_cols)
        self.nucl = make_column_array(nlev, n_cols=n_cols)
        self.P = make_column_array(nlev, n_cols=n_cols)
        self.B = make_column_array(nlev, n_cols=n_cols)
        self.Pb = make_column_array(nlev, n_cols=n_cols)
        self.Px = make_column_array(nlev, n_cols=n_cols)
        self.PSTK = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_production(
    nlev: int,
    iw_model: int,
    alpha: float,
    has_xP: int,
    has_sscstk: int,
    has_ssstk: int,
    NN: np.ndarray,
    SS: np.ndarray,
    xP: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nucl: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Pb: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
) -> None:
    r"""Update turbulence production terms (single column)."""
    alpha_eff = 0.0
    if iw_model == 1:
        alpha_eff = alpha

    for i in range(nlev + 1):
        P[i] = num[i] * (SS[i] + alpha_eff * NN[i])
        B[i] = -nuh[i] * NN[i]
        Pb[i] = -B[i] * NN[i]

        if has_xP != 0:
            Px[i] = xP[i]

        if has_sscstk != 0:
            P[i] = P[i] + nucl[i] * SSCSTK[i]
            PSTK[i] = num[i] * SSCSTK[i]

        if has_ssstk != 0:
            if has_sscstk == 0:
                PSTK[i] = 0.0
            PSTK[i] = PSTK[i] + nucl[i] * SSSTK[i]


@numba.njit(parallel=True, cache=True)
def step_production(
    batch_size: int,
    nlev: int,
    iw_model: int,
    alpha: float,
    has_xP: int,
    has_sscstk: int,
    has_ssstk: int,
    NN: np.ndarray,
    SS: np.ndarray,
    xP: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nucl: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Pb: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
) -> None:
    r"""Update turbulence production terms (batch)."""
    for b in numba.prange(batch_size):
        _step_production(
            nlev,
            iw_model,
            alpha,
            has_xP,
            has_sscstk,
            has_ssstk,
            NN[b],
            SS[b],
            xP[b],
            SSCSTK[b],
            SSSTK[b],
            num[b],
            nuh[b],
            nucl[b],
            P[b],
            B[b],
            Pb[b],
            Px[b],
            PSTK[b],
        )


step_production_single = _step_production
