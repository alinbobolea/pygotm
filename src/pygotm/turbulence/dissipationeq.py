# ruff: noqa: E501
"""
Dynamic transport equation for the dissipation rate :math:`\\varepsilon`.

Implements GOTM Section 4.7.27 (dissipationeq.F90) — solves the
:math:`k`–:math:`\\varepsilon` dissipation equation (Eq. 163):

.. math::

   \\dot{\\varepsilon} = \\mathcal{D}_\\varepsilon
   + \\frac{\\varepsilon}{k}\\bigl(
       c_{\\varepsilon 1}\\,P + c_{\\varepsilon 3}\\,G
       + c_{\\varepsilon x}\\,P_x + c_{\\varepsilon 4}\\,P_s
       - c_{\\varepsilon 2}\\,\\varepsilon
     \\bigr) \\comma

with diffusive transport (Eq. 164):

.. math::

   \\mathcal{D}_\\varepsilon = \\frac{\\partial}{\\partial z}
       \\left( \\frac{\\nu_t}{\\sigma_\\varepsilon}
               \\frac{\\partial \\varepsilon}{\\partial z} \\right) \\comma

and Schmidt number :math:`\\sigma_\\varepsilon`.  After the transport step the
turbulent length scale is recovered as:

.. math::

   l = c_{de} \\frac{k^{3/2}}{\\varepsilon} \\point

Model constants (Rodi 1987)
-----------------------------

The default constants for the :math:`k`–:math:`\\varepsilon` model are:

.. list-table::
   :header-rows: 1

   * - Constant
     - Value
   * - :math:`c_\\mu^0`
     - 0.5477
   * - :math:`\\sigma_k`
     - 1.0
   * - :math:`\\sigma_\\varepsilon`
     - 1.3
   * - :math:`c_{\\varepsilon 1}`
     - 1.44
   * - :math:`c_{\\varepsilon 2}`
     - 1.92

Length-scale limiter
--------------------

When ``length_lim`` is active, Galperin et al. (1988) stability criterion:

.. math::

   \\varepsilon \\ge \\frac{c_{de}}{\\sqrt{2}\\,\\gamma}
       k \\sqrt{\\max(N^2, 0)}

is enforced after the transport step, preventing excessive length scales in
stably stratified flows.

Boundary conditions
-------------------

Boundary conditions follow logarithmic-layer theory (Dirichlet):

.. math::

   \\varepsilon = \\frac{c_{de}\\, k^{3/2}}{\\kappa\\,(z + z_0)} \\comma

or Craig–Banner wave-breaking injection.

Author (original Fortran): Lars Umlauf.
"""

import math

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array
from pygotm.turbulence.turbulence import Dirichlet as _DIRICHLET
from pygotm.turbulence.turbulence import Neumann as _NEUMANN
from pygotm.turbulence.turbulence import injection as _INJECTION
from pygotm.turbulence.turbulence import logarithmic as _LOGARITHMIC
from pygotm.util.diff_face import diff_face

__all__ = [
    "DissipationEquationWorkspace",
    "step_dissipationeq",
    "step_dissipationeq_single",
]

_CNPAR: float = 1.0
_F90_SQRT_TWO: float = float(np.float32(math.sqrt(2.0)))


class DissipationEquationWorkspace(ColumnWorkspace):
    """Workspace arrays for the translated epsilon equation."""

    tke: np.ndarray
    tkeo: np.ndarray
    eps: np.ndarray
    L: np.ndarray
    h: np.ndarray
    NN: np.ndarray
    SS: np.ndarray
    P: np.ndarray
    B: np.ndarray
    Px: np.ndarray
    PSTK: np.ndarray
    num: np.ndarray
    avh: np.ndarray
    sig_eff: np.ndarray
    l_sour: np.ndarray
    q_sour: np.ndarray
    u_taus: np.ndarray
    u_taub: np.ndarray
    z0s: np.ndarray
    z0b: np.ndarray
    au: np.ndarray
    bu: np.ndarray
    cu: np.ndarray
    du: np.ndarray
    ru: np.ndarray
    qu: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.tkeo = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.L = make_column_array(nlev, n_cols=n_cols)
        self.h = make_column_array(nlev, n_cols=n_cols)
        self.NN = make_column_array(nlev, n_cols=n_cols)
        self.SS = make_column_array(nlev, n_cols=n_cols)
        self.P = make_column_array(nlev, n_cols=n_cols)
        self.B = make_column_array(nlev, n_cols=n_cols)
        self.Px = make_column_array(nlev, n_cols=n_cols)
        self.PSTK = make_column_array(nlev, n_cols=n_cols)
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.avh = make_column_array(nlev, n_cols=n_cols)
        self.sig_eff = make_column_array(nlev, n_cols=n_cols)
        self.l_sour = make_column_array(nlev, n_cols=n_cols)
        self.q_sour = make_column_array(nlev, n_cols=n_cols)
        self.u_taus = make_column_array(nlev, n_cols=n_cols)
        self.u_taub = make_column_array(nlev, n_cols=n_cols)
        self.z0s = make_column_array(nlev, n_cols=n_cols)
        self.z0b = make_column_array(nlev, n_cols=n_cols)
        self.au = make_column_array(nlev, n_cols=n_cols)
        self.bu = make_column_array(nlev, n_cols=n_cols)
        self.cu = make_column_array(nlev, n_cols=n_cols)
        self.du = make_column_array(nlev, n_cols=n_cols)
        self.ru = make_column_array(nlev, n_cols=n_cols)
        self.qu = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _fk_craig(u_tau: float, eta: float) -> float:
    return eta * u_tau**3


@numba.njit(cache=True)
def _epsilon_bc_value(
    bc: int,
    type_: int,
    zi: float,
    ki: float,
    z0: float,
    u_tau: float,
    cm0: float,
    cde: float,
    kappa: float,
    sig_k: float,
    sig_e: float,
    sig_e0: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
) -> float:
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = cde * ki**1.5 / (kappa * (zi + z0))
        else:
            value = cm0**4 * ki**2 / (sig_e * (zi + z0))

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ((-sig_k * f_k / (cmsf * gen_alpha * gen_l)) ** (2.0 / 3.0)) / (
            z0**gen_alpha
        )

        if bc == _DIRICHLET:
            value = cde * capital_k**1.5 / gen_l * (zi + z0) ** (1.5 * gen_alpha - 1.0)
        else:
            value = (
                -cmsf
                * cde
                / sig_e0
                * capital_k**2
                * (1.5 * gen_alpha - 1.0)
                * (zi + z0) ** (2.0 * gen_alpha - 1.0)
            )

    return value


@numba.njit(cache=True)
def _step_dissipationeq(
    nlev: int,
    dt: float,
    ce1: float,
    ce2: float,
    ce3plus: float,
    ce3minus: float,
    cex: float,
    ce4: float,
    cm0: float,
    cde: float,
    kappa: float,
    galp: float,
    sig_k: float,
    sig_e: float,
    sig_e0: float,
    sig_peps: int,
    length_lim: int,
    eps_min: float,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    eps: np.ndarray,
    L: np.ndarray,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    avh: np.ndarray,
    sig_eff: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic epsilon-equation (single column)."""
    for i in range(nlev + 1):
        avh[i] = 0.0
        sig_eff[i] = sig_e
        l_sour[i] = 0.0
        q_sour[i] = 0.0

    if sig_peps != 0:
        sig_eff[nlev] = sig_e0
        for i in range(1, nlev):
            peps = (P[i] + Px[i] + B[i]) / eps[i]
            if peps > 1.0:
                peps = 1.0
            sig_eff[i] = peps * sig_e + (1.0 - peps) * sig_e0
        sig_eff[0] = sig_e

    for i in range(1, nlev):
        avh[i] = num[i] / sig_eff[i]

        ce3 = ce3minus
        if B[i] > 0.0:
            ce3 = ce3plus

        eps_over_tke = eps[i] / tkeo[i]
        prod = eps_over_tke * (ce1 * P[i] + cex * Px[i] + ce4 * PSTK[i])
        buoyan = ce3 * eps_over_tke * B[i]
        diss = ce2 * eps_over_tke * eps[i]

        if prod + buoyan > 0.0:
            q_sour[i] = prod + buoyan
            l_sour[i] = -diss / eps[i]
        else:
            q_sour[i] = prod
            l_sour[i] = -(diss - buoyan) / eps[i]

    ki = tke[nlev - 1]
    pos_bc = h[nlev]
    if psi_ubc == _NEUMANN:
        pos_bc = 0.5 * h[nlev]
    diff_eps_up = _epsilon_bc_value(
        psi_ubc,
        ubc_type,
        pos_bc,
        ki,
        z0s,
        u_taus,
        cm0,
        cde,
        kappa,
        sig_k,
        sig_e,
        sig_e0,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    ki = tke[1]
    pos_bc = h[1]
    if psi_lbc == _NEUMANN:
        pos_bc = 0.5 * h[1]
    diff_eps_down = _epsilon_bc_value(
        psi_lbc,
        lbc_type,
        pos_bc,
        ki,
        z0b,
        u_taub,
        cm0,
        cde,
        kappa,
        sig_k,
        sig_e,
        sig_e0,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    diff_face(
        nlev,
        dt,
        _CNPAR,
        h,
        psi_ubc,
        psi_lbc,
        diff_eps_up,
        diff_eps_down,
        avh,
        l_sour,
        q_sour,
        eps,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )

    eps[nlev] = _epsilon_bc_value(
        _DIRICHLET,
        ubc_type,
        z0s,
        tke[nlev],
        z0s,
        u_taus,
        cm0,
        cde,
        kappa,
        sig_k,
        sig_e,
        sig_e0,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )
    eps[0] = _epsilon_bc_value(
        _DIRICHLET,
        lbc_type,
        z0b,
        tke[0],
        z0b,
        u_taub,
        cm0,
        cde,
        kappa,
        sig_k,
        sig_e,
        sig_e0,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    for i in range(nlev + 1):
        if eps[i] < eps_min:
            eps[i] = eps_min

    if length_lim != 0:
        for i in range(nlev + 1):
            nn_pos = 0.5 * (NN[i] + abs(NN[i]))
            epslim = cde / _F90_SQRT_TWO / galp * tke[i] * math.sqrt(nn_pos)
            if eps[i] < epslim:
                eps[i] = epslim

    for i in range(nlev + 1):
        L[i] = cde * math.sqrt(tke[i] * tke[i] * tke[i]) / eps[i]


@numba.njit(parallel=True, cache=True)
def step_dissipationeq(
    batch_size: int,
    nlev: int,
    dt: float,
    ce1: float,
    ce2: float,
    ce3plus: float,
    ce3minus: float,
    cex: float,
    ce4: float,
    cm0: float,
    cde: float,
    kappa: float,
    galp: float,
    sig_k: float,
    sig_e: float,
    sig_e0: float,
    sig_peps: int,
    length_lim: int,
    eps_min: float,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    eps: np.ndarray,
    L: np.ndarray,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    avh: np.ndarray,
    sig_eff: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    u_taus: np.ndarray,
    u_taub: np.ndarray,
    z0s: np.ndarray,
    z0b: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic epsilon-equation (batch)."""
    for b in numba.prange(batch_size):
        _step_dissipationeq(
            nlev,
            dt,
            ce1,
            ce2,
            ce3plus,
            ce3minus,
            cex,
            ce4,
            cm0,
            cde,
            kappa,
            galp,
            sig_k,
            sig_e,
            sig_e0,
            sig_peps,
            length_lim,
            eps_min,
            psi_ubc,
            psi_lbc,
            ubc_type,
            lbc_type,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
            tke[b],
            tkeo[b],
            eps[b],
            L[b],
            h[b],
            NN[b],
            SS[b],
            P[b],
            B[b],
            Px[b],
            PSTK[b],
            num[b],
            avh[b],
            sig_eff[b],
            l_sour[b],
            q_sour[b],
            u_taus[b, 0],
            u_taub[b, 0],
            z0s[b, 0],
            z0b[b, 0],
            au[b],
            bu[b],
            cu[b],
            du[b],
            ru[b],
            qu[b],
        )


step_dissipationeq_single = _step_dissipationeq
