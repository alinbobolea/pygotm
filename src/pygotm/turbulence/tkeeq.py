# ruff: noqa: E501
"""
Dynamic transport equation for turbulent kinetic energy :math:`k`.

Implements GOTM Section 4.7.23 (tkeeq.F90) — solves the :math:`k`-equation for
a Boussinesq fluid in horizontally homogeneous flow (Eq. 150):

.. math::

   \\dot{k} = \\mathcal{D}_k + P + G + P_x + P_s - \\varepsilon \\comma

where :math:`P` is shear production, :math:`G` is buoyancy production
(negative in stable stratification), :math:`P_s` is Stokes shear production,
:math:`P_x` is extra production, and :math:`\\varepsilon` is the dissipation
rate.  The diffusive transport is (Eq. 151):

.. math::

   \\mathcal{D}_k = \\frac{\\partial}{\\partial z}\\left(
       \\frac{\\nu_t}{\\sigma_k} \\frac{\\partial k}{\\partial z}\\right) \\comma

with Schmidt number :math:`\\sigma_k`.  The dissipation rate
:math:`\\varepsilon` is supplied either from the dynamic
:mod:`~pygotm.turbulence.dissipationeq` or :mod:`~pygotm.turbulence.omegaeq`,
or from a prescribed length-scale closure (Eq. 153):

.. math::

   \\varepsilon = (c_\\mu^0)^3 \\frac{k^{3/2}}{l} \\point

Boundary conditions
-------------------

At solid boundaries (surface and bottom), boundary conditions follow
logarithmic-layer theory (Dirichlet):

.. math::

   k = \\frac{u_\\tau^2}{(c_\\mu^0)^{1/2}} \\comma

or wave-breaking injection following Craig and Banner (1994):

.. math::

   F_k = \\eta u_{\\tau s}^3 \\comma

where :math:`\\eta \\approx 100` is the Craig–Banner coefficient.  The
injection boundary condition results in a non-trivial k-profile near the
surface (Dirichlet or Neumann depending on ``k_ubc``/``k_lbc``).

Authors (original Fortran): Lars Umlauf (rewrite), Hans Burchard, Karsten Bolding.
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array
from pygotm.turbulence.turbulence import (
    Dirichlet as _DIRICHLET,
)
from pygotm.turbulence.turbulence import (
    Neumann as _NEUMANN,
)
from pygotm.turbulence.turbulence import (
    injection as _INJECTION,
)
from pygotm.turbulence.turbulence import (
    logarithmic as _LOGARITHMIC,
)
from pygotm.util.diff_face import diff_face

__all__ = [
    "TKEEquationWorkspace",
    "step_tkeeq",
    "step_tkeeq_single",
]

_CNPAR: float = 1.0


class TKEEquationWorkspace(ColumnWorkspace):
    """Workspace arrays for the translated TKE equation."""

    tke: np.ndarray
    tkeo: np.ndarray
    h: np.ndarray
    P: np.ndarray
    B: np.ndarray
    Px: np.ndarray
    PSTK: np.ndarray
    num: np.ndarray
    eps: np.ndarray
    avh: np.ndarray
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
        self.h = make_column_array(nlev, n_cols=n_cols)
        self.P = make_column_array(nlev, n_cols=n_cols)
        self.B = make_column_array(nlev, n_cols=n_cols)
        self.Px = make_column_array(nlev, n_cols=n_cols)
        self.PSTK = make_column_array(nlev, n_cols=n_cols)
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.avh = make_column_array(nlev, n_cols=n_cols)
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
def _k_bc_value(
    bc: int,
    type_: int,
    zi: float,
    z0: float,
    u_tau: float,
    cm0: float,
    sig_k: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
) -> float:
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = u_tau**2 / cm0**2
        else:
            value = 0.0

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ((-sig_k * f_k / (cmsf * gen_alpha * gen_l)) ** (2.0 / 3.0)) / (
            z0**gen_alpha
        )

        if bc == _DIRICHLET:
            value = capital_k * (zi + z0) ** gen_alpha
        else:
            value = (
                -cmsf
                / sig_k
                * capital_k**1.5
                * gen_alpha
                * gen_l
                * (zi + z0) ** (1.5 * gen_alpha)
            )

    return value


@numba.njit(cache=True)
def _step_tkeeq(
    nlev: int,
    dt: float,
    sig_k: float,
    k_min: float,
    k_ubc: int,
    k_lbc: int,
    ubc_type: int,
    lbc_type: int,
    cm0: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    h: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    eps: np.ndarray,
    avh: np.ndarray,
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
    r"""Advance the dynamic k-equation for a single column."""

    for i in range(nlev + 1):
        tkeo[i] = tke[i]
        avh[i] = 0.0
        l_sour[i] = 0.0
        q_sour[i] = 0.0

    for i in range(1, nlev):
        avh[i] = num[i] / sig_k

        prod = P[i] + Px[i] + PSTK[i]
        buoyan = B[i]
        diss = eps[i]

        if prod + buoyan > 0.0:
            q_sour[i] = prod + buoyan
            l_sour[i] = -diss / tke[i]
        else:
            q_sour[i] = prod
            l_sour[i] = -(diss - buoyan) / tke[i]

    pos_bc = h[nlev]
    if k_ubc == _NEUMANN:
        pos_bc = 0.5 * h[nlev]
    diff_k_up = _k_bc_value(
        k_ubc,
        ubc_type,
        pos_bc,
        z0s,
        u_taus,
        cm0,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    pos_bc = h[1]
    if k_lbc == _NEUMANN:
        pos_bc = 0.5 * h[1]
    diff_k_down = _k_bc_value(
        k_lbc,
        lbc_type,
        pos_bc,
        z0b,
        u_taub,
        cm0,
        sig_k,
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
        k_ubc,
        k_lbc,
        diff_k_up,
        diff_k_down,
        avh,
        l_sour,
        q_sour,
        tke,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )

    tke[nlev] = _k_bc_value(
        _DIRICHLET,
        ubc_type,
        z0s,
        z0s,
        u_taus,
        cm0,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )
    tke[0] = _k_bc_value(
        _DIRICHLET,
        lbc_type,
        z0b,
        z0b,
        u_taub,
        cm0,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    for i in range(nlev + 1):
        if tke[i] < k_min:
            tke[i] = k_min


@numba.njit(parallel=True, cache=True)
def step_tkeeq(
    batch_size: int,
    nlev: int,
    dt: float,
    sig_k: float,
    k_min: float,
    k_ubc: int,
    k_lbc: int,
    ubc_type: int,
    lbc_type: int,
    cm0: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    h: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    eps: np.ndarray,
    avh: np.ndarray,
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
    r"""Advance the dynamic k-equation for one or more columns."""

    for col in numba.prange(batch_size):
        _step_tkeeq(
            nlev,
            dt,
            sig_k,
            k_min,
            k_ubc,
            k_lbc,
            ubc_type,
            lbc_type,
            cm0,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
            tke[col],
            tkeo[col],
            h[col],
            P[col],
            B[col],
            Px[col],
            PSTK[col],
            num[col],
            eps[col],
            avh[col],
            l_sour[col],
            q_sour[col],
            u_taus[col, 0],
            u_taub[col, 0],
            z0s[col, 0],
            z0b[col, 0],
            au[col],
            bu[col],
            cu[col],
            du[col],
            ru[col],
            qu[col],
        )


step_tkeeq_single = _step_tkeeq
