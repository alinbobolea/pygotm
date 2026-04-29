# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic q2/2-equation \label{sec:q2over2eq}
!
! !INTERFACE:
!   subroutine q2over2eq(nlev,dt,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! The transport equation for the TKE $q^2/2=k$ can be written as
! \begin{equation}
!   \label{tkeB}
!   \dot{\overline{q^2/2}}
!   =
!   {\cal D}_q +  P + G + P_x + P_s - \epsilon
!   \comma
! \end{equation}
! where $\dot{\overline{q^2/2}}$ denotes the material derivative of $q^2/2$.
! With $P$ and $G$ following from \eq{PandG}, evidently, this equation is
! formally identical to \eq{tkeA}. The only reason why it is discretized
! seperately here, is the slightly different down-gradient model for the
! transport term,
! \begin{equation}
!   \label{diffusionMYTKE}
!   {\cal D}_q = \frstder{z} \left( q l S_q \partder{q^2/2}{z} \right)
!  \comma
! \end{equation}
! where $S_q$ is a model constant. The notation has been chosen according
! to that introduced by \cite{MellorYamada82}. Using their notation,
! also \eq{epsilon} can be expressed in mathematically identical form
! as
! \begin{equation}
!   \label{epsilonMY}
!   \epsilon = \frac{q^3}{B_1 l}
!   \comma
! \end{equation}
! where $B_1$ is a constant of the model. Note, that the equivalence of
! \eq{epsilon} and \eq{epsilonMY} requires that
! \begin{equation}
!   \label{B1}
!   (c_\mu^0)^{-2} = \frac{1}{2} B_1^\frac{2}{3}
!   \point
! \end{equation}
!
! !USES:
!   use turbulence,   only: P,B,Px,PSTK
!   use turbulence,   only: tke,tkeo,k_min,eps,L
!   use turbulence,   only: q2over2_bc, k_ubc, k_lbc, ubc_type, lbc_type
!   use turbulence,   only: sq_var
!   use util,         only: Dirichlet,Neumann
!
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
!  number of vertical layers
!   integer,  intent(in)                :: nlev
!
!  time step (s)
!   REALTYPE, intent(in)                :: dt
!
!  surface and bottom
!  friction velocity (m/s)
!   REALTYPE, intent(in)                :: u_taus,u_taub
!
!  surface and bottom
!  roughness length (m)
!   REALTYPE, intent(in)                :: z0s,z0b
!
!  layer thickness (m)
!   REALTYPE, intent(in)                :: h(0:nlev)
!
!  square of shear and buoyancy
!  frequency (1/s^2)
!   REALTYPE, intent(in)                :: NN(0:nlev),SS(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!------------------------------------------------------------------------
!
! !LOCAL VARIABLES:
!   REALTYPE                  :: DiffKup,DiffKdw,pos_bc
!   REALTYPE                  :: prod,buoyan,diss
!   REALTYPE                  :: prod_pos,prod_neg,buoyan_pos,buoyan_neg
!   REALTYPE                  :: cnpar=_ONE_
!   REALTYPE                  :: avh(0:nlev)
!   REALTYPE                  :: Lsour(0:nlev),Qsour(0:nlev)
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
!  position for upper BC
!
!  obtain BC for upper boundary of type "ubc_type"
!
!  position for lower BC
!
!  obtain BC for lower boundary of type "lbc_type"
!
!  do diffusion step
!
!  fill top and bottom value with something nice
!  (only for output)
!
!  clip at k_min
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
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
    "Q2Over2EquationWorkspace",
    "step_q2over2eq",
]

_CNPAR: float = 1.0
_SQRT2: float = 1.4142135623730951


class Q2Over2EquationWorkspace(ColumnWorkspace):
    """Workspace arrays for the translated Mellor-Yamada q2/2 equation."""

    tke: np.ndarray
    tkeo: np.ndarray
    h: np.ndarray
    P: np.ndarray
    B: np.ndarray
    Px: np.ndarray
    PSTK: np.ndarray
    eps: np.ndarray
    L: np.ndarray
    sq_var: np.ndarray
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
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.L = make_column_array(nlev, n_cols=n_cols)
        self.sq_var = make_column_array(nlev, n_cols=n_cols)
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
def _q2over2_bc_value(
    bc: int,
    type_: int,
    zi: float,
    z0: float,
    u_tau: float,
    b1: float,
    sq: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
) -> float:
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = u_tau**2 * b1 ** (2.0 / 3.0) / 2.0
        else:
            value = 0.0

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ((-f_k / (_SQRT2 * sq * gen_alpha * gen_l)) ** (2.0 / 3.0)) / (
            z0**gen_alpha
        )

        if bc == _DIRICHLET:
            value = capital_k * (zi + z0) ** gen_alpha
        else:
            value = (
                -_SQRT2
                * sq
                * capital_k**1.5
                * gen_alpha
                * gen_l
                * (zi + z0) ** (1.5 * gen_alpha)
            )

    return value


@numba.njit(cache=True)
def _step_q2over2eq(
    nlev: int,
    dt: float,
    k_min: float,
    b1: float,
    k_ubc: int,
    k_lbc: int,
    ubc_type: int,
    lbc_type: int,
    sq: float,
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
    eps: np.ndarray,
    L: np.ndarray,
    sq_var: np.ndarray,
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
    r"""Advance the dynamic q2/2-equation for a single column."""

    for i in range(nlev + 1):
        tkeo[i] = tke[i]
        avh[i] = 0.0
        l_sour[i] = 0.0
        q_sour[i] = 0.0

    for i in range(1, nlev):
        avh[i] = sq_var[i] * math.sqrt(2.0 * tke[i]) * L[i]

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
    diff_k_up = _q2over2_bc_value(
        k_ubc,
        ubc_type,
        pos_bc,
        z0s,
        u_taus,
        b1,
        sq,
        cw,
        gen_alpha,
        gen_l,
    )

    pos_bc = h[1]
    if k_lbc == _NEUMANN:
        pos_bc = 0.5 * h[1]
    diff_k_down = _q2over2_bc_value(
        k_lbc,
        lbc_type,
        pos_bc,
        z0b,
        u_taub,
        b1,
        sq,
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

    tke[nlev] = _q2over2_bc_value(
        _DIRICHLET,
        ubc_type,
        z0s,
        z0s,
        u_taus,
        b1,
        sq,
        cw,
        gen_alpha,
        gen_l,
    )
    tke[0] = _q2over2_bc_value(
        _DIRICHLET,
        lbc_type,
        z0b,
        z0b,
        u_taub,
        b1,
        sq,
        cw,
        gen_alpha,
        gen_l,
    )

    for i in range(nlev + 1):
        if tke[i] < k_min:
            tke[i] = k_min


@numba.njit(parallel=True, cache=True)
def step_q2over2eq(
    batch_size: int,
    nlev: int,
    dt: float,
    k_min: float,
    b1: float,
    k_ubc: int,
    k_lbc: int,
    ubc_type: int,
    lbc_type: int,
    sq: float,
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
    eps: np.ndarray,
    L: np.ndarray,
    sq_var: np.ndarray,
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
    r"""Advance the dynamic q2/2-equation for one or more columns."""
    for b in numba.prange(batch_size):
        _step_q2over2eq(
            nlev,
            dt,
            k_min,
            b1,
            k_ubc,
            k_lbc,
            ubc_type,
            lbc_type,
            sq,
            cw,
            gen_alpha,
            gen_l,
            tke[b],
            tkeo[b],
            h[b],
            P[b],
            B[b],
            Px[b],
            PSTK[b],
            eps[b],
            L[b],
            sq_var[b],
            avh[b],
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
