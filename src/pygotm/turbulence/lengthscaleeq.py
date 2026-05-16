# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic q2l-equation\label{sec:lengthscaleeq}
!
! !INTERFACE:
!   subroutine lengthscaleeq(nlev,dt,depth,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! Following suggestions of \cite{Rotta51a}, \cite{MellorYamada82}
! proposed an equation for the product $q^2 l$ expressed by
! \begin{equation}
!   \label{MY}
!   \dot{\overline{q^2 l}}
!   = {\cal D}_l + l ( E_1  P + E_3 G + E_x P_x + E_6 P_s - E_2  F \epsilon )
!   \comma
! \end{equation}
! where $\dot{\overline{q^2 l}}$ denotes the material derivative of $q^2 l$.
! The production terms $P$ and $G$ follow from \eq{PandG}.
! $P_s$ is Stokes shear production defined in \eq{computePs}
! and $P_x$ accounts for extra turbulence production.
! $\epsilon$
! can be computed either directly from \eq{epsilonMY}, or from \eq{epsilon}
! with the help \eq{B1}.
!
! The so-called wall function, $F$, appearing in \eq{MY} is defined by
! \begin{equation}
!   \label{F}
!   F = 1 + E_2 \left( \dfrac{l}{\kappa {\cal L}_z} \right)^2
!   \comma
! \end{equation}
! $\kappa$ being the von K{\'a}rm{\'a}n constant and ${\cal L}_z$ some
! measure for the distance from the wall. Different possiblities
! for  ${\cal L}_z$ are implemented in GOTM, which can be activated
! be setting the parameter {\tt MY\_length} in {\tt gotm.yaml} to
! appropriate values. Close to the wall, however, one always has
! ${\cal L}_z= \overline{z}$, where $\overline{z}$ is the distance from
! the wall.
!
! For horizontally homogeneous flows, the transport term ${\cal D}_l$
! appearing in \eq{MY} is expressed by a simple gradient formulation,
! \begin{equation}
!   \label{diffusionMYlength}
!   {\cal D}_l = \frstder{z} \left( q l S_l \partder{q^2 l}{z} \right)
!  \comma
! \end{equation}
! where $S_l$ is a constant of the model. The values for the model
! constants recommended by \cite{MellorYamada82} are displayed in
! \tab{tab:MY_constants}. They can be set in {\tt gotm.yaml}. Note,
! that the parameter $E_3$ in stably stratifed flows is in principle
! a function of the so-called steady state Richardson-number,
! as discussed by \cite{Burchard2001c}, see discussion in the context
! of \eq{Ri_st}.
! \begin{table}[ht]
!   \begin{center}
! \begin{tabular}{ccccccc}
!                           & $B_1$  & $S_q$ & $S_l$ & $E_1$ & $E_2$ & $E_3$    \\[1mm]
!      \hline
!     \cite{MellorYamada82} & $16.6$ & $0.2$ & $0.2$ & $1.8$ & $1.33$ & $1.8$\\
!   \end{tabular}
!   \caption{\label{tab:MY_constants} Constants appearing in \eq{MY}
!     and \eq{epsilonMY}}
!   \end{center}
! \end{table}
!
! At the end of this routine the length-scale can be constrained according to a
! suggestion of \cite{Galperinetal88}. This feature is optional and can be activated
! by setting {\tt length\_lim = .true.} in {\tt gotm.yaml}.
!
! !USES:
!   use turbulence, only: P,B,Px,PSTK
!   use turbulence, only: tke,tkeo,k_min,eps,eps_min,L
!   use turbulence, only: kappa,e1,e2,e3,ex,e6,b1
!   use turbulence, only: MY_length,cm0,cde,galp,length_lim
!   use turbulence, only: q2l_bc, psi_ubc, psi_lbc, ubc_type, lbc_type
!   use turbulence, only: sl_var
!   use util,       only: Dirichlet,Neumann
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
!  local water depth (m)
!   REALTYPE, intent(in)                :: depth
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
!                     (re-write after first version of
!                      H. Burchard and K. Bolding
!EOP
!------------------------------------------------------------------------
!
! !LOCAL VARIABLES:
!   REALTYPE                  :: DiffQ2lup,DiffQ2ldw,pos_bc
!   REALTYPE                  :: prod,buoyan,diss
!   REALTYPE                  :: prod_pos,prod_neg,buoyan_pos,buoyan_neg
!   REALTYPE                  :: ki,epslim,NN_pos
!   REALTYPE                  :: ds,db,Lcrit
!   REALTYPE                  :: cnpar=_ONE_
!   REALTYPE                  :: q2l(0:nlev),q3(0:nlev)
!   REALTYPE                  :: avh(0:nlev)
!   REALTYPE                  :: Lz(0:nlev)
!   REALTYPE                  :: Lsour(0:nlev),Qsour(0:nlev)
!
!   REALTYPE                  :: l_min
!
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
! compute lower bound for length scale
!
!  some quantities in Mellor-Yamada notation
!
!  diagnostic length scale for wall function
!
! prepare the production terms
!
!     compute diffusivity
!
!     compute production terms in q^2 l - equation
!
!     compute positive and negative parts of RHS
!
!  TKE and position for upper BC
!
!  obtain BC for upper boundary of type "ubc_type"
!
!  TKE and position for lower BC
!
!  obtain BC for lower boundary of type "lbc_type"
!
!  do diffusion step
!
!  fill top and bottom value with something nice
!  (only for output)
!
! compute L and epsilon
!
!    apply the length-scale clipping of Galperin et al. (1988)
!
!    check for very small lengh scale
!
!    compute dissipation rate
!
!    substitute minimum value
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
    "LengthScaleEquationWorkspace",
    "step_lengthscaleeq",
]

_CNPAR: float = 1.0
_SQRT2: float = 1.4142135623730951


class LengthScaleEquationWorkspace(ColumnWorkspace):
    """Workspace arrays for the translated q2l length-scale equation."""

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
    sl_var: np.ndarray
    depth: np.ndarray
    u_taus: np.ndarray
    u_taub: np.ndarray
    z0s: np.ndarray
    z0b: np.ndarray
    q2l: np.ndarray
    avh: np.ndarray
    l_sour: np.ndarray
    q_sour: np.ndarray
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
        self.sl_var = make_column_array(nlev, n_cols=n_cols)
        self.depth = make_column_array(nlev, n_cols=n_cols)
        self.u_taus = make_column_array(nlev, n_cols=n_cols)
        self.u_taub = make_column_array(nlev, n_cols=n_cols)
        self.z0s = make_column_array(nlev, n_cols=n_cols)
        self.z0b = make_column_array(nlev, n_cols=n_cols)
        self.q2l = make_column_array(nlev, n_cols=n_cols)
        self.avh = make_column_array(nlev, n_cols=n_cols)
        self.l_sour = make_column_array(nlev, n_cols=n_cols)
        self.q_sour = make_column_array(nlev, n_cols=n_cols)
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
def _q2l_bc_value(
    bc: int,
    type_: int,
    zi: float,
    ki: float,
    z0: float,
    u_tau: float,
    kappa: float,
    sl: float,
    sq: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
) -> float:
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = 2.0 * kappa * ki * (zi + z0)
        else:
            value = -2.0 * _SQRT2 * sl * kappa * kappa * ki**1.5 * (zi + z0)

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ((-f_k / (_SQRT2 * sq * gen_alpha * gen_l)) ** (2.0 / 3.0)) / (
            z0**gen_alpha
        )
        if bc == _DIRICHLET:
            value = 2.0 * capital_k * gen_l * (zi + z0) ** (gen_alpha + 1.0)
        else:
            value = (
                -2.0
                * _SQRT2
                * sl
                * (gen_alpha + 1.0)
                * capital_k**1.5
                * gen_l
                * gen_l
                * (zi + z0) ** (1.5 * gen_alpha + 1.0)
            )

    return value


@numba.njit(cache=True)
def _step_lengthscaleeq(
    nlev: int,
    dt: float,
    k_min: float,
    eps_min: float,
    kappa: float,
    e1: float,
    e2: float,
    e3: float,
    ex: float,
    e6: float,
    b1: float,
    cde: float,
    my_length: int,
    galp: float,
    length_lim: int,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    sl: float,
    sq: float,
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
    sl_var: np.ndarray,
    depth: float,
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    q2l: np.ndarray,
    avh: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic q2l-equation for a single column."""

    l_min = cde * math.sqrt(k_min * k_min * k_min) / eps_min

    for i in range(nlev + 1):
        q2l[i] = 0.0
        avh[i] = 0.0
        l_sour[i] = 0.0
        q_sour[i] = 0.0

    for i in range(1, nlev):
        q2l[i] = 2.0 * tkeo[i] * L[i]

    db = 0.0
    for i in range(1, nlev):
        db += h[i]
        ds = depth - db

        lz = 0.0
        if my_length == 1:
            lz = kappa * (ds + z0s) * (db + z0b) / (ds + z0s + db + z0b)
        if my_length == 2:
            lz = kappa * min(ds + z0s, db + z0b)
        if my_length == 3:
            lz = kappa * (ds + z0s)

        avh[i] = sl_var[i] * math.sqrt(2.0 * tkeo[i]) * L[i]

        prod = L[i] * (e1 * P[i] + ex * Px[i] + e6 * PSTK[i])
        buoyan = e3 * L[i] * B[i]
        q3 = math.sqrt(8.0 * tkeo[i] * tkeo[i] * tkeo[i])
        diss = q3 / b1 * (1.0 + e2 * (L[i] / lz) * (L[i] / lz))

        if prod + buoyan > 0.0:
            q_sour[i] = prod + buoyan
            l_sour[i] = -diss / q2l[i]
        else:
            q_sour[i] = prod
            l_sour[i] = -(diss - buoyan) / q2l[i]

    ki = tke[nlev - 1]
    pos_bc = h[nlev]
    if psi_ubc == _NEUMANN:
        pos_bc = 0.5 * h[nlev]
    diff_q2l_up = _q2l_bc_value(
        psi_ubc,
        ubc_type,
        pos_bc,
        ki,
        z0s,
        u_taus,
        kappa,
        sl,
        sq,
        cw,
        gen_alpha,
        gen_l,
    )

    ki = tke[1]
    pos_bc = h[1]
    if psi_lbc == _NEUMANN:
        pos_bc = 0.5 * h[1]
    diff_q2l_down = _q2l_bc_value(
        psi_lbc,
        lbc_type,
        pos_bc,
        ki,
        z0b,
        u_taub,
        kappa,
        sl,
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
        psi_ubc,
        psi_lbc,
        diff_q2l_up,
        diff_q2l_down,
        avh,
        l_sour,
        q_sour,
        q2l,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )

    q2l[nlev] = _q2l_bc_value(
        _DIRICHLET,
        ubc_type,
        z0s,
        tke[nlev],
        z0s,
        u_taus,
        kappa,
        sl,
        sq,
        cw,
        gen_alpha,
        gen_l,
    )
    q2l[0] = _q2l_bc_value(
        _DIRICHLET,
        lbc_type,
        z0b,
        tke[0],
        z0b,
        u_taub,
        kappa,
        sl,
        sq,
        cw,
        gen_alpha,
        gen_l,
    )

    for i in range(nlev + 1):
        L[i] = q2l[i] / (2.0 * tke[i])

        if NN[i] > 0.0 and length_lim != 0:
            l_crit = math.sqrt(2.0 * galp * galp * tke[i] / NN[i])
            if L[i] > l_crit:
                L[i] = l_crit

        if L[i] < l_min:
            L[i] = l_min

        eps[i] = cde * math.sqrt(tke[i] * tke[i] * tke[i]) / L[i]

        if eps[i] < eps_min:
            eps[i] = eps_min
            L[i] = cde * math.sqrt(tke[i] * tke[i] * tke[i]) / eps_min


@numba.njit(parallel=True, cache=True)
def step_lengthscaleeq(
    batch_size: int,
    nlev: int,
    dt: float,
    k_min: float,
    eps_min: float,
    kappa: float,
    e1: float,
    e2: float,
    e3: float,
    ex: float,
    e6: float,
    b1: float,
    cde: float,
    my_length: int,
    galp: float,
    length_lim: int,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    sl: float,
    sq: float,
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
    sl_var: np.ndarray,
    depth: np.ndarray,
    u_taus: np.ndarray,
    u_taub: np.ndarray,
    z0s: np.ndarray,
    z0b: np.ndarray,
    q2l: np.ndarray,
    avh: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic q2l-equation for one or more columns."""
    for b in numba.prange(batch_size):
        _step_lengthscaleeq(
            nlev,
            dt,
            k_min,
            eps_min,
            kappa,
            e1,
            e2,
            e3,
            ex,
            e6,
            b1,
            cde,
            my_length,
            galp,
            length_lim,
            psi_ubc,
            psi_lbc,
            ubc_type,
            lbc_type,
            sl,
            sq,
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
            sl_var[b],
            depth[b, 0],
            u_taus[b, 0],
            u_taub[b, 0],
            z0s[b, 0],
            z0b[b, 0],
            q2l[b],
            avh[b],
            l_sour[b],
            q_sour[b],
            au[b],
            bu[b],
            cu[b],
            du[b],
            ru[b],
            qu[b],
        )
