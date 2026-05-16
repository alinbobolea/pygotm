# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic psi-equation  \label{sec:genericeq}
!
! !INTERFACE:
!   subroutine genericeq(nlev,dt,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! This model has been formulated by \cite{UmlaufBurchard2003},
! who introduced a `generic' variable,
! \begin{equation}
!   \label{psi_l}
!   \psi = (c_\mu^0)^p k^m l^n
!   \comma
! \end{equation}
! where $k$ is the turbulent kinetic energy computed from \eq{tkeA} and
! $l$ is the dissipative length-scale defined in \eq{epsilon}.
! For appropriate choices of the exponents $p$, $m$, and $n$, the variable
! $\psi$ can be directly identified with the classic length-scale determining
! variables like the rate of dissipation, $\epsilon$, or the product
! $kl$ used by \cite{MellorYamada82} (see \sect{sec:lengthscaleeq}
! and \sect{sec:dissipationeq}).
!  Some examples are compiled in \tab{tab:psi}.
!
! The transport equation for $\psi$ can written as
! \begin{equation}
!   \label{generic}
!   \dot{\psi} = {\cal D}_\psi
!   + \frac{\psi}{k} (  c_{\psi_1} P + c_{\psi_3} G
!                     + c_{\psi x} P_x
!                     + c_{\psi 4} P_s
!    - c_{\psi 2} \epsilon )
!   \comma
! \end{equation}
! where $\dot{\psi}$ denotes the material derivative of $\psi$,
! see \cite{UmlaufBurchard2003}.
! The production terms $P$ and $G$ follow from \eq{PandG}.
! $P_s$ is Stokes shear production defined in \eq{computePs}
! and $P_x$ accounts for extra turbulence production.
! ${\cal D}_\psi$ represents the sum of the viscous and turbulent
! transport terms. The rate of dissipation can computed by solving
! \eq{psi_l} for $l$ and inserting the result into \eq{epsilon}.
!
! For horizontally homogeneous flows, the transport terms ${\cal D}_\psi$
! appearing in \eq{generic} are expressed by a simple
! gradient formulation,
! \begin{equation}
!   \label{diffusionGeneric}
!   {\cal D}_\psi = \frstder{z}
!   \left( \dfrac{\nu_t}{\sigma_\psi} \partder{\psi}{z} \right)
!  \point
! \end{equation}
!
! For appropriate choices of the parameters, most of the classic transport
! equations can be directly recovered from the generic equation \eq{generic}.
! An example is the transport equation for the inverse turbulent time scale,
! $\omega \propto \epsilon / k$, which has been formulated by \cite{Wilcox88}
! and extended to buoyancy affected flows by \cite{Umlaufetal2003}. The precise
! definition of $\omega$ follows from \tab{tab:psi}, and its transport
! equation can be written as
! \begin{equation}
!   \label{KW}
!   \dot{\omega}
!   =
!   {\cal D}_\omega
!   + \frac{\omega}{k} (  c_{\omega_1} P + c_{\omega_3} G
!   - c_{\omega 2} \epsilon )
!   \comma
! \end{equation}
! which is clearly a special case of \eq{generic}. Model constants for this
! and other traditional models are given in \tab{tab:constants}.
!
! Apart from having to code only one equation to recover all of the
! traditional models, the main advantage of the generic equation is its
! flexibility. After choosing meaningful values for physically relevant
! parameters like  the von K{\'a}rm{\'a}n constant, $\kappa$, the temporal
! decay rate for homogeneous turbulence, $d$, some parameters related to
! breaking surface waves, etc, a two-equation model can be generated,
! which has exactly the required properties. This is discussed in
! great detail in  \cite{UmlaufBurchard2003}. All algorithms have been
! implemented in GOTM and are described in \sect{sec:generate}.
!
! !USES:
!   use turbulence, only: P,B,Px,PSTK,num
!   use turbulence, only: tke,tkeo,k_min,eps,eps_min,L
!   use turbulence, only: cpsi1,cpsi2,cpsi3plus,cpsi3minus,cpsix,cpsi4,sig_psi
!   use turbulence, only: gen_m,gen_n,gen_p
!   use turbulence, only: cm0,cde,galp,length_lim
!   use turbulence, only: psi_bc, psi_ubc, psi_lbc, ubc_type, lbc_type
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
!  Original author(s): Lars Umlauf and Hans Burchard
!
!EOP
!------------------------------------------------------------------------
!
! !LOCAL VARIABLES:
!   REALTYPE                  :: DiffPsiup,DiffPsidw,pos_bc
!   REALTYPE                  :: prod,buoyan,diss
!   REALTYPE                  :: prod_pos,prod_neg,buoyan_pos,buoyan_neg
!   REALTYPE                  :: ki,epslim,PsiOverTke,NN_pos
!   REALTYPE                  :: cnpar=_ONE_
!   REALTYPE                  :: exp1,exp2,exp3
!   REALTYPE                  :: psi(0:nlev)
!   REALTYPE                  :: avh(0:nlev)
!   REALTYPE                  :: Lsour(0:nlev),Qsour(0:nlev)
!   REALTYPE                  :: cpsi3
!
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
!  compute some parameters
!
!  re-construct psi at "old" timestep
!
!  compute RHS
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
!     recover dissipation rate from k and psi
!
!     clip at eps_min
!
!  limit dissipation rate under stable stratification,
!  see Galperin et al. (1988)
!
!        look for N^2 > 0
!
!        compute limit
!
!        clip at limit
!
!     compute dissipative scale
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
    "GenericEquationWorkspace",
    "step_genericeq",
]

_CNPAR: float = 1.0


class GenericEquationWorkspace(ColumnWorkspace):
    """Workspace arrays for the translated generic-psi equation."""

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
    u_taus: np.ndarray
    u_taub: np.ndarray
    z0s: np.ndarray
    z0b: np.ndarray
    psi: np.ndarray
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
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.u_taus = make_column_array(nlev, n_cols=n_cols)
        self.u_taub = make_column_array(nlev, n_cols=n_cols)
        self.z0s = make_column_array(nlev, n_cols=n_cols)
        self.z0b = make_column_array(nlev, n_cols=n_cols)
        self.psi = make_column_array(nlev, n_cols=n_cols)
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
def _psi_bc_value(
    bc: int,
    type_: int,
    zi: float,
    ki: float,
    z0: float,
    u_tau: float,
    cm0: float,
    kappa: float,
    sig_k: float,
    cmsf: float,
    cw: float,
    gen_m: float,
    gen_n: float,
    gen_p: float,
    sig_psi: float,
    gen_alpha: float,
    gen_l: float,
) -> float:
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = cm0**gen_p * kappa**gen_n * ki**gen_m * (zi + z0) ** gen_n
        else:
            value = (
                -gen_n
                * cm0 ** (gen_p + 1.0)
                * kappa ** (gen_n + 1.0)
                / sig_psi
                * ki ** (gen_m + 0.5)
                * (zi + z0) ** gen_n
            )

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ((-sig_k * f_k / (cmsf * gen_alpha * gen_l)) ** (2.0 / 3.0)) / (
            z0**gen_alpha
        )

        if bc == _DIRICHLET:
            value = (
                cm0**gen_p
                * capital_k**gen_m
                * gen_l**gen_n
                * (zi + z0) ** (gen_m * gen_alpha + gen_n)
            )
        else:
            value = (
                -(gen_m * gen_alpha + gen_n)
                * cmsf
                * cm0**gen_p
                / sig_psi
                * capital_k ** (gen_m + 0.5)
                * gen_l ** (gen_n + 1.0)
                * (zi + z0) ** ((gen_m + 0.5) * gen_alpha + gen_n)
            )

    return value


@numba.njit(cache=True)
def _step_genericeq(
    nlev: int,
    dt: float,
    cpsi1: float,
    cpsi2: float,
    cpsi3plus: float,
    cpsi3minus: float,
    cpsix: float,
    cpsi4: float,
    sig_psi: float,
    cm0: float,
    kappa: float,
    cde: float,
    galp: float,
    length_lim: int,
    eps_min: float,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    sig_k: float,
    cmsf: float,
    cw: float,
    gen_m: float,
    gen_n: float,
    gen_p: float,
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
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    psi: np.ndarray,
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
    r"""Advance the dynamic generic-psi equation for a single column."""
    exp1 = 3.0 + gen_p / gen_n
    exp2 = 1.5 + gen_m / gen_n
    exp3 = -1.0 / gen_n

    for i in range(nlev + 1):
        psi[i] = cm0**gen_p * tkeo[i] ** gen_m * L[i] ** gen_n
        avh[i] = 0.0
        l_sour[i] = 0.0
        q_sour[i] = 0.0

    for i in range(1, nlev):
        avh[i] = num[i] / sig_psi

        cpsi3 = cpsi3minus
        if B[i] > 0.0:
            cpsi3 = cpsi3plus

        psi_over_tke = psi[i] / tkeo[i]
        prod = psi_over_tke * (cpsi1 * P[i] + cpsix * Px[i] + cpsi4 * PSTK[i])
        buoyan = cpsi3 * psi_over_tke * B[i]
        diss = cpsi2 * psi_over_tke * eps[i]

        if prod + buoyan > 0.0:
            q_sour[i] = prod + buoyan
            l_sour[i] = -diss / psi[i]
        else:
            q_sour[i] = prod
            l_sour[i] = -(diss - buoyan) / psi[i]

    ki = tke[nlev - 1]
    pos_bc = h[nlev]
    if psi_ubc == _NEUMANN:
        pos_bc = 0.5 * h[nlev]
    diff_psi_up = _psi_bc_value(
        psi_ubc,
        ubc_type,
        pos_bc,
        ki,
        z0s,
        u_taus,
        cm0,
        kappa,
        sig_k,
        cmsf,
        cw,
        gen_m,
        gen_n,
        gen_p,
        sig_psi,
        gen_alpha,
        gen_l,
    )

    ki = tke[1]
    pos_bc = h[1]
    if psi_lbc == _NEUMANN:
        pos_bc = 0.5 * h[1]
    diff_psi_down = _psi_bc_value(
        psi_lbc,
        lbc_type,
        pos_bc,
        ki,
        z0b,
        u_taub,
        cm0,
        kappa,
        sig_k,
        cmsf,
        cw,
        gen_m,
        gen_n,
        gen_p,
        sig_psi,
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
        diff_psi_up,
        diff_psi_down,
        avh,
        l_sour,
        q_sour,
        psi,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )

    psi[nlev] = _psi_bc_value(
        _DIRICHLET,
        ubc_type,
        z0s,
        tke[nlev],
        z0s,
        u_taus,
        cm0,
        kappa,
        sig_k,
        cmsf,
        cw,
        gen_m,
        gen_n,
        gen_p,
        sig_psi,
        gen_alpha,
        gen_l,
    )
    psi[0] = _psi_bc_value(
        _DIRICHLET,
        lbc_type,
        z0b,
        tke[0],
        z0b,
        u_taub,
        cm0,
        kappa,
        sig_k,
        cmsf,
        cw,
        gen_m,
        gen_n,
        gen_p,
        sig_psi,
        gen_alpha,
        gen_l,
    )

    for i in range(nlev + 1):
        eps[i] = cm0**exp1 * tke[i] ** exp2 * psi[i] ** exp3
        if eps[i] < eps_min:
            eps[i] = eps_min

    if length_lim != 0:
        for i in range(nlev + 1):
            nn_pos = 0.5 * (NN[i] + abs(NN[i]))
            epslim = cde / math.sqrt(2.0) / galp * tke[i] * math.sqrt(nn_pos)
            if eps[i] < epslim:
                eps[i] = epslim

    for i in range(nlev + 1):
        L[i] = cde * math.sqrt(tke[i] ** 3) / eps[i]


@numba.njit(parallel=True, cache=True)
def step_genericeq(
    batch_size: int,
    nlev: int,
    dt: float,
    cpsi1: float,
    cpsi2: float,
    cpsi3plus: float,
    cpsi3minus: float,
    cpsix: float,
    cpsi4: float,
    sig_psi: float,
    cm0: float,
    kappa: float,
    cde: float,
    galp: float,
    length_lim: int,
    eps_min: float,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    sig_k: float,
    cmsf: float,
    cw: float,
    gen_m: float,
    gen_n: float,
    gen_p: float,
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
    u_taus: np.ndarray,
    u_taub: np.ndarray,
    z0s: np.ndarray,
    z0b: np.ndarray,
    psi: np.ndarray,
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
    r"""Advance the dynamic generic-psi equation for one or more columns."""
    for b in numba.prange(batch_size):
        _step_genericeq(
            nlev,
            dt,
            cpsi1,
            cpsi2,
            cpsi3plus,
            cpsi3minus,
            cpsix,
            cpsi4,
            sig_psi,
            cm0,
            kappa,
            cde,
            galp,
            length_lim,
            eps_min,
            psi_ubc,
            psi_lbc,
            ubc_type,
            lbc_type,
            sig_k,
            cmsf,
            cw,
            gen_m,
            gen_n,
            gen_p,
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
            u_taus[b, 0],
            u_taub[b, 0],
            z0s[b, 0],
            z0b[b, 0],
            psi[b],
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
