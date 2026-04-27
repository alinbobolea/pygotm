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

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.turbulence.turbulence import Dirichlet as _DIRICHLET
from pygotm.turbulence.turbulence import Neumann as _NEUMANN
from pygotm.turbulence.turbulence import injection as _INJECTION
from pygotm.turbulence.turbulence import logarithmic as _LOGARITHMIC
from pygotm.util.diff_face import diff_face_column

__all__ = [
    "GenericEquationWorkspace",
    "step_genericeq",
]

_CNPAR: float = 1.0


class GenericEquationWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated generic-psi equation."""

    tke: ti.Field
    tkeo: ti.Field
    eps: ti.Field
    L: ti.Field
    h: ti.Field
    NN: ti.Field
    SS: ti.Field
    P: ti.Field
    B: ti.Field
    Px: ti.Field
    PSTK: ti.Field
    num: ti.Field
    u_taus: ti.Field
    u_taub: ti.Field
    z0s: ti.Field
    z0b: ti.Field
    psi: ti.Field
    avh: ti.Field
    l_sour: ti.Field
    q_sour: ti.Field
    au: ti.Field
    bu: ti.Field
    cu: ti.Field
    du: ti.Field
    ru: ti.Field
    qu: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "tkeo", "eps", "L", "h", "NN", "SS"))
        self.allocate_many(("P", "B", "Px", "PSTK", "num"))
        self.allocate_many(("u_taus", "u_taub", "z0s", "z0b"))
        self.allocate_many(("psi", "avh", "l_sour", "q_sour"))
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))


@ti.func
def _fk_craig(u_tau, eta):  # type: ignore[no-untyped-def]
    return eta * u_tau**3


@ti.func
def _psi_bc_value(  # type: ignore[no-untyped-def]
    bc,
    type_,
    zi,
    ki,
    z0,
    u_tau,
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
):
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = (
                ti.pow(cm0, gen_p)
                * ti.pow(kappa, gen_n)
                * ti.pow(ki, gen_m)
                * ti.pow(zi + z0, gen_n)
            )
        else:
            value = (
                -gen_n
                * ti.pow(cm0, gen_p + 1.0)
                * ti.pow(kappa, gen_n + 1.0)
                / sig_psi
                * ti.pow(ki, gen_m + 0.5)
                * ti.pow(zi + z0, gen_n)
            )

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ti.pow(
            -sig_k * f_k / (cmsf * gen_alpha * gen_l),
            2.0 / 3.0,
        ) / ti.pow(z0, gen_alpha)

        if bc == _DIRICHLET:
            value = (
                ti.pow(cm0, gen_p)
                * ti.pow(capital_k, gen_m)
                * ti.pow(gen_l, gen_n)
                * ti.pow(zi + z0, gen_m * gen_alpha + gen_n)
            )
        else:
            value = (
                -(gen_m * gen_alpha + gen_n)
                * cmsf
                * ti.pow(cm0, gen_p)
                / sig_psi
                * ti.pow(capital_k, gen_m + 0.5)
                * ti.pow(gen_l, gen_n + 1.0)
                * ti.pow(zi + z0, (gen_m + 0.5) * gen_alpha + gen_n)
            )

    return value


@ti_kernel
def step_genericeq(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    cpsi1: ti.f64,
    cpsi2: ti.f64,
    cpsi3plus: ti.f64,
    cpsi3minus: ti.f64,
    cpsix: ti.f64,
    cpsi4: ti.f64,
    sig_psi: ti.f64,
    cm0: ti.f64,
    kappa: ti.f64,
    cde: ti.f64,
    galp: ti.f64,
    length_lim: ti.i32,
    eps_min: ti.f64,
    psi_ubc: ti.i32,
    psi_lbc: ti.i32,
    ubc_type: ti.i32,
    lbc_type: ti.i32,
    sig_k: ti.f64,
    cmsf: ti.f64,
    cw: ti.f64,
    gen_m: ti.f64,
    gen_n: ti.f64,
    gen_p: ti.f64,
    gen_alpha: ti.f64,
    gen_l: ti.f64,
    tke: TemplateArg,
    tkeo: TemplateArg,
    eps: TemplateArg,
    L: TemplateArg,
    h: TemplateArg,
    NN: TemplateArg,
    SS: TemplateArg,
    P: TemplateArg,
    B: TemplateArg,
    Px: TemplateArg,
    PSTK: TemplateArg,
    num: TemplateArg,
    u_taus: TemplateArg,
    u_taub: TemplateArg,
    z0s: TemplateArg,
    z0b: TemplateArg,
    psi: TemplateArg,
    avh: TemplateArg,
    l_sour: TemplateArg,
    q_sour: TemplateArg,
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
):
    r"""Advance the dynamic generic-psi equation for one or more columns."""

    exp1 = 3.0 + gen_p / gen_n
    exp2 = 1.5 + gen_m / gen_n
    exp3 = -1.0 / gen_n

    for col in range(n_cols):
        for i in range(nlev + 1):
            psi[col, i] = (
                ti.pow(cm0, gen_p)
                * ti.pow(tkeo[col, i], gen_m)
                * ti.pow(L[col, i], gen_n)
            )
            avh[col, i] = 0.0
            l_sour[col, i] = 0.0
            q_sour[col, i] = 0.0

        for i in range(1, nlev):
            avh[col, i] = num[col, i] / sig_psi

            cpsi3 = cpsi3minus
            if B[col, i] > 0.0:
                cpsi3 = cpsi3plus

            psi_over_tke = psi[col, i] / tkeo[col, i]
            prod = psi_over_tke * (
                cpsi1 * P[col, i] + cpsix * Px[col, i] + cpsi4 * PSTK[col, i]
            )
            buoyan = cpsi3 * psi_over_tke * B[col, i]
            diss = cpsi2 * psi_over_tke * eps[col, i]

            if prod + buoyan > 0.0:
                q_sour[col, i] = prod + buoyan
                l_sour[col, i] = -diss / psi[col, i]
            else:
                q_sour[col, i] = prod
                l_sour[col, i] = -(diss - buoyan) / psi[col, i]

        ki = tke[col, nlev - 1]
        pos_bc = h[col, nlev]
        if psi_ubc == _NEUMANN:
            pos_bc = 0.5 * h[col, nlev]
        diff_psi_up = _psi_bc_value(
            psi_ubc,
            ubc_type,
            pos_bc,
            ki,
            z0s[col, 0],
            u_taus[col, 0],
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

        ki = tke[col, 1]
        pos_bc = h[col, 1]
        if psi_lbc == _NEUMANN:
            pos_bc = 0.5 * h[col, 1]
        diff_psi_down = _psi_bc_value(
            psi_lbc,
            lbc_type,
            pos_bc,
            ki,
            z0b[col, 0],
            u_taub[col, 0],
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

        diff_face_column(
            col,
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

        psi[col, nlev] = _psi_bc_value(
            _DIRICHLET,
            ubc_type,
            z0s[col, 0],
            tke[col, nlev],
            z0s[col, 0],
            u_taus[col, 0],
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
        psi[col, 0] = _psi_bc_value(
            _DIRICHLET,
            lbc_type,
            z0b[col, 0],
            tke[col, 0],
            z0b[col, 0],
            u_taub[col, 0],
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
            eps[col, i] = (
                ti.pow(cm0, exp1)
                * ti.pow(tke[col, i], exp2)
                * ti.pow(psi[col, i], exp3)
            )
            if eps[col, i] < eps_min:
                eps[col, i] = eps_min

        if length_lim != 0:
            for i in range(nlev + 1):
                nn_pos = 0.5 * (NN[col, i] + ti.abs(NN[col, i]))
                epslim = cde / ti.sqrt(2.0) / galp * tke[col, i] * ti.sqrt(nn_pos)
                if eps[col, i] < epslim:
                    eps[col, i] = epslim

        for i in range(nlev + 1):
            L[col, i] = cde * ti.sqrt(tke[col, i] ** 3) / eps[col, i]
