# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic epsilon-equation \label{sec:dissipationeq}
!
! !INTERFACE:
!   subroutine dissipationeq(nlev,dt,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! The $k$-$\epsilon$ model in its form suggested by \cite{Rodi87} has been
! implemented in GOTM.
! In this model, the rate of dissipation is balanced according to
! \begin{equation}
!   \label{dissipation}
!   \dot{\epsilon}
!   =
!   {\cal D}_\epsilon
!   + \frac{\epsilon}{k} ( c_{\epsilon 1} P + c_{\epsilon 3} G
!                        + c_{\epsilon x} P_x
!                        + c_{\epsilon 4} P_s
!                        - c_{\epsilon 2} \epsilon )
!   \comma
! \end{equation}
! where $\dot{\epsilon}$ denotes the material derivative of $\epsilon$.
! The production terms $P$ and $G$ follow from \eq{PandG}.
! $P_s$ is Stokes shear production defined in \eq{computePs}
! and $P_x$ accounts for extra turbulence production.
! ${\cal D}_\epsilon$ represents the sum of the viscous and turbulent
! transport terms.
!
! For horizontally homogeneous flows, the transport term ${\cal D}_\epsilon$
! appearing in \eq{dissipation} is presently expressed by a simple
! gradient formulation,
! \begin{equation}
!   \label{diffusionEps}
!   {\cal D}_\epsilon = \frstder{z}
!    \left( \dfrac{\nu_t}{\sigma_\epsilon} \partder{\epsilon}{z} \right)
!  \comma
! \end{equation}
! where $\sigma_\epsilon$ is the constant Schmidt-number for $\epsilon$.
!
! It should be pointed out that not all authors retain the buoyancy term
! in \eq{dissipation}, see e.g.\ \cite{GibsonLaunder76}.  Similar to the
! model of \cite{MellorYamada82}, \cite{Craftetal96a} set
! $c_{\epsilon 1}=c_{\epsilon 3}$.
! However, in both cases, the $k$-$\epsilon$ model cannot
! predict a proper state of full equilibrium in stratified flows at a
! predefined value of the Richardson number (see
! \cite{Umlaufetal2003} and discussion around \eq{Ri_st}). Model constants are
! summarised in \tab{tab:KE_constants}.
! \begin{table}[ht]
!   \begin{center}
! \begin{tabular}{cccccc}
!     & $c_\mu^0$ & $\sigma_k$  & $\sigma_\epsilon$
!     & $c_{\epsilon 1}$ & $c_{\epsilon 2}$  \\[1mm] \hline
!     \cite{Rodi87} & $0.5577$ & $1.0$ &  $1.3$ & $1.44$ & $1.92$ \\
!   \end{tabular}
!   \caption{\label{tab:KE_constants} Constants appearing in
!    \eq{dissipation} and \eq{epsilon}.}
!   \end{center}
! \end{table}
!
! At the end of this routine the length-scale can be constrained according to a
! suggestion of \cite{Galperinetal88}. This feature is optional and can be activated
! by setting {\tt length\_lim = .true.} in {\tt gotm.yaml}.
!
! !USES:
!   use turbulence, only: P,B,Px,PSTK,num
!   use turbulence, only: tke,tkeo,k_min,eps,eps_min,L
!   use turbulence, only: ce1,ce2,ce3plus,ce3minus,cex,ce4
!   use turbulence, only: cm0,cde,galp,length_lim
!   use turbulence, only: epsilon_bc, psi_ubc, psi_lbc, ubc_type, lbc_type
!   use turbulence, only: sig_e,sig_e0,sig_peps
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
!  Original author(s): Lars Umlauf
!                     (re-write after first version of
!                      H. Burchard and K. Bolding
!EOP
!
! !LOCAL VARIABLES:
!   REALTYPE                  :: DiffEpsup,DiffEpsdw,pos_bc
!   REALTYPE                  :: prod,buoyan,diss
!   REALTYPE                  :: prod_pos,prod_neg,buoyan_pos,buoyan_neg
!   REALTYPE                  :: ki,epslim,peps,EpsOverTke,NN_pos
!   REALTYPE                  :: cnpar=_ONE_
!   REALTYPE                  :: avh(0:nlev),sig_eff(0:nlev)
!   REALTYPE                  :: Lsour(0:nlev),Qsour(0:nlev)
!   REALTYPE                  :: ce3
!
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
!  Determination of the turbulent Schmidt number for the Craig & Banner (1994)
!  parameterisation for breaking surface waves suggested by Burchard (2001):
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
!  clip at eps_min
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
!        compute dissipative scale
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
    "DissipationEquationWorkspace",
    "step_dissipationeq",
]

_CNPAR: float = 1.0


class DissipationEquationWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated epsilon equation."""

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
    avh: ti.Field
    sig_eff: ti.Field
    l_sour: ti.Field
    q_sour: ti.Field
    u_taus: ti.Field
    u_taub: ti.Field
    z0s: ti.Field
    z0b: ti.Field
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
        self.allocate_many(("avh", "sig_eff", "l_sour", "q_sour"))
        self.allocate_many(("u_taus", "u_taub", "z0s", "z0b"))
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))


@ti.func
def _fk_craig(u_tau, eta):  # type: ignore[no-untyped-def]
    return eta * u_tau**3


@ti.func
def _epsilon_bc_value(  # type: ignore[no-untyped-def]
    bc,
    type_,
    zi,
    ki,
    z0,
    u_tau,
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
):
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = cde * ti.pow(ki, 1.5) / (kappa * (zi + z0))
        else:
            value = ti.pow(cm0, 4.0) * ki**2 / (sig_e * (zi + z0))

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ti.pow(
            -sig_k * f_k / (cmsf * gen_alpha * gen_l),
            2.0 / 3.0,
        ) / ti.pow(z0, gen_alpha)

        if bc == _DIRICHLET:
            value = (
                cde
                * ti.pow(capital_k, 1.5)
                / gen_l
                * ti.pow(zi + z0, 1.5 * gen_alpha - 1.0)
            )
        else:
            value = (
                -cmsf
                * cde
                / sig_e0
                * capital_k**2
                * (1.5 * gen_alpha - 1.0)
                * ti.pow(zi + z0, 2.0 * gen_alpha - 1.0)
            )

    return value


@ti_kernel
def step_dissipationeq(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    ce1: ti.f64,
    ce2: ti.f64,
    ce3plus: ti.f64,
    ce3minus: ti.f64,
    cex: ti.f64,
    ce4: ti.f64,
    cm0: ti.f64,
    cde: ti.f64,
    kappa: ti.f64,
    galp: ti.f64,
    sig_k: ti.f64,
    sig_e: ti.f64,
    sig_e0: ti.f64,
    sig_peps: ti.i32,
    length_lim: ti.i32,
    eps_min: ti.f64,
    psi_ubc: ti.i32,
    psi_lbc: ti.i32,
    ubc_type: ti.i32,
    lbc_type: ti.i32,
    cmsf: ti.f64,
    cw: ti.f64,
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
    avh: TemplateArg,
    sig_eff: TemplateArg,
    l_sour: TemplateArg,
    q_sour: TemplateArg,
    u_taus: TemplateArg,
    u_taub: TemplateArg,
    z0s: TemplateArg,
    z0b: TemplateArg,
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
):
    r"""Advance the dynamic epsilon-equation for one or more columns."""

    for col in range(n_cols):
        for i in range(nlev + 1):
            avh[col, i] = 0.0
            sig_eff[col, i] = sig_e
            l_sour[col, i] = 0.0
            q_sour[col, i] = 0.0

        if sig_peps != 0:
            sig_eff[col, nlev] = sig_e0
            for i in range(1, nlev):
                peps = (P[col, i] + Px[col, i] + B[col, i]) / eps[col, i]
                if peps > 1.0:
                    peps = 1.0
                sig_eff[col, i] = peps * sig_e + (1.0 - peps) * sig_e0
            sig_eff[col, 0] = sig_e

        for i in range(1, nlev):
            avh[col, i] = num[col, i] / sig_eff[col, i]

            ce3 = ce3minus
            if B[col, i] > 0.0:
                ce3 = ce3plus

            eps_over_tke = eps[col, i] / tkeo[col, i]
            prod = eps_over_tke * (
                ce1 * P[col, i] + cex * Px[col, i] + ce4 * PSTK[col, i]
            )
            buoyan = ce3 * eps_over_tke * B[col, i]
            diss = ce2 * eps_over_tke * eps[col, i]

            if prod + buoyan > 0.0:
                q_sour[col, i] = prod + buoyan
                l_sour[col, i] = -diss / eps[col, i]
            else:
                q_sour[col, i] = prod
                l_sour[col, i] = -(diss - buoyan) / eps[col, i]

        ki = tke[col, nlev - 1]
        pos_bc = h[col, nlev]
        if psi_ubc == _NEUMANN:
            pos_bc = 0.5 * h[col, nlev]
        diff_eps_up = _epsilon_bc_value(
            psi_ubc,
            ubc_type,
            pos_bc,
            ki,
            z0s[col, 0],
            u_taus[col, 0],
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

        ki = tke[col, 1]
        pos_bc = h[col, 1]
        if psi_lbc == _NEUMANN:
            pos_bc = 0.5 * h[col, 1]
        diff_eps_down = _epsilon_bc_value(
            psi_lbc,
            lbc_type,
            pos_bc,
            ki,
            z0b[col, 0],
            u_taub[col, 0],
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

        diff_face_column(
            col,
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

        eps[col, nlev] = _epsilon_bc_value(
            _DIRICHLET,
            ubc_type,
            z0s[col, 0],
            tke[col, nlev],
            z0s[col, 0],
            u_taus[col, 0],
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
        eps[col, 0] = _epsilon_bc_value(
            _DIRICHLET,
            lbc_type,
            z0b[col, 0],
            tke[col, 0],
            z0b[col, 0],
            u_taub[col, 0],
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
