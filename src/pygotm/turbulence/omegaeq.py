# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic epsilon-equation \label{sec:dissipationeq}
!
! !INTERFACE:
!   subroutine omegaeq(nlev,dt,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! The $k$-$\omega$ model described by \cite{UmlaufEtAl3003} solves
! a transport equation for the inverse turbulence time scale,
! $ \omega = (c_\mu^0)^4 \varepsilon /k$, of the following form:
! \begin{equation}
!   \label{omega}
!   \dot{\omega}
!   =
!   {\cal D}_\omega
!   + \frac{\omega}{k} ( c_{\omega 1} P + c_{\omega 3} G
!                        + c_{\omega x} P_x
!                        + c_{\omega 4} P_s
!                        - c_{\omega 2} \varepsilon )
!   \comma
! \end{equation}
! where $\dot{\omega}$ denotes the material derivative of $\omega$.
! The production terms $P$ and $G$ follow from \eq{PandG}.
! $P_s$ is Stokes shear production defined in \eq{computePs}
! and $P_x$ accounts for extra turbulence production.
! ${\cal D}_\omega$ represents the sum of the viscous and turbulent
! transport terms.
!
! For horizontally homogeneous flows, the transport term ${\cal D}_\omega$
! appearing in \eq{dissipation} is presently expressed by a simple
! gradient formulation,
! \begin{equation}
!   \label{diffusionOmega}
!   {\cal D}_\omega = \frstder{z}
!    \left( \dfrac{\nu_t}{\sigma_\omega} \partder{\omega}{z} \right)
!  \comma
! \end{equation}
! where $\sigma_\omega$ is the constant Schmidt-number for $\omega$.
!
! Model constants are summarized in \tab{tab:KW_constants}. Similar
! to the two-equations models, the model parameter $c_{omega 3}$
! determines the value of the stationory Richardson number. It is
! computed numerically by solving \eq{Ri_st}.
! \begin{table}[ht]
!   \begin{center}
! \begin{tabular}{cccccc}
!     & $c_\mu^0$ & $\sigma_k$  & $\sigma_\omega$
!     & $c_{\omega 1}$ & $c_{\omega 2}$  \\[1mm] \hline
!     \cite{Rodi87} & $0.55$ & $2.0$ &  $2.0$ & $0.56$ & $0.83$ \\
!   \end{tabular}
!   \caption{\label{tab:KW_constants} Constants appearing in
!    \eq{omega} and \eq{diffusionOmega}.}
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
!   use turbulence, only: cw1,cw2,cw3plus,cw3minus,cwx,cw4
!   use turbulence, only: cm0,cde,galp,length_lim
!   use turbulence, only: omega_bc, psi_ubc, psi_lbc, ubc_type, lbc_type
!   use turbulence, only: sig_w
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
!
!EOP
!------------------------------------------------------------------------
!
! !LOCAL VARIABLES:
!   REALTYPE                  :: DiffOmgUp,DiffOmgDw,pos_bc
!   REALTYPE                  :: prod,buoyan,diss
!   REALTYPE                  :: prod_pos,prod_neg,buoyan_pos,buoyan_neg
!   REALTYPE                  :: ki,epslim,OmgOverTke,NN_pos
!   REALTYPE                  :: cnpar=_ONE_
!   REALTYPE                  :: omega(0:nlev)
!   REALTYPE                  :: avh(0:nlev)
!   REALTYPE                  :: Lsour(0:nlev),Qsour(0:nlev)
!   REALTYPE                  :: cw3
!
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
!  re-construct omega at "old" timestep
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
!     recover dissipation rate from k and omega
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
    "OmegaEquationWorkspace",
    "step_omegaeq",
]

_CNPAR: float = 1.0


class OmegaEquationWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated omega equation."""

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
    omega: ti.Field
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
        self.allocate_many(("omega", "avh", "l_sour", "q_sour"))
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))


@ti.func
def _fk_craig(u_tau, eta):  # type: ignore[no-untyped-def]
    return eta * u_tau**3


@ti.func
def _omega_bc_value(  # type: ignore[no-untyped-def]
    bc,
    type_,
    zi,
    ki,
    z0,
    u_tau,
    cm0,
    kappa,
    sig_w,
    sig_k,
    cmsf,
    cw,
    gen_alpha,
    gen_l,
):
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = ti.sqrt(ki) / (cm0 * kappa * (zi + z0))
        else:
            value = ki / (sig_w * (zi + z0))

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ti.pow(
            -sig_k * f_k / (cmsf * gen_alpha * gen_l),
            2.0 / 3.0,
        ) / ti.pow(z0, gen_alpha)

        if bc == _DIRICHLET:
            value = (
                ti.sqrt(capital_k)
                / (cm0 * gen_l)
                * ti.pow(zi + z0, 0.5 * gen_alpha - 1.0)
            )
        else:
            value = (
                -cmsf
                * capital_k
                * (0.5 * gen_alpha - 1.0)
                / (sig_w * cm0)
                * ti.pow(zi + z0, gen_alpha - 1.0)
            )

    return value


@ti_kernel
def step_omegaeq(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    cw1: ti.f64,
    cw2: ti.f64,
    cw3plus: ti.f64,
    cw3minus: ti.f64,
    cwx: ti.f64,
    cw4: ti.f64,
    sig_w: ti.f64,
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
    omega: TemplateArg,
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
    r"""Advance the dynamic omega-equation for one or more columns."""

    for col in range(n_cols):
        for i in range(nlev + 1):
            omega[col, i] = ti.sqrt(tkeo[col, i]) / (cm0 * L[col, i])
            avh[col, i] = 0.0
            l_sour[col, i] = 0.0
            q_sour[col, i] = 0.0

        for i in range(1, nlev):
            avh[col, i] = num[col, i] / sig_w

            cw3 = cw3minus
            if B[col, i] > 0.0:
                cw3 = cw3plus

            omg_over_tke = omega[col, i] / tkeo[col, i]
            prod = omg_over_tke * (
                cw1 * P[col, i] + cwx * Px[col, i] + cw4 * PSTK[col, i]
            )
            buoyan = cw3 * omg_over_tke * B[col, i]
            diss = cw2 * omg_over_tke * eps[col, i]

            if prod + buoyan > 0.0:
                q_sour[col, i] = prod + buoyan
                l_sour[col, i] = -diss / omega[col, i]
            else:
                q_sour[col, i] = prod
                l_sour[col, i] = -(diss - buoyan) / omega[col, i]

        ki = tke[col, nlev - 1]
        pos_bc = h[col, nlev]
        if psi_ubc == _NEUMANN:
            pos_bc = 0.5 * h[col, nlev]
        diff_omega_up = _omega_bc_value(
            psi_ubc,
            ubc_type,
            pos_bc,
            ki,
            z0s[col, 0],
            u_taus[col, 0],
            cm0,
            kappa,
            sig_w,
            sig_k,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
        )

        ki = tke[col, 1]
        pos_bc = h[col, 1]
        if psi_lbc == _NEUMANN:
            pos_bc = 0.5 * h[col, 1]
        diff_omega_down = _omega_bc_value(
            psi_lbc,
            lbc_type,
            pos_bc,
            ki,
            z0b[col, 0],
            u_taub[col, 0],
            cm0,
            kappa,
            sig_w,
            sig_k,
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
            diff_omega_up,
            diff_omega_down,
            avh,
            l_sour,
            q_sour,
            omega,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )

        omega[col, nlev] = _omega_bc_value(
            _DIRICHLET,
            ubc_type,
            z0s[col, 0],
            tke[col, nlev],
            z0s[col, 0],
            u_taus[col, 0],
            cm0,
            kappa,
            sig_w,
            sig_k,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
        )
        omega[col, 0] = _omega_bc_value(
            _DIRICHLET,
            lbc_type,
            z0b[col, 0],
            tke[col, 0],
            z0b[col, 0],
            u_taub[col, 0],
            cm0,
            kappa,
            sig_w,
            sig_k,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
        )

        for i in range(nlev + 1):
            eps[col, i] = ti.pow(cm0, 4.0) * tke[col, i] * omega[col, i]
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
