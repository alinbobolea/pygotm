# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic k-equation \label{sec:tkeeq}
!
! !INTERFACE:
!   subroutine tkeeq(nlev,dt,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! The transport equation for the turbulent kinetic energy, $k$,
! follows immediately from the contraction of the Reynolds-stress
! tensor. In the case of a Boussinesq-fluid, this equation can
! be written as
! \begin{equation}
!   \label{tkeA}
!   \dot{k}
!   =
!   {\cal D}_k +  P + G + P_x + P_s - \epsilon
!   \comma
! \end{equation}
! where $\dot{k}$ denotes the material derivative of $k$. $P$ and $G$ are
! the production of $k$ by mean shear and buoyancy, respectively, and
! $\epsilon$ the rate of dissipation.
! $P_s$ is Stokes shear production defined in \eq{computePs}
! and $P_x$ accounts for extra turbulence production.
! ${\cal D}_k$ represents the sum of
! the viscous and turbulent transport terms.
! For horizontally homogeneous flows, the transport term ${\cal D}_k$
! appearing in \eq{tkeA} is presently expressed by a simple
! gradient formulation,
! \begin{equation}
!   \label{diffusionTKE}
!   {\cal D}_k = \frstder{z} \left( \dfrac{\nu_t}{\sigma_k} \partder{k}{z} \right)
!  \comma
! \end{equation}
! where $\sigma_k$ is the constant Schmidt-number for $k$.
!
! In horizontally homogeneous flows, the shear and the buoyancy
! production, $P$ and $G$, can be written as
! \begin{equation}
!   \label{PandG}
!   \begin{array}{rcl}
!   P &=& - \mean{u'w'} \partder{U}{z} - \mean{v'w'} \partder{V}{z}  \comma \\[3mm]
!   G &=&  \mean{w'b'}                                               \comma
!   \end{array}
! \end{equation}
! see \eq{PG}. Their computation is discussed in \sect{sec:production}.
!
! The rate of dissipation, $\epsilon$, can be either obtained directly
! from its parameterised transport equation as discussed in
! \sect{sec:dissipationeq}, or from any other model yielding
! an appropriate description of the dissipative length-scale, $l$.
! Then, $\epsilon$ follows from the well-known cascading relation
! of turbulence,
! \begin{equation}
!   \label{epsilon}
!   \epsilon = (c_\mu^0)^3 \frac{k^{\frac{3}{2}}}{l}
!   \comma
! \end{equation}
! where $c_\mu^0$ is a constant of the model.
!
! !USES:
!   use turbulence,   only: P,B,Px,PSTK,num
!   use turbulence,   only: tke,tkeo,k_min,eps
!   use turbulence,   only: k_bc, k_ubc, k_lbc, ubc_type, lbc_type
!   use turbulence,   only: sig_k
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
!                     (re-write after first version of
!                      H. Burchard and K. Bolding)
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
!
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
!   tkeo=tke
!
!   do i=1,nlev-1
!
! !     compute diffusivity
!      avh(i) = num(i)/sig_k
!
! !     compute production terms in k-equation
!      prod     = P(i) + Px(i) + PSTK(i)
!      buoyan   = B(i)
!      diss     = eps(i)
!
!
!      if (prod+buoyan.gt.0) then
!         Qsour(i)  = prod+buoyan
!         Lsour(i) = -diss/tke(i)
!      else
!         Qsour(i)  = prod
!         Lsour(i) = -(diss-buoyan)/tke(i)
!      end if
!
!   end do
!
!
! !  position for upper BC
!   if (k_ubc.eq.Neumann) then
! !     flux at center "nlev"
!      pos_bc = 0.5*h(nlev)
!   else
! !     value at face "nlev-1"
!      pos_bc = h(nlev)
!   end if
!
! !  obtain BC for upper boundary of type "ubc_type"
!   DiffKup  = k_bc(k_ubc,ubc_type,pos_bc,z0s,u_taus)
!
!
! !  position for lower BC
!   if (k_lbc.eq.Neumann) then
! !     flux at center "1"
!      pos_bc = 0.5*h(1)
!   else
! !     value at face "1"
!      pos_bc = h(1)
!   end if
!
! !  obtain BC for lower boundary of type "lbc_type"
!   DiffKdw  = k_bc(k_lbc,lbc_type,pos_bc,z0b,u_taub)
!
!
! !  do diffusion step
!   call diff_face(nlev,dt,cnpar,h,k_ubc,k_lbc,                          &
!                  DiffKup,DiffKdw,avh,Lsour,Qsour,tke)
!
!
! !  fill top and bottom value with something nice
! !  (only for output)
!   tke(nlev)  = k_bc(Dirichlet,ubc_type,z0s,z0s,u_taus)
!   tke(0   )  = k_bc(Dirichlet,lbc_type,z0b,z0b,u_taub)
!
! !  clip at k_min
!   do i=0,nlev
!      tke(i) = max(tke(i),k_min)
!   enddo
!
!   return
!   end subroutine tkeeq
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel
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
from pygotm.util.diff_face import diff_face_column

__all__ = [
    "TKEEquationWorkspace",
    "step_tkeeq",
]

_CNPAR: float = 1.0


class TKEEquationWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated TKE equation."""

    tke: ti.Field
    tkeo: ti.Field
    h: ti.Field
    P: ti.Field
    B: ti.Field
    Px: ti.Field
    PSTK: ti.Field
    num: ti.Field
    eps: ti.Field
    avh: ti.Field
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
        self.allocate_many(("tke", "tkeo", "h"))
        self.allocate_many(("P", "B", "Px", "PSTK", "num", "eps"))
        self.allocate_many(("avh", "l_sour", "q_sour"))
        self.allocate_many(("u_taus", "u_taub", "z0s", "z0b"))
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))


@ti.func
def _fk_craig(u_tau, eta):  # type: ignore[no-untyped-def]
    return eta * u_tau**3


@ti.func
def _k_bc_value(  # type: ignore[no-untyped-def]
    bc,
    type_,
    zi,
    z0,
    u_tau,
    cm0,
    sig_k,
    cmsf,
    cw,
    gen_alpha,
    gen_l,
):
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = u_tau**2 / cm0**2
        else:
            value = 0.0

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ti.pow(
            -sig_k * f_k / (cmsf * gen_alpha * gen_l),
            2.0 / 3.0,
        ) / ti.pow(z0, gen_alpha)

        if bc == _DIRICHLET:
            value = capital_k * ti.pow(zi + z0, gen_alpha)
        else:
            value = (
                -cmsf
                / sig_k
                * ti.pow(capital_k, 1.5)
                * gen_alpha
                * gen_l
                * ti.pow(zi + z0, 1.5 * gen_alpha)
            )

    return value


@ti_kernel
def step_tkeeq(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    sig_k: ti.f64,
    k_min: ti.f64,
    k_ubc: ti.i32,
    k_lbc: ti.i32,
    ubc_type: ti.i32,
    lbc_type: ti.i32,
    cm0: ti.f64,
    cmsf: ti.f64,
    cw: ti.f64,
    gen_alpha: ti.f64,
    gen_l: ti.f64,
    tke: TemplateArg,
    tkeo: TemplateArg,
    h: TemplateArg,
    P: TemplateArg,
    B: TemplateArg,
    Px: TemplateArg,
    PSTK: TemplateArg,
    num: TemplateArg,
    eps: TemplateArg,
    avh: TemplateArg,
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
    r"""Advance the dynamic k-equation for one or more columns."""

    for col in range(n_cols):
        for i in range(nlev + 1):
            tkeo[col, i] = tke[col, i]
            avh[col, i] = 0.0
            l_sour[col, i] = 0.0
            q_sour[col, i] = 0.0

        for i in range(1, nlev):
            avh[col, i] = num[col, i] / sig_k

            prod = P[col, i] + Px[col, i] + PSTK[col, i]
            buoyan = B[col, i]
            diss = eps[col, i]

            if prod + buoyan > 0.0:
                q_sour[col, i] = prod + buoyan
                l_sour[col, i] = -diss / tke[col, i]
            else:
                q_sour[col, i] = prod
                l_sour[col, i] = -(diss - buoyan) / tke[col, i]

        pos_bc = h[col, nlev]
        if k_ubc == _NEUMANN:
            pos_bc = 0.5 * h[col, nlev]
        diff_k_up = _k_bc_value(
            k_ubc,
            ubc_type,
            pos_bc,
            z0s[col, 0],
            u_taus[col, 0],
            cm0,
            sig_k,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
        )

        pos_bc = h[col, 1]
        if k_lbc == _NEUMANN:
            pos_bc = 0.5 * h[col, 1]
        diff_k_down = _k_bc_value(
            k_lbc,
            lbc_type,
            pos_bc,
            z0b[col, 0],
            u_taub[col, 0],
            cm0,
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

        tke[col, nlev] = _k_bc_value(
            _DIRICHLET,
            ubc_type,
            z0s[col, 0],
            z0s[col, 0],
            u_taus[col, 0],
            cm0,
            sig_k,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
        )
        tke[col, 0] = _k_bc_value(
            _DIRICHLET,
            lbc_type,
            z0b[col, 0],
            z0b[col, 0],
            u_taub[col, 0],
            cm0,
            sig_k,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
        )

        for i in range(nlev + 1):
            if tke[col, i] < k_min:
                tke[col, i] = k_min
