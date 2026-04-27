# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic kb-equation \label{sec:kbeq}
!
! !INTERFACE:
!   subroutine kbeq(nlev,dt,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! The transport equation for (half the) buoyancy variance,
! $k_b=\mean{b'^2}/2$,
! follows from the equation for the buoyancy fluctations (see \cite{Sander98a}).
! In the case of a Boussinesq-fluid, this equation can
! be written as
! \begin{equation}
!   \label{kbeq}
!   \dot{k_b}
!   =
!   {\cal D}_b +  P_b - \epsilon_b
!   \comma
! \end{equation}
! where $\dot{k_b}$ denotes the material derivative of $k_b$. $P_b$ is
! the production of $k_b$ be mean density gradients,  and
! $\epsilon_b$ the rate of molecular destruction. ${\cal D}_b$ represents
! the sum of the viscous and turbulent transport terms. It is presently
! evaluated with a simple down gradient model in GOTM.
!
! The production of buoyancy variance by the vertical density gradient
! is
! \begin{equation}
!   \label{Pbvertical}
!   P_b = - \mean{w'b'} \partder{B}{z} = -\mean{w'b'} N^2
!   \point
! \end{equation}
! Its computation is discussed in \sect{sec:production}.
!
! The rate of molecular destruction, $\epsilon_b$,  can be computed
! from either a transport equation or a algebraic expression, \sect{sec:updateEpsb}.
!
!
! !USES:
!   use turbulence,   only: Pb,epsb,nuh
!   use turbulence,   only: kb,kb_min
!   use turbulence,   only: k_ubc, k_lbc, ubc_type, lbc_type
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
!   REALTYPE                  :: DiffKbup,DiffKbdw,pos_bc
!   REALTYPE                  :: prod,diss
!   REALTYPE                  :: prod_pos,prod_neg
!   REALTYPE                  :: cnpar=_ONE_
!   REALTYPE                  :: avh(0:nlev)
!   REALTYPE                  :: Lsour(0:nlev),Qsour(0:nlev)
!
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
!  compute diffusivity
!   avh = nuh
!
!   do i=1,nlev-1
!
! !     compute production terms in k-equation
!      prod     = Pb(i)
!      diss     = epsb(i)
!
! !     compute positive and negative parts of RHS
!      prod_pos    =  0.5*( prod   + abs(prod  ) )
!      prod_neg    = prod    - prod_pos
!
! !     compose source terms
!      Qsour(i) =   prod_pos
!      Lsour(i) =  (prod_neg - diss)/kb(i)
!
!   end do
!
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
!   DiffKbup  = _ZERO_
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
!   DiffKbdw  = _ZERO_
!
!
! !  do diffusion step
!   call diff_face(nlev,dt,cnpar,h,k_ubc,k_lbc,                          &
!                  DiffKbup,DiffKbdw,avh,Lsour,Qsour,kb)
!
!
! !  fill top and bottom value with something nice
! !  (only for output)
!   kb(nlev)  = _ZERO_
!   kb(0   )  = _ZERO_
!
! !  clip at k_min
!   do i=0,nlev
!      kb(i) = max(kb(i),kb_min)
!   enddo
!
!   return
!   end subroutine kbeq
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.util.diff_face import diff_face_column

__all__ = [
    "KBEquationWorkspace",
    "step_kbeq",
]

_CNPAR: float = 1.0
_ZERO: float = 0.0


class KBEquationWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated buoyancy-variance equation."""

    kb: ti.Field
    h: ti.Field
    Pb: ti.Field
    epsb: ti.Field
    nuh: ti.Field
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
        self.allocate_many(("kb", "h"))
        self.allocate_many(("Pb", "epsb", "nuh"))
        self.allocate_many(("avh", "l_sour", "q_sour"))
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))


@ti_kernel
def step_kbeq(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    kb_min: ti.f64,
    k_ubc: ti.i32,
    k_lbc: ti.i32,
    kb: TemplateArg,
    h: TemplateArg,
    Pb: TemplateArg,
    epsb: TemplateArg,
    nuh: TemplateArg,
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
    r"""Advance the dynamic buoyancy-variance equation for one or more columns."""

    for col in range(n_cols):
        for i in range(nlev + 1):
            avh[col, i] = nuh[col, i]
            l_sour[col, i] = 0.0
            q_sour[col, i] = 0.0

        for i in range(1, nlev):
            prod = Pb[col, i]
            diss = epsb[col, i]

            prod_pos = 0.5 * (prod + ti.abs(prod))
            prod_neg = prod - prod_pos

            q_sour[col, i] = prod_pos
            l_sour[col, i] = (prod_neg - diss) / kb[col, i]

        diff_face_column(
            col,
            nlev,
            dt,
            _CNPAR,
            h,
            k_ubc,
            k_lbc,
            _ZERO,
            _ZERO,
            avh,
            l_sour,
            q_sour,
            kb,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )

        kb[col, nlev] = _ZERO
        kb[col, 0] = _ZERO

        for i in range(nlev + 1):
            if kb[col, i] < kb_min:
                kb[col, i] = kb_min
