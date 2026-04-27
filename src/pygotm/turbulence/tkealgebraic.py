# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The algebraic k-equation\label{sec:tkealgebraic}
!
! !INTERFACE:
!   subroutine tkealgebraic(nlev,u_taus,u_taub,NN,SS)
!
! !DESCRIPTION:
!  This subroutine computes the turbulent kinetic energy based
!  on \eq{tkeA}, but using the local equilibrium assumption
!  \begin{equation}
!   \label{localEQa}
!     P+G-\epsilon=0
!    \point
!  \end{equation}
! This statement can be re-expressed in the form
!  \begin{equation}
!   \label{localEQb}
!     k= (c_\mu^0)^{-3} \, l^2 ( c_\mu M^2 - c'_\mu N^2 )
!    \comma
!  \end{equation}
!  were we used the expressions in \eq{PandG} together with
!  \eq{fluxes} and \eq{nu}. The rate of dissipaton, $\epsilon$,
!  has been expressed in terms of $l$ via \eq{epsilon}.
!  This equation has been implemented to update $k$ in a diagnostic
!  way. It is possible to compute the value of $k$ as the weighted average
!  of \eq{localEQb} and the value of $k$ at the old timestep. The weighting factor
!  is defined by the {\tt parameter c\_filt}. It is recommended to take this factor
!  small (e.g.\ {\tt c\_filt = 0.2}) in order to reduce the strong oscillations
!  associated with this scheme, and to couple it with an algebraically prescribed
!  length scale with the length scale limitation active ({\tt length\_lim=.true.} in
!  {\tt gotm.yaml}, see \cite{Galperinetal88}).
!
! !USES:
!   use turbulence,   only: tke,tkeo,L,k_min
!   use turbulence,   only: cmue2,cde,cmue1,cm0
!
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
!  number of vertical layers
!   integer,  intent(in)                :: nlev
!
!  surface and bottom
!  friction velocity (m/s)
!   REALTYPE, intent(in)                :: u_taus,u_taub
!
!  square of shear and buoyancy
!  frequency (1/s^2)
!   REALTYPE, intent(in)                :: NN(0:nlev),SS(0:nlev)
!
! !DEFINED PARAMETERS:
!   REALTYPE , parameter                :: c_filt=1.0
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!-----------------------------------------------------------------------
!
! !LOCAL VARIABLES:
!   integer                   :: i
!
!-----------------------------------------------------------------------
!BOC
!
!  save value at old time step
!
!  compute new tke as the weighted average of old and new value
!
!  formally compute BC
!
!
!  clip at k_min
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel

__all__ = [
    "TKEAlgebraicWorkspace",
    "step_tkealgebraic",
]

_C_FILT: float = 1.0


class TKEAlgebraicWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated algebraic TKE update."""

    tke: ti.Field
    tkeo: ti.Field
    L: ti.Field
    NN: ti.Field
    SS: ti.Field
    cmue1: ti.Field
    cmue2: ti.Field
    u_taus: ti.Field
    u_taub: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "tkeo", "L", "NN", "SS", "cmue1", "cmue2"))
        self.allocate_many(("u_taus", "u_taub"))


@ti_kernel
def step_tkealgebraic(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    k_min: ti.f64,
    cm0: ti.f64,
    cde: ti.f64,
    tke: TemplateArg,
    tkeo: TemplateArg,
    L: TemplateArg,
    NN: TemplateArg,
    SS: TemplateArg,
    cmue1: TemplateArg,
    cmue2: TemplateArg,
    u_taus: TemplateArg,
    u_taub: TemplateArg,
):
    r"""Advance the algebraic TKE closure for one or more columns."""

    for col in range(n_cols):
        for i in range(nlev + 1):
            tkeo[col, i] = tke[col, i]

        for i in range(1, nlev):
            tke[col, i] = (
                _C_FILT
                * (
                    L[col, i]
                    * L[col, i]
                    / cde
                    * (cmue1[col, i] * SS[col, i] - cmue2[col, i] * NN[col, i])
                )
                + (1.0 - _C_FILT) * tkeo[col, i]
            )

        boundary_scale = ti.sqrt(cm0 * cde)
        tke[col, 0] = u_taub[col, 0] * u_taub[col, 0] / boundary_scale
        tke[col, nlev] = u_taus[col, 0] * u_taus[col, 0] / boundary_scale

        for i in range(nlev + 1):
            if tke[col, i] < k_min:
                tke[col, i] = k_min
