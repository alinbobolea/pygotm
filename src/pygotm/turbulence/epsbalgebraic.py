# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The algebraic epsilonb-equation\label{sec:epsbalgebraic}
!
! !INTERFACE:
!   subroutine epsbalgebraic(nlev)
!
! !DESCRIPTION:
! The algebraic equation for $\epsilon_b$, the molecular rate of
! destruction of buoyancy variance, see \eq{kbeq}, simply assumes a
! constant time scale ratio $r=c_b$, see \eq{DefR}. From
! this assumption, it follows immediately that
! \begin{equation}
!   \label{epsbAgebraic}
!     \epsilon_b = \dfrac{1}{c_b} \dfrac{\epsilon}{k} k_b
!   \point
! \end{equation}
!
! !USES:
!  use turbulence,  only:     tke,eps,kb,epsb
!  use turbulence,  only:     ctt,epsb_min
!
!  IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
! number of vertical layers
!  integer,  intent(in)                 :: nlev
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
!  REALTYPE                             :: one_over_ctt
!  integer                              :: i
!
!-----------------------------------------------------------------------
!BOC
!
!  clip at epsb_min
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
    "EpsBAlgebraicWorkspace",
    "step_epsbalgebraic",
]


class EpsBAlgebraicWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated algebraic buoyancy-destruction update."""

    tke: ti.Field
    eps: ti.Field
    kb: ti.Field
    epsb: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "eps", "kb", "epsb"))


@ti_kernel
def step_epsbalgebraic(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    ctt: ti.f64,
    epsb_min: ti.f64,
    tke: TemplateArg,
    eps: TemplateArg,
    kb: TemplateArg,
    epsb: TemplateArg,
):
    r"""Advance the algebraic buoyancy-destruction closure for one or more columns."""

    one_over_ctt = 1.0 / ctt

    for col in range(n_cols):
        for i in range(nlev + 1):
            epsb[col, i] = one_over_ctt * eps[col, i] / tke[col, i] * kb[col, i]

            if epsb[col, i] < epsb_min:
                epsb[col, i] = epsb_min
