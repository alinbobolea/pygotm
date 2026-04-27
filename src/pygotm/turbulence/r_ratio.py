# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Update time scale ratio
!
! !INTERFACE:
   subroutine r_ratio(nlev)
!
! !DESCRIPTION:
! This routine updates the ratio $r$ of the dissipation
! time scales as defined in \eq{DefR}.
!
! !USES:
  use turbulence,  only:     tke,eps,kb,epsb
  use turbulence,  only:     r
!
  IMPLICIT NONE
!
! !INPUT PARAMETERS:
  integer, intent(in)        :: nlev
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
!BOC
!
   r = kb*eps/(epsb*tke)
!
   return
end subroutine r_ratio
!
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel

__all__ = ["RRatioWorkspace", "step_r_ratio"]


class RRatioWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated dissipation-time-scale ratio update."""

    tke: ti.Field
    eps: ti.Field
    kb: ti.Field
    epsb: ti.Field
    r: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "eps", "kb", "epsb", "r"))


@ti_kernel
def step_r_ratio(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    tke: TemplateArg,
    eps: TemplateArg,
    kb: TemplateArg,
    epsb: TemplateArg,
    r: TemplateArg,
):
    r"""Update the dissipation-time-scale ratio ``r`` for one or more columns."""

    for col in range(n_cols):
        for i in range(nlev + 1):
            r[col, i] = kb[col, i] * eps[col, i] / (epsb[col, i] * tke[col, i])
