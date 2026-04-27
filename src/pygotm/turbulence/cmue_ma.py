# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The Munk and Anderson (1948) stability function\label{sec:cmueMA}
!
! !INTERFACE:
!   subroutine cmue_ma(nlev)
!
! !DESCRIPTION:
!  This subroutine computes the stability functions
!  according to \cite{MunkAnderson48}. These are expressed
!  by the empirical relations
!  \begin{equation}
!    \begin{array}{ll}
!      c_{\mu} = c_\mu^0                          \comma             \\[3mm]
!      c_{\mu}'= \dfrac{c_{\mu}}{Pr_t^0} \,
!      \dfrac{(1+10 Ri)^{1/2}}{(1+3.33 Ri)^{3/2}} \comma &  Ri \geq 0 \\
!      c_{\mu}'= c_{\mu}                          \comma &  Ri  <   0
!      \comma
!    \end{array}
!  \end{equation}
!  where where $Ri$ is the gradient Richardson-number and $Pr_t^0$
! is the turbulent Prandtl-number for $Ri \rightarrow 0$. $Pr_t^0$
! and the fixed value $c_\mu^0$ have to be set in {\tt gotm.yaml}.
!
! !USES:
!   use turbulence, only: cm0_fix,Prandtl0_fix
!   use turbulence, only: cmue1,cmue2,as,an
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   integer, intent(in)                 :: nlev
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                   :: i
!   REALTYPE                  :: Ri,Prandtl
!
!-----------------------------------------------------------------------
!BOC
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
    "CmueMAWorkspace",
    "step_cmue_ma",
]

_RI_EPSILON: float = 1.0e-8
_RI_THRESHOLD: float = 1.0e-10


class CmueMAWorkspace(TaichiFieldCollection):
    """Taichi fields for the Munk-Anderson stability functions."""

    as_: ti.Field
    an: ti.Field
    cmue1: ti.Field
    cmue2: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("as_", "an", "cmue1", "cmue2"))


@ti_kernel
def step_cmue_ma(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    cm0_fix: ti.f64,
    prandtl0_fix: ti.f64,
    as_: TemplateArg,
    an: TemplateArg,
    cmue1: TemplateArg,
    cmue2: TemplateArg,
):
    r"""Update Munk-Anderson stability functions for one or more columns."""

    for col in range(n_cols):
        for i in range(1, nlev):
            ri = an[col, i] / (as_[col, i] + _RI_EPSILON)
            prandtl = prandtl0_fix
            if ri >= _RI_THRESHOLD:
                prandtl = (
                    prandtl0_fix
                    * (1.0 + 3.33 * ri) ** 1.5
                    / ti.sqrt(1.0 + 10.0 * ri)
                )
            cmue1[col, i] = cm0_fix
            cmue2[col, i] = cm0_fix / prandtl
