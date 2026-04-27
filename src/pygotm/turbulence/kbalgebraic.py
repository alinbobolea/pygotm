# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The algebraic kb-equation\label{sec:kbalgebraic}
!
! !INTERFACE:
!   subroutine kbalgebraic(nlev)
!
! !DESCRIPTION:
! The algebraic equation for $k_b$ simply assumes equilibrium in \eq{kbeq},
! \begin{equation}
!   \label{kbEquilibrium}
!   P_b = \epsilon_b
!   \point
! \end{equation}
! This equation can be re-written as
! \begin{equation}
!   \label{kbAgebraic}
!   k_b = \dfrac{k_b \epsilon}{k \epsilon_b} \dfrac{k}{\epsilon} P_b
!       = r \dfrac{k}{\epsilon} P_b = c_b \dfrac{k}{\epsilon} P_b
!   \comma
! \end{equation}
! where we used the definition of the time scale ratio $r$ in
! \eq{DefR}, and assumed that $r=c_b$ is a constant.
!
!
! !USES:
!   use turbulence,  only:     tke,eps,kb,Pb
!   use turbulence,  only:     ctt,kb_min
!
!  IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
! number of vertical layers
!   integer,  intent(in)                 :: nlev
!
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
!
!   integer                             :: i
!
!-----------------------------------------------------------------------
!BOC
!
!  clip at kb_min
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
    "KBAlgebraicWorkspace",
    "step_kbalgebraic",
]


class KBAlgebraicWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated algebraic buoyancy-variance update."""

    tke: ti.Field
    eps: ti.Field
    kb: ti.Field
    Pb: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "eps", "kb", "Pb"))


@ti_kernel
def step_kbalgebraic(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    ctt: ti.f64,
    kb_min: ti.f64,
    tke: TemplateArg,
    eps: TemplateArg,
    kb: TemplateArg,
    Pb: TemplateArg,
):
    r"""Advance the algebraic buoyancy-variance closure for one or more columns."""

    for col in range(n_cols):
        for i in range(nlev + 1):
            kb[col, i] = ctt * tke[col, i] / eps[col, i] * Pb[col, i]

            if kb[col, i] < kb_min:
                kb[col, i] = kb_min
