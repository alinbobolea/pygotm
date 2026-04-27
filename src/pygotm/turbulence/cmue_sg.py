# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The Schumann and Gerz (1995) stability function\label{sec:sg}
!
! !INTERFACE:
!   subroutine cmue_sg(nlev)
!
! !DESCRIPTION:
!  This subroutine computes stability functions according to
! \begin{equation}
! c_{\mu}=c_{\mu}^0,\qquad c'_{\mu}=\frac{c_{\mu}^0}{Pr_t}
! \end{equation}
! with constant $c_{\mu}^0$. Based simulation data on stratified homogeneous
! shear-flows, \cite{SchumannGerz95} proposed the empirical relation
! for the turbulent Prandtl--number,
! \begin{equation}
!   Pr_t = Pr_t^0 \exp\left(-\frac{Ri}{Pr_t^0 Ri^{\infty}}\right)
!   -\frac{Ri}{Ri^{\infty}}
!   \comma
! \end{equation}
! where where $Ri$ is the gradient Richardson--number and $Pr_t^0$
! is the turbulent Prandtl--number for $Ri \rightarrow 0$. $Pr_t^0$
! and the fixed value $c_\mu^0$ have to be set in {\tt gotm.yaml}.
! \cite{SchumannGerz95}  suggested $Pr_t^0=0.74$ and $Ri^{\infty}=0.25$.
!
! !USES:
!   use turbulence, only: Prandtl0_fix,cm0_fix
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
!   REALTYPE,parameter        :: limit=3.
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
    "CmueSGWorkspace",
    "step_cmue_sg",
]

_RI_EPSILON: float = 1.0e-8
_RI_THRESHOLD: float = 1.0e-10
_SG_LIMIT: float = 3.0
_RI_INFINITY: float = 0.25


class CmueSGWorkspace(TaichiFieldCollection):
    """Taichi fields for the Schumann-Gerz stability functions."""

    as_: ti.Field
    an: ti.Field
    cmue1: ti.Field
    cmue2: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("as_", "an", "cmue1", "cmue2"))


@ti_kernel
def step_cmue_sg(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    cm0_fix: ti.f64,
    prandtl0_fix: ti.f64,
    as_: TemplateArg,
    an: TemplateArg,
    cmue1: TemplateArg,
    cmue2: TemplateArg,
):
    r"""Update Schumann-Gerz stability functions for one or more columns."""

    for col in range(n_cols):
        for i in range(1, nlev):
            ri = an[col, i] / (as_[col, i] + _RI_EPSILON)
            prandtl = prandtl0_fix
            if ri >= _RI_THRESHOLD:
                prandtl = (
                    prandtl0_fix
                    * ti.exp(-ri / (prandtl0_fix * _RI_INFINITY))
                    + ri / _RI_INFINITY
                )

            cmue1[col, i] = cm0_fix
            cmue2[col, i] = cm0_fix / ti.min(_SG_LIMIT, prandtl)
