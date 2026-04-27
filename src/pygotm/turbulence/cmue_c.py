# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The local, weak-equilibrium stability functions \label{sec:cmueC}
!
! !INTERFACE:
!   subroutine cmue_c(nlev)
!
! !DESCRIPTION:
!
!  This subroutine updates the explicit solution of
!  \eq{bijVertical} and \eq{giVertical} with shape indicated
!  by \eq{b13}. In addition to the simplifications discussed
!  in \sect{sec:cmueB}, $P_b = \epsilon_b$ is assumed
!  in \eq{giVertical} to eliminate the dependency on $\overline{T}$
!  according to \eq{Tequilibrium}.
!  As discussed in \sect{sec:EASM}, this implies that the last of \eq{giVertical}
!  is replaced by \eq{giVerticalEq}. Thus, the $\Gamma$-term in \eq{b13} drops
!  out, and the solution is characterized by $c_\mu$ and $c'_\mu$ only.
!
! !USES:
!   use turbulence, only: an,as,at
!   use turbulence, only: cmue1,cmue2
!   use turbulence, only: cm0
!   use turbulence, only: cc1
!   use turbulence, only: ct1,ctt
!   use turbulence, only: a1,a2,a3,a4,a5
!   use turbulence, only: at1,at2,at3,at4,at5
!
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
!  number of vertical layers
!   integer, intent(in)       :: nlev
!
! !DEFINED PARAMETERS:
!   REALTYPE, parameter       :: asLimitFact=1.0d0
!   REALTYPE, parameter       :: anLimitFact=0.5d0
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
!
!     integer                 ::   i
!     REALTYPE                ::   N,Nt
!     REALTYPE                ::   d0,d1,d2,d3,d4,d5
!     REALTYPE                ::   n0,n1,n2,nt0,nt1,nt2
!     REALTYPE                ::   dCm,nCm,nCmp,cm3_inv
!     REALTYPE                ::   tmp0,tmp1,tmp2
!     REALTYPE                ::   asMax,asMaxNum,asMaxDen
!     REALTYPE                ::   anMin,anMinNum,anMinDen
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
    "CmueCWorkspace",
    "step_cmue_c",
]

_AS_LIMIT_FACT: float = 1.0
_AN_LIMIT_FACT: float = 0.5


class CmueCWorkspace(TaichiFieldCollection):
    """Taichi fields for the local weak-equilibrium stability closure."""

    as_: ti.Field
    an: ti.Field
    cmue1: ti.Field
    cmue2: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("as_", "an", "cmue1", "cmue2"))


@ti_kernel
def step_cmue_c(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    cm0: ti.f64,
    cc1: ti.f64,
    ct1: ti.f64,
    a1: ti.f64,
    a2: ti.f64,
    a3: ti.f64,
    a5: ti.f64,
    at1: ti.f64,
    at2: ti.f64,
    at3: ti.f64,
    at5: ti.f64,
    as_: TemplateArg,
    an: TemplateArg,
    cmue1: TemplateArg,
    cmue2: TemplateArg,
):
    r"""Update the local weak-equilibrium stability functions."""

    n_val = 0.5 * cc1
    nt_val = ct1

    n_sq = n_val * n_val
    n_cube = n_sq * n_val
    nt_sq = nt_val * nt_val

    d0 = 36.0 * n_cube * nt_sq
    d1 = 84.0 * a5 * at3 * n_sq * nt_val + 36.0 * at5 * n_cube * nt_val
    d2 = 9.0 * (at2 * at2 - at1 * at1) * n_cube - 12.0 * (
        a2 * a2 - 3.0 * a3 * a3
    ) * n_val * nt_sq
    d3 = (
        12.0 * a5 * at3 * (a2 * at1 - 3.0 * a3 * at2) * n_val
        + 12.0 * a5 * at3 * (a3 * a3 - a2 * a2) * nt_val
        + 12.0 * at5 * (3.0 * a3 * a3 - a2 * a2) * n_val * nt_val
    )
    d4 = 48.0 * a5 * a5 * at3 * at3 * n_val + 36.0 * a5 * at3 * at5 * n_sq
    d5 = 3.0 * (a2 * a2 - 3.0 * a3 * a3) * (at1 * at1 - at2 * at2) * n_val

    n0 = 36.0 * a1 * n_sq * nt_sq
    n1 = (
        -12.0 * a5 * at3 * (at1 + at2) * n_sq
        + 8.0 * a5 * at3 * (6.0 * a1 - a2 - 3.0 * a3) * n_val * nt_val
        + 36.0 * a1 * at5 * n_sq * nt_val
    )
    n2 = 9.0 * a1 * (at2 * at2 - at1 * at1) * n_sq

    nt0 = 12.0 * at3 * n_cube * nt_val
    nt1 = 12.0 * a5 * at3 * at3 * n_sq
    nt2 = 9.0 * a1 * at3 * (at1 - at2) * n_sq + (
        6.0 * a1 * (a2 - 3.0 * a3) - 4.0 * (a2 * a2 - 3.0 * a3 * a3)
    ) * at3 * n_val * nt_val

    cm3_inv = 1.0 / (cm0 * cm0 * cm0)

    an_min_num = -(d1 + nt0) + ti.sqrt((d1 + nt0) * (d1 + nt0) - 4.0 * d0 * (d4 + nt1))
    an_min_den = 2.0 * (d4 + nt1)
    an_min = an_min_num / an_min_den

    for col in range(n_cols):
        for i in range(1, nlev):
            an[col, i] = ti.max(an[col, i], _AN_LIMIT_FACT * an_min)

            as_max_num = (
                d0 * n0
                + (d0 * n1 + d1 * n0) * an[col, i]
                + (d1 * n1 + d4 * n0) * an[col, i] * an[col, i]
                + d4 * n1 * an[col, i] * an[col, i] * an[col, i]
            )
            as_max_den = (
                d2 * n0
                + (d2 * n1 + d3 * n0) * an[col, i]
                + d3 * n1 * an[col, i] * an[col, i]
            )
            as_max = as_max_num / as_max_den
            as_[col, i] = ti.min(as_[col, i], _AS_LIMIT_FACT * as_max)

            d_cm = (
                d0
                + d1 * an[col, i]
                + d2 * as_[col, i]
                + d3 * an[col, i] * as_[col, i]
                + d4 * an[col, i] * an[col, i]
                + d5 * as_[col, i] * as_[col, i]
            )
            n_cm = n0 + n1 * an[col, i] + n2 * as_[col, i]
            n_cmp = nt0 + nt1 * an[col, i] + nt2 * as_[col, i]

            cmue1[col, i] = cm3_inv * n_cm / d_cm
            cmue2[col, i] = cm3_inv * n_cmp / d_cm
