# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The non-local, approximate weak-equilibrium stability function\label{sec:cmueB}
!
! !INTERFACE:
!   subroutine cmue_b(nlev)
!
! !DESCRIPTION:
!  This subroutine is used to update the quantities
!  $c_\mu$, $c'_\mu$ and $\Gamma$, defined in \eq{b13}, from which all turbulent
!  fluxes can be computed. This done exactly as described in \sect{sec:cmueA}, with
!  the exception that equilibrium $P+G=\epsilon$ and $P_b = \epsilon_b$ is assumed
!  in computing the non-linear terms in \eq{NandNb}, leading to the particularly
!  simple expressions
!  \begin{equation}
!    \label{NandNbEq}
!      {\cal N} = \dfrac{c_1}{2} \comma
!      {\cal N}_b =  c_{b1}
!      \point
!  \end{equation}
!
! !USES:
!   use turbulence, only: an,as,at
!   use turbulence, only: cmue1,cmue2,gam
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
! !BUGS:
! Test stage. Do not yet use.
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
!     REALTYPE                ::   n0,n1,n2,n3,nt0,nt1,nt2
!     REALTYPE                ::   gam0,gam1,gam2
!     REALTYPE                ::   dCm,nCm,nCmp,nGam,cm3_inv
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
    "CmueBWorkspace",
    "step_cmue_b",
]


class CmueBWorkspace(TaichiFieldCollection):
    """Taichi fields for the approximate weak-equilibrium stability closure."""

    as_: ti.Field
    an: ti.Field
    at: ti.Field
    cmue1: ti.Field
    cmue2: ti.Field
    gam: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("as_", "an", "at", "cmue1", "cmue2", "gam"))


@ti_kernel
def step_cmue_b(  # type: ignore[no-untyped-def]
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
    at4: ti.f64,
    as_: TemplateArg,
    an: TemplateArg,
    at: TemplateArg,
    cmue1: TemplateArg,
    cmue2: TemplateArg,
    gam: TemplateArg,
):
    r"""Update the approximate weak-equilibrium stability functions."""

    n_val = 0.5 * cc1
    nt_val = ct1

    n_sq = n_val * n_val
    n_cube = n_sq * n_val
    nt_sq = nt_val * nt_val

    d0 = 36.0 * n_cube * nt_sq
    d1 = 84.0 * a5 * at3 * n_sq * nt_val
    d2 = 9.0 * (at2 * at2 - at1 * at1) * n_cube - 12.0 * (
        a2 * a2 - 3.0 * a3 * a3
    ) * n_val * nt_sq
    d3 = 12.0 * a5 * at3 * (a2 * at1 - 3.0 * a3 * at2) * n_val + 12.0 * a5 * at3 * (
        a3 * a3 - a2 * a2
    ) * nt_val
    d4 = 48.0 * a5 * a5 * at3 * at3 * n_val
    d5 = 3.0 * (a2 * a2 - 3.0 * a3 * a3) * (at1 * at1 - at2 * at2) * n_val

    n0 = 36.0 * a1 * n_sq * nt_sq
    n1 = -12.0 * a5 * at3 * (at1 + at2) * n_sq + 8.0 * a5 * at3 * (
        6.0 * a1 - a2 - 3.0 * a3
    ) * n_val * nt_val
    n2 = 9.0 * a1 * (at2 * at2 - at1 * at1) * n_sq
    n3 = 12.0 * a5 * at4 * (
        3.0 * (at1 + at2) * n_sq + 2.0 * (a2 + 3.0 * a3) * n_val * nt_val
    )

    nt0 = 12.0 * at3 * n_cube * nt_val
    nt1 = 12.0 * a5 * at3 * at3 * n_sq
    nt2 = 9.0 * a1 * at3 * (at1 - at2) * n_sq + (
        6.0 * a1 * (a2 - 3.0 * a3) - 4.0 * (a2 * a2 - 3.0 * a3 * a3)
    ) * at3 * n_val * nt_val

    gam0 = 36.0 * at4 * n_cube * nt_val
    gam1 = 36.0 * a5 * at3 * at4 * n_sq
    gam2 = -12.0 * at4 * (a2 * a2 - 3.0 * a3 * a3) * n_val * nt_val

    cm3_inv = 1.0 / (cm0 * cm0 * cm0)

    for col in range(n_cols):
        for i in range(1, nlev):
            d_cm = (
                d0
                + d1 * an[col, i]
                + d2 * as_[col, i]
                + d3 * an[col, i] * as_[col, i]
                + d4 * an[col, i] * an[col, i]
                + d5 * as_[col, i] * as_[col, i]
            )
            n_cm = n0 + n1 * an[col, i] + n2 * as_[col, i] + n3 * at[col, i]
            n_cmp = nt0 + nt1 * an[col, i] + nt2 * as_[col, i]
            n_gam = (gam0 + gam1 * an[col, i] + gam2 * as_[col, i]) * at[col, i]

            cmue1[col, i] = cm3_inv * n_cm / d_cm
            cmue2[col, i] = cm3_inv * n_cmp / d_cm
            gam[col, i] = n_gam / d_cm
