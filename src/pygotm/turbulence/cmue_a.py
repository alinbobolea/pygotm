# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The non-local, exact weak-equilibrium stability function \label{sec:cmueA}
!
! !INTERFACE:
!   subroutine cmue_a(nlev)
!
! !DESCRIPTION:
!
!  The solution of \eq{bijVertical} and \eq{giVertical} has the shape indicated
!  by \eq{b13}. This subroutine is used to update the quantities
!  $c_\mu$, $c'_\mu$ and $\Gamma$, defined in \eq{b13}, from which all turbulent
!  fluxes can be computed. The non-linear terms ${\cal N}$ and ${\cal N}_b$ are updated
!  by evaluating the right hand side of \eq{NandNb} at the old time step.
!
!  The numerators and the denominator appearing in \eq{cm}
!  are polynomials of the form
!  \begin{equation}
!   \label{vdng}
!    \begin{array}{rcl}
!    D         &=& d_0
!               +  d_1 \overline{N}^2  + d_2 \overline{S}^2
!               +  d_3 \overline{N}^2 \overline{S}^2
!               + d_4 \overline{N}^4   + d_5 \overline{S}^4      \comma \\[3mm]
!    N_n        &=& n_0
!               +  n_1 \overline{N}^2  + n_2 \overline{S}^2
!               +  n_3 \overline{T}                              \comma \\[3mm]
!    N_b       &=& n_{b0}
!               +  n_{b1} \overline{N}^2 + n_{b2} \overline{S}^2 \comma \\[3mm]
!    N_\Gamma  &=& ( g_0
!               +  g_1 \overline{N}^2  + g_2 \overline{S}^2 ) \overline{T}
!   \point
!   \end{array}
!  \end{equation}
!
! !USES:
!   use turbulence, only: eps
!   use turbulence, only: P,B,Px,Pb,epsb
!   use turbulence, only: an,as,at,r
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
!-----------------------------------------------------------------------!
! !LOCAL VARIABLES:
!     integer                 ::   i
!     REALTYPE                ::   N,Nt,Pe,Pbeb
!     REALTYPE                ::   xd0,xd1,xd2,xd3,xd4,xd5,xd6,xd7
!     REALTYPE                ::   xn0,xn1,xn2,xn3,xn4,xn5
!     REALTYPE                ::   xt0,xt1,xt2,xt3
!     REALTYPE                ::   xg0,xg1,xg2
!     REALTYPE                ::   d0,d1,d2,d3,d4,d5
!     REALTYPE                ::   n0,n1,n2,n3,nt0,nt1,nt2
!     REALTYPE                ::   gam0,gam1,gam2
!     REALTYPE                ::   nGam,dCm,nCm,nCmp
!     REALTYPE                ::   cm3_inv,r_i
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
    "CmueAWorkspace",
    "step_cmue_a",
]


class CmueAWorkspace(TaichiFieldCollection):
    """Taichi fields for the non-local exact weak-equilibrium stability closure."""

    eps: ti.Field
    P: ti.Field
    B: ti.Field
    Px: ti.Field
    Pb: ti.Field
    epsb: ti.Field
    as_: ti.Field
    an: ti.Field
    at: ti.Field
    r: ti.Field
    cmue1: ti.Field
    cmue2: ti.Field
    gam: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("eps", "P", "B", "Px", "Pb", "epsb"))
        self.allocate_many(("as_", "an", "at", "r", "cmue1", "cmue2", "gam"))


@ti_kernel
def step_cmue_a(  # type: ignore[no-untyped-def]
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
    eps: TemplateArg,
    P: TemplateArg,
    B: TemplateArg,
    Px: TemplateArg,
    Pb: TemplateArg,
    epsb: TemplateArg,
    as_: TemplateArg,
    an: TemplateArg,
    at: TemplateArg,
    r: TemplateArg,
    cmue1: TemplateArg,
    cmue2: TemplateArg,
    gam: TemplateArg,
):
    r"""Update the exact weak-equilibrium stability functions."""

    cm3_inv = 1.0 / (cm0 * cm0 * cm0)

    xd0 = 36.0
    xd1 = 84.0 * a5 * at3
    xd2 = 9.0 * (at2 * at2 - at1 * at1)
    xd3 = -12.0 * (a2 * a2 - 3.0 * a3 * a3)
    xd4 = 12.0 * a5 * at3 * (a2 * at1 - 3.0 * a3 * at2)
    xd5 = 12.0 * a5 * at3 * (a3 * a3 - a2 * a2)
    xd6 = 48.0 * a5 * a5 * at3 * at3
    xd7 = 3.0 * (a2 * a2 - 3.0 * a3 * a3) * (at1 * at1 - at2 * at2)

    xn0 = 36.0 * a1
    xn1 = -12.0 * a5 * at3 * (at1 + at2)
    xn2 = 8.0 * a5 * at3 * (6.0 * a1 - a2 - 3.0 * a3)
    xn3 = 9.0 * a1 * (at2 * at2 - at1 * at1)
    xn4 = 36.0 * a5 * at4 * (at1 + at2)
    xn5 = 24.0 * a5 * at4 * (a2 + 3.0 * a3)

    xt0 = 12.0 * at3
    xt1 = 12.0 * a5 * at3 * at3
    xt2 = 9.0 * a1 * at3 * (at1 - at2)
    xt3 = (6.0 * a1 * (a2 - 3.0 * a3) - 4.0 * (a2 * a2 - 3.0 * a3 * a3)) * at3

    xg0 = 36.0 * at4
    xg1 = 36.0 * a5 * at3 * at4
    xg2 = -12.0 * at4 * (a2 * a2 - 3.0 * a3 * a3)

    for col in range(n_cols):
        for i in range(1, nlev):
            pe = (P[col, i] + Px[col, i] + B[col, i]) / eps[col, i]
            pbeb = Pb[col, i] / epsb[col, i]
            r_i = 1.0 / r[col, i]

            n_val = pe + 0.5 * cc1 - 1.0
            nt_val = 0.5 * (pe - 1.0) + ct1 + 0.5 * r_i * (pbeb - 1.0)
            nt_val = (pe - 1.0) + ct1

            n_sq = n_val * n_val
            n_cube = n_sq * n_val
            nt_sq = nt_val * nt_val

            d0 = xd0 * n_cube * nt_sq
            d1 = xd1 * n_sq * nt_val
            d2 = xd2 * n_cube + xd3 * n_val * nt_sq
            d3 = xd4 * n_val + xd5 * nt_val
            d4 = xd6 * n_val
            d5 = xd7 * n_val

            n0 = xn0 * n_sq * nt_sq
            n1 = xn1 * n_sq + xn2 * n_val * nt_val
            n2 = xn3 * n_sq
            n3 = xn4 * n_sq + xn5 * n_val * nt_val

            nt0 = xt0 * n_cube * nt_val
            nt1 = xt1 * n_sq
            nt2 = xt2 * n_sq + xt3 * n_val * nt_val

            gam0 = xg0 * n_cube * nt_val
            gam1 = xg1 * n_sq
            gam2 = xg2 * n_val * nt_val

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
