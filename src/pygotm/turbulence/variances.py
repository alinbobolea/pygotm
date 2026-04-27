# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The algebraic velocity variances \label{sec:variances}
!
! !INTERFACE:
   subroutine variances(nlev,SSU,SSV)
!
! !DESCRIPTION:
!
!  Using \eq{bijVertical} and the solution shown in \eq{b13} and
!  the variances of the turbulent velocity fluctations can be
!  evaluated according to
!  \begin{equation}
!    \label{variances}
!    \begin{array}{rcl}
!      \dfrac{\langle u'^2 \rangle}{k}
!      &=& \dfrac23 + \dfrac{1}{\mathcal{N}\eps} \left(
!        \left(\dfrac{a_2}{3}+a_3\right) \nu_t \left( \partder{U}{z} \right)^2
!                          -\dfrac23 a_2 \nu_t \left( \partder{V}{z} \right)^2
!                          -\dfrac43 a_5 G \right)
!      \comma
!      \\[7mm]
!      \dfrac{\langle v'^2 \rangle}{k}
!      &=& \dfrac23
!      +\dfrac{1}{\mathcal{N}\eps} \left(
!        \left(\dfrac{a_2}{3}+a_3\right) \nu_t \left(\partder{V}{z}\right)^2
!                          -\dfrac23 a_2 \nu_t \left(\partder{U}{z}\right)^2
!                          -\dfrac43 a_5 G \right)
!      \comma
!      \\[7mm]
!      \dfrac{\langle w'^2 \rangle}{k}
!      &=& \dfrac23
!      +\dfrac{1}{\mathcal{N}\eps} \left(
!        \left(\dfrac{a_2}{3}-a_3\right) P
!        +\dfrac83 a_5 G
!      \right)
!      \comma
!    \end{array}
!  \end{equation}
!  where the diffusivities are computed according to \eq{nu}
!  (also see \sect{sec:cmueC} and \sect{sec:cmueD}),
!  and the buoyancy production, $G$, follows from \eq{computeG}.
!
! !USES:
  use turbulence,  only:     uu,vv,ww
  use turbulence,  only:     tke,eps,P,B,Px,num
  use turbulence,  only:     cc1,ct1,a2,a3,a5
  IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
! number of vertical layers
  integer,  intent(in)                 :: nlev
!
! square of shear frequency (1/s^2)
! (from u- and v-component)
  REALTYPE, intent(in)                :: SSU(0:nlev),SSV(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
!
   integer                             :: i
   REALTYPE                            :: N,Nt
   REALTYPE                            :: fac1,fac2,fac3,fac4,fac5
!
!
!-----------------------------------------------------------------------
!BOC
!
   N    =   0.5*cc1
   Nt   =   ct1
!
   do i=0,nlev
!
      fac1 = 2./3.
      fac2 = 1.0/( N*eps(i) )
      fac3 = a2/3.0 + a3
      fac4 = a2/3.0 - a3
      fac5 = 2./3.*a2
!
      uu(i) = tke(i)*( fac1 + fac2*( fac3*num(i)*SSU(i)                &
                          - fac5*num(i)*SSV(i) - 4./3.*a5*B(i) ) )
!
      vv(i) = tke(i)*( fac1 + fac2*( fac3*num(i)*SSV(i)                &
                          - fac5*num(i)*SSU(i) - 4./3.*a5*B(i) ) )
!
      ww(i) = tke(i)*( fac1 + fac2*( fac4*(P(i)+Px(i)) + 8./3.*a5*B(i) ) )
!
   enddo
!
   return
   end subroutine variances
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

__all__ = ["VariancesWorkspace", "step_variances"]

_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0
_EIGHT_THIRDS = 8.0 / 3.0


class VariancesWorkspace(TaichiFieldCollection):
    """Taichi fields for the algebraic turbulent-velocity variances."""

    tke: ti.Field
    eps: ti.Field
    P: ti.Field
    B: ti.Field
    Px: ti.Field
    num: ti.Field
    SSU: ti.Field
    SSV: ti.Field
    uu: ti.Field
    vv: ti.Field
    ww: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "eps", "P", "B", "Px", "num", "SSU", "SSV"))
        self.allocate_many(("uu", "vv", "ww"))


@ti_kernel
def step_variances(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    cc1: ti.f64,
    ct1: ti.f64,
    a2: ti.f64,
    a3: ti.f64,
    a5: ti.f64,
    tke: TemplateArg,
    eps: TemplateArg,
    P: TemplateArg,
    B: TemplateArg,
    Px: TemplateArg,
    num: TemplateArg,
    SSU: TemplateArg,
    SSV: TemplateArg,
    uu: TemplateArg,
    vv: TemplateArg,
    ww: TemplateArg,
):
    r"""Update the algebraic velocity variances for one or more columns."""

    n_value = 0.5 * cc1
    _ = ct1
    fac3 = a2 / 3.0 + a3
    fac4 = a2 / 3.0 - a3
    fac5 = _TWO_THIRDS * a2

    for col in range(n_cols):
        for i in range(nlev + 1):
            fac2 = 1.0 / (n_value * eps[col, i])

            uu[col, i] = tke[col, i] * (
                _TWO_THIRDS
                + fac2
                * (
                    fac3 * num[col, i] * SSU[col, i]
                    - fac5 * num[col, i] * SSV[col, i]
                    - _FOUR_THIRDS * a5 * B[col, i]
                )
            )

            vv[col, i] = tke[col, i] * (
                _TWO_THIRDS
                + fac2
                * (
                    fac3 * num[col, i] * SSV[col, i]
                    - fac5 * num[col, i] * SSU[col, i]
                    - _FOUR_THIRDS * a5 * B[col, i]
                )
            )

            ww[col, i] = tke[col, i] * (
                _TWO_THIRDS
                + fac2
                * (
                    fac4 * (P[col, i] + Px[col, i])
                    + _EIGHT_THIRDS * a5 * B[col, i]
                )
            )
