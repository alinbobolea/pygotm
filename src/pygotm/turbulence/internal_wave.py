# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Update internal wave mixing\label{sec:internalWaves}
!
! !INTERFACE:
   subroutine internal_wave(nlev,NN,SS)
!
! !DESCRIPTION:
!  Imposes eddy viscosity and diffusivity characteristic
!  of internal wave activity and shear instability when there is extinction
!  of turbulence as suggested by \cite{KanthaClayson94}.
!  In this case, the new values of $\nu_t$ and $\nu'_t=\nu^B_t$,
!  defined in \eq{fluxes},
!  are used instead of those computed with the model.
!
!  When k is small (extinction of turbulence, diagnosed by
!  $k<${\tt klimiw}),
!  $\nu_t$ and $\nu'_t$ are set to empirical values typical
!  in the presence of internal wave activity (IW) and shear
!  instability (SI). This model is described by
!  \begin{equation}
!    \nu_t = \nu_t^{IW}  + \nu_t^{SI}      \comma
!    \nu'_t= \nu'^{IW}_t + \nu'^{SI}_t     \comma
!  \end{equation}
!  where
!  \begin{equation}
!    \nu_t^{IW}  =         10^{-4}          \comma
!    \nu'^{IW}_t = 5 \cdot 10^{-5}          \point
!  \end{equation}
!  The `SI' parts are functions of the Richardson number according to
!  \begin{eqnarray}
!  \nu_t^{SI} = \nu'^{SI}_t = 0              \comma
!     & R_i>0.7 \comma \\[4mm]
!  \nu_t^{SI} = \nu'^{SI}_t = 5 \cdot 10^{-3} \left( 1-\left(\frac {R_i}
!  {0.7}\right)^2\right)^3                    \comma
!     & 0<R_i<0.7 \comma \\[4mm]
!  \nu_t^{SI} = \nu'^{SI}_t = 5 \cdot 10^{-3} \comma
!     & R_i < 0
!     \point
!  \end{eqnarray}
!  The unit of all diffusivities is m$^2$s$^{-1}$.
!
! !USES:
   use turbulence,    only:            iw_model,alpha,klimiw,rich_cr
   use turbulence,    only:            numiw,nuhiw,numshear
   use turbulence,    only:            tke,num,nuh
   IMPLICIT NONE
!
! !INPUT PARAMETERS:
   integer,  intent(in)                :: nlev
   REALTYPE, intent(in)                :: NN(0:nlev),SS(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding, Hans Burchard,
!                      Manuel Ruiz Villarreal
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
   REALTYPE              :: rich(0:nlev)
   REALTYPE              :: rich2,pot,x
   integer               :: i
!
!-----------------------------------------------------------------------
!BOC
   if (iw_model.eq.2) then
      rich2 = rich_cr*rich_cr
      do i=1,nlev-1
         if (tke(i).le.klimiw) then
            rich(i)=NN(i)/(SS(i)+1.e-10)
            if (rich(i).lt.rich_cr) then
               if (rich(i).gt.0) then
                  pot=1-rich(i)*rich(i)/rich2
                  x=numshear*pot*pot*pot
                  num(i)=numiw+x
                  nuh(i)=nuhiw+x
               else
                  num(i)=numiw+numshear
                  nuh(i)=nuhiw+numshear
               end if
            else
               num(i)=numiw
               nuh(i)=nuhiw
            end if
         end if
      end do
   end if
!
   return
   end subroutine internal_wave
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

__all__ = ["InternalWaveWorkspace", "step_internal_wave"]

_SHEAR_EPSILON = 1.0e-10


class InternalWaveWorkspace(TaichiFieldCollection):
    """Taichi fields for the internal-wave diffusivity adjustment."""

    tke: ti.Field
    num: ti.Field
    nuh: ti.Field
    NN: ti.Field
    SS: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "num", "nuh", "NN", "SS"))


@ti_kernel
def step_internal_wave(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    iw_model: ti.i32,
    klimiw: ti.f64,
    rich_cr: ti.f64,
    numiw: ti.f64,
    nuhiw: ti.f64,
    numshear: ti.f64,
    tke: TemplateArg,
    num: TemplateArg,
    nuh: TemplateArg,
    NN: TemplateArg,
    SS: TemplateArg,
):
    r"""Apply the Kantha-Clayson internal-wave mixing fallback."""

    if iw_model == 2:
        rich2 = rich_cr * rich_cr

        for col in range(n_cols):
            for i in range(1, nlev):
                if tke[col, i] <= klimiw:
                    rich = NN[col, i] / (SS[col, i] + _SHEAR_EPSILON)
                    if rich < rich_cr:
                        if rich > 0.0:
                            pot = 1.0 - rich * rich / rich2
                            x = numshear * pot * pot * pot
                            num[col, i] = numiw + x
                            nuh[col, i] = nuhiw + x
                        else:
                            num[col, i] = numiw + numshear
                            nuh[col, i] = nuhiw + numshear
                    else:
                        num[col, i] = numiw
                        nuh[col, i] = nuhiw
