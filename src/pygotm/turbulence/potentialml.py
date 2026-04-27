# ruff: noqa: E501
r"""
!-------------------------------------------------------------------------
!BOP
!
! !ROUTINE: Algebraic length-scale with two master scales \label{sec:potentialml}
!
! !INTERFACE:
   subroutine potentialml(nlev,z0b,z0s,h,depth,NN)
!
! !DESCRIPTION:
!  Computes the length scale by defining two master
!  length scales $l_u$ and $l_d$
!  \begin{equation}
!  \begin{array}{l}
!  \int_{z_0}^{z_0+l_u(z_0)} (b(z_0)-b(z)) dz =k(z_0) \comma \\[4mm]
!  \int_{z_0-l_d(z_0)}^{z_0} (b(z)-b(z_0)) dz =k(z_0)
!  \end{array}
!  \end{equation}
!
!   From $l_u$ and $l_d$ two length--scales are defined: $l_k$,
!   a characteristic mixing length,
!   and $l_\epsilon$, a characteristic dissipation length.
!   They are computed according to
!   \begin{equation}
!   \begin{array}{l}
!   l_k(z_0)= \text{Min} ( l_d(z_0),l_u(z_0)) \comma \\[4mm]
!   l_{\epsilon}(z_0)=\left( l_d(z_0)l_u(z_0)\right)^\frac{1}{2}
!   \point
!   \end{array}
!   \end{equation}
!
!   $l_k$ is used in {\tt kolpran()} to compute eddy viscosity/difussivity.
!   $l_{\epsilon}$ is used to compute the dissipation rate, $\epsilon$
!    according to
!   \begin{equation}
!     \epsilon=C_{\epsilon} k^{3/2} l_{\epsilon}^{-1}
!     \comma
!     C_{\epsilon}=0.7
!    \point
!   \end{equation}
!
! !USES:
   use turbulence, only: L,eps,tke,k_min,eps_min
   use turbulence, only: cde,galp,kappa,length_lim
!
   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
!  number of vertical layers
   integer,  intent(in)                :: nlev
!
!  bottom and surface roughness (m)
   REALTYPE, intent(in)                :: z0b,z0s
!
!  layer thickness (m)
   REALTYPE, intent(in)                :: h(0:nlev)
!
!  local depth (m)
   REALTYPE, intent(in)                :: depth
!
!  buoyancy frequency (1/s^2)
   REALTYPE, intent(in)                :: NN(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s):  Manuel Ruiz Villarreal, Hans Burchard
!
!EOP
!-------------------------------------------------------------------------
!
! !LOCAL VARIABLES:
   integer                   :: i,j
   REALTYPE                  :: ds(0:nlev),db(0:nlev)
   REALTYPE                  :: lu(0:nlev),ld(0:nlev)
   REALTYPE                  :: lk(0:nlev),leps(0:nlev)
   REALTYPE                  :: Lcrit,buoydiff,integral,ceps
   REALTYPE, parameter       :: NNmin=1.e-8
!
!-------------------------------------------------------------------------
!BOC
   db(0)=0.
   ds(nlev)=0.
!
   do i=1,nlev-1
      db(i)=db(i-1)+h(i)      ! distance of intercace i from bottom
      ds(i)=depth-db(i)       ! distance of intercace i from surface
   end do
!
!  Calculation of lu and ld by solving the integral equation following
!  Gaspar (1990). Some other approximations of the integral equation
!  are possible.
!
! Computation of lupward
!
   do i=1,nlev-1
      lu(i)=0.
      integral=0.
      buoydiff=0.
      do j=i+1,nlev
         buoydiff=buoydiff+NN(j-1)*0.5*(h(j)+h(j-1))
         integral=integral+buoydiff*h(j)
         if (integral.ge.tke(i)) then
            if(j.ne.nlev) then
               if(j.ne.i+1) then
                  lu(i)=lu(i)-(integral-tke(i))/buoydiff
               else
!           To avoid lu(i) from becoming too large if NN(i) is too small
               if(NN(i).gt.NNmin) then
                     lu(i)=sqrt(2.)*sqrt(tke(i))/sqrt(NN(i))
                  else
                     lu(i)=h(i)
                  end if
               end if
               goto 600
            end if
         end if
         lu(i)=lu(i)+h(j)
      end do
600   continue
!     Implicitely done in the do loop: if (lu(i).gt.ds(i)) lu(i)=ds(i)
!     lu limited by distance to surface
   end do
!
!  Computation of ldownward
   do i=nlev-1,1,-1
      ld(i)=0.
      integral=0.
      buoydiff=0.
      do j=i-1,1,-1
         buoydiff=buoydiff+NN(j)*0.5*(h(j+1)+h(j))
         integral=integral-buoydiff*h(j)
         if (integral.ge.tke(i)) then
            if(j.ne.0) then
               if(j.ne.i-1) then
                  ld(i)=ld(i)-(integral-tke(i))/buoydiff
               else
!              To avoid ld(i) from becoming too large if NN(i) is too small
                  if(NN(i).gt.NNmin) then
                     ld(i)=sqrt(2.)*sqrt(tke(i))/sqrt(NN(i))
                  else
                     ld(i)=h(i)
                  end if
               end if
               goto 610
            end if
         end if
         ld(i)=ld(i)+h(j)
      end do
610   continue
!     if (ld(i).gt.db(i)) ld(i)=db(i) !ld limited by distance to bottom
   end do
!
!   Calculation of lk and leps, mixing and dissipation lengths
   do i=nlev-1,1,-1
!  Suggested by Gaspar:        lk(i)   = min(lu(i),ld(i))
      lk(i)=sqrt(lu(i)*ld(i))
      leps(i) = sqrt(lu(i)*ld(i))
   end do
!
!  We set L=lk because it is the one we use to calculate num and nuh
   ceps=0.7
   do i=1,nlev-1
      L(i)=lk(i)
   end do
!
! do the boundaries assuming linear log-law length-scale
   L(0)=kappa*z0b
   L(nlev)=kappa*z0s
!
   do i=0,nlev
!
      !  clip the length-scale at the Galperin et al. (1988) value
      !  under stable stratifcitation
      if ((NN(i).gt.0).and.(length_lim)) then
         Lcrit=sqrt(2*galp*galp*tke(i)/NN(i))
         if (L(i).gt.Lcrit) L(i)=Lcrit
      end if
!
!     compute the dissipation rate
      eps(i)=cde*sqrt(tke(i)*tke(i)*tke(i))/L(i)
!
      ! substitute minimum value
      if (eps(i).lt.eps_min) then
        eps(i) = eps_min
          L(i) = cde*sqrt(tke(i)*tke(i)*tke(i))/eps_min
      endif
!
   enddo
!
   return
   end subroutine potentialml
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel

__all__ = ["PotentialMLWorkspace", "step_potentialml"]

_NN_MIN: float = 1.0e-8
_SQRT_TWO: float = 1.4142135623730951


class PotentialMLWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated two-master-scale closure."""

    tke: ti.Field
    eps: ti.Field
    L: ti.Field
    h: ti.Field
    NN: ti.Field
    depth: ti.Field
    z0b: ti.Field
    z0s: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "eps", "L", "h", "NN"))
        self.allocate_many(("depth", "z0b", "z0s"))


@ti_kernel
def step_potentialml(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    kappa: ti.f64,
    cde: ti.f64,
    galp: ti.f64,
    length_lim: ti.i32,
    eps_min: ti.f64,
    tke: TemplateArg,
    eps: TemplateArg,
    L: TemplateArg,
    h: TemplateArg,
    NN: TemplateArg,
    depth: TemplateArg,
    z0b: TemplateArg,
    z0s: TemplateArg,
):
    r"""Update the potential mixed-layer length scale and dissipation."""

    for col in range(n_cols):
        local_z0b = z0b[col, 0]
        local_z0s = z0s[col, 0]

        for i in range(1, nlev):
            lu = 0.0
            integral = 0.0
            buoydiff = 0.0
            found = 0

            for j in range(i + 1, nlev + 1):
                if found == 0:
                    buoydiff = buoydiff + NN[col, j - 1] * 0.5 * (
                        h[col, j] + h[col, j - 1]
                    )
                    integral = integral + buoydiff * h[col, j]
                    if integral >= tke[col, i]:
                        if j != nlev:
                            if j != i + 1:
                                lu = lu - (integral - tke[col, i]) / buoydiff
                            else:
                                if NN[col, i] > _NN_MIN:
                                    lu = (
                                        _SQRT_TWO
                                        * ti.sqrt(tke[col, i])
                                        / ti.sqrt(NN[col, i])
                                    )
                                else:
                                    lu = h[col, i]
                            found = 1
                    if found == 0:
                        lu = lu + h[col, j]

            ld = 0.0
            integral = 0.0
            buoydiff = 0.0
            found = 0

            for offset in range(i):
                j = i - 1 - offset
                if j >= 1 and found == 0:
                    buoydiff = buoydiff + NN[col, j] * 0.5 * (
                        h[col, j + 1] + h[col, j]
                    )
                    integral = integral - buoydiff * h[col, j]
                    if integral >= tke[col, i]:
                        if j != i - 1:
                            ld = ld - (integral - tke[col, i]) / buoydiff
                        else:
                            if NN[col, i] > _NN_MIN:
                                ld = (
                                    _SQRT_TWO
                                    * ti.sqrt(tke[col, i])
                                    / ti.sqrt(NN[col, i])
                                )
                            else:
                                ld = h[col, i]
                        found = 1
                    if found == 0:
                        ld = ld + h[col, j]

            L[col, i] = ti.sqrt(lu * ld)

        L[col, 0] = kappa * local_z0b
        L[col, nlev] = kappa * local_z0s

        for i in range(nlev + 1):
            if NN[col, i] > 0.0 and length_lim != 0:
                lcrit = ti.sqrt(2.0 * galp * galp * tke[col, i] / NN[col, i])
                if L[col, i] > lcrit:
                    L[col, i] = lcrit

            tke32 = ti.sqrt(tke[col, i] * tke[col, i] * tke[col, i])
            eps[col, i] = cde * tke32 / L[col, i]

            if eps[col, i] < eps_min:
                eps[col, i] = eps_min
                L[col, i] = cde * tke32 / eps_min
