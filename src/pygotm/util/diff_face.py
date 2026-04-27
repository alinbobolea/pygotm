r"""!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Diffusion schemes --- grid faces\label{sec:diffusionFace}
!
! !INTERFACE:
!
! !DESCRIPTION:
!
! !USES:
!
! !INPUT PARAMETERS:
!
!  number of vertical layers
!   integer,  intent(in)                :: N
!
!  time step (s)
!   REALTYPE, intent(in)                :: dt
!
!  "implicitness" parameter
!   REALTYPE, intent(in)                :: cnpar
!
!  layer thickness (m)
!   REALTYPE, intent(in)                :: h(0:N)
!
!  type of upper BC
!   integer,  intent(in)                :: Bcup
!
!  type of lower BC
!   integer,  intent(in)                :: Bcdw
!
!  value of upper BC
!   REALTYPE, intent(in)                :: Yup
!
!  value of lower BC
!   REALTYPE, intent(in)                :: Ydw
!
!  diffusivity of Y
! !   REALTYPE, intent(in)                :: nuY(0:N)
!   REALTYPE                            :: nuY(0:N) ! Bug fix Georg Umgiesser
!
!  linear source term
!  (treated implicitly)
!   REALTYPE, intent(in)                :: Lsour(0:N)
!
!  constant source term
!  (treated explicitly)
!   REALTYPE, intent(in)                :: Qsour(0:N)
!
!
! !INPUT/OUTPUT PARAMETERS:
!   REALTYPE, intent(inout)             :: Y(0:N)
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                   :: i
!   REALTYPE                  :: a,c,l
!
!-----------------------------------------------------------------------
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.util.tridiagonal import tridiagonal, tridiagonal_column
from pygotm.util.util import Dirichlet as DIRICHLET
from pygotm.util.util import Neumann as NEUMANN

__all__ = [
    "DIRICHLET",
    "NEUMANN",
    "diff_face",
    "diff_face_column",
]



@ti.func
def diff_face(  # type: ignore[no-untyped-def]
    nlev,
    dt,
    cnpar,
    h,
    bc_up,
    bc_down,
    y_up,
    y_down,
    nu_y,
    l_sour,
    q_sour,
    y,
    au,
    bu,
    cu,
    du,
    ru,
    qu,
):
    r"""! !ROUTINE: Diffusion schemes --- grid faces"""

    if nlev == 2:
        nu_y[0] = nu_y[1]
        nu_y[nlev] = nu_y[1]
        y[0] = y[1]
        y[nlev] = y[1]

    for i in range(2, nlev - 1):
        c = dt * (nu_y[i + 1] + nu_y[i]) / (h[i] + h[i + 1]) / h[i + 1]
        a = dt * (nu_y[i] + nu_y[i - 1]) / (h[i] + h[i + 1]) / h[i]
        linear_source = dt * l_sour[i]

        cu[i] = -cnpar * c
        au[i] = -cnpar * a
        bu[i] = 1.0 + cnpar * (a + c) - linear_source
        du[i] = (1.0 - (1.0 - cnpar) * (a + c)) * y[i]
        du[i] += (1.0 - cnpar) * (a * y[i - 1] + c * y[i + 1])
        du[i] += dt * q_sour[i]

    if bc_up == NEUMANN:
        a = dt * (nu_y[nlev - 1] + nu_y[nlev - 2]) / (h[nlev - 1] + h[nlev])
        a /= h[nlev - 1]
        linear_source = dt * l_sour[nlev - 1]

        au[nlev - 1] = -cnpar * a
        bu[nlev - 1] = 1.0 + cnpar * a - linear_source
        du[nlev - 1] = (1.0 - (1.0 - cnpar) * a) * y[nlev - 1]
        du[nlev - 1] += (1.0 - cnpar) * a * y[nlev - 2]
        du[nlev - 1] += dt * q_sour[nlev - 1]
        du[nlev - 1] += 2.0 * dt * y_up / (h[nlev - 1] + h[nlev])
    else:
        au[nlev - 1] = 0.0
        bu[nlev - 1] = 1.0
        du[nlev - 1] = y_up

    if bc_down == NEUMANN:
        c = dt * (nu_y[2] + nu_y[1]) / (h[1] + h[2]) / h[2]
        linear_source = dt * l_sour[1]

        cu[1] = -cnpar * c
        bu[1] = 1.0 + cnpar * c - linear_source
        du[1] = (1.0 - (1.0 - cnpar) * c) * y[1]
        du[1] += (1.0 - cnpar) * c * y[2]
        du[1] += dt * q_sour[1]
        du[1] += 2.0 * dt * y_down / (h[1] + h[2])
    else:
        bu[1] = 1.0
        cu[1] = 0.0
        du[1] = y_down

    tridiagonal(au, bu, cu, du, ru, qu, y, 1, nlev - 1)


@ti.func
def diff_face_column(  # type: ignore[no-untyped-def]
    col,
    nlev,
    dt,
    cnpar,
    h,
    bc_up,
    bc_down,
    y_up,
    y_down,
    nu_y,
    l_sour,
    q_sour,
    y,
    au,
    bu,
    cu,
    du,
    ru,
    qu,
):
    r"""! !ROUTINE: Diffusion schemes --- grid faces"""

    if nlev == 2:
        nu_y[col, 0] = nu_y[col, 1]
        nu_y[col, nlev] = nu_y[col, 1]
        y[col, 0] = y[col, 1]
        y[col, nlev] = y[col, 1]

    for i in range(2, nlev - 1):
        c = dt * (nu_y[col, i + 1] + nu_y[col, i]) / (h[col, i] + h[col, i + 1])
        c /= h[col, i + 1]
        a = dt * (nu_y[col, i] + nu_y[col, i - 1]) / (h[col, i] + h[col, i + 1])
        a /= h[col, i]
        linear_source = dt * l_sour[col, i]

        cu[col, i] = -cnpar * c
        au[col, i] = -cnpar * a
        bu[col, i] = 1.0 + cnpar * (a + c) - linear_source
        du[col, i] = (1.0 - (1.0 - cnpar) * (a + c)) * y[col, i]
        du[col, i] += (1.0 - cnpar) * (a * y[col, i - 1] + c * y[col, i + 1])
        du[col, i] += dt * q_sour[col, i]

    if bc_up == NEUMANN:
        a = (
            dt
            * (nu_y[col, nlev - 1] + nu_y[col, nlev - 2])
            / (h[col, nlev - 1] + h[col, nlev])
            / h[col, nlev - 1]
        )
        linear_source = dt * l_sour[col, nlev - 1]

        au[col, nlev - 1] = -cnpar * a
        bu[col, nlev - 1] = 1.0 + cnpar * a - linear_source
        du[col, nlev - 1] = (1.0 - (1.0 - cnpar) * a) * y[col, nlev - 1]
        du[col, nlev - 1] += (1.0 - cnpar) * a * y[col, nlev - 2]
        du[col, nlev - 1] += dt * q_sour[col, nlev - 1]
        du[col, nlev - 1] += 2.0 * dt * y_up / (h[col, nlev - 1] + h[col, nlev])
    else:
        au[col, nlev - 1] = 0.0
        bu[col, nlev - 1] = 1.0
        du[col, nlev - 1] = y_up

    if bc_down == NEUMANN:
        c = dt * (nu_y[col, 2] + nu_y[col, 1]) / (h[col, 1] + h[col, 2])
        c /= h[col, 2]
        linear_source = dt * l_sour[col, 1]

        cu[col, 1] = -cnpar * c
        bu[col, 1] = 1.0 + cnpar * c - linear_source
        du[col, 1] = (1.0 - (1.0 - cnpar) * c) * y[col, 1]
        du[col, 1] += (1.0 - cnpar) * c * y[col, 2]
        du[col, 1] += dt * q_sour[col, 1]
        du[col, 1] += 2.0 * dt * y_down / (h[col, 1] + h[col, 2])
    else:
        bu[col, 1] = 1.0
        cu[col, 1] = 0.0
        du[col, 1] = y_down

    tridiagonal_column(col, au, bu, cu, du, ru, qu, y, 1, nlev - 1)
