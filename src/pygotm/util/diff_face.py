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

import numba
import numpy as np

from pygotm.util.tridiagonal import tridiagonal
from pygotm.util.util import Dirichlet as DIRICHLET
from pygotm.util.util import Neumann as NEUMANN

__all__ = [
    "DIRICHLET",
    "NEUMANN",
    "diff_face",
    "diff_face_batch",
]


@numba.njit(cache=True)
def diff_face(
    nlev: int,
    dt: float,
    cnpar: float,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    y: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""! !ROUTINE: Diffusion schemes --- grid faces"""

    # Bug fix Georg Umgiesser: set boundary nu and y values for nlev==2
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


@numba.njit(parallel=True, cache=True)
def diff_face_batch(
    batch_size: int,
    nlev: int,
    dt: float,
    cnpar: float,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    y: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    """Batch variant: process batch_size columns in parallel with numba.prange."""
    for b in numba.prange(batch_size):
        diff_face(
            nlev,
            dt,
            cnpar,
            h[b],
            bc_up,
            bc_down,
            y_up,
            y_down,
            nu_y[b],
            l_sour[b],
            q_sour[b],
            y[b],
            au[b],
            bu[b],
            cu[b],
            du[b],
            ru[b],
            qu[b],
        )
