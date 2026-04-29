r"""!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Diffusion schemes --- grid centers\label{sec:diffusionMean}
!
! !INTERFACE:
!
! !DESCRIPTION:
! This subroutine solves the one-dimensional diffusion equation
! including source terms,
!  \begin{equation}
!   \label{YdiffCenter}
!    \partder{Y}{t}
!    = \partder{}{z} \left( \nu_Y \partder{Y}{z} \right)
!    - \frac{1}{\tau_R}(Y-Y_{obs})
!    + Y L_{\text{sour}} + Q_{\text{sour}}
!    \comma
!  \end{equation}
! for al variables defined at the centers of the grid cells, and
! a diffusion coefficient $\nu_Y$ defined at the faces.
! Relaxation with time scale $\tau_R$ towards observed values
! $Y_{\text{obs}}$ is possible. $L_{\text{sour}}$ specifies a
! linear source term, and $Q_{\text{sour}}$ a constant source term.
! Central differences are used to discretize the problem
! as discussed in \sect{SectionNumericsMean}. The diffusion term,
! the linear source term, and the linear part arising from the
! relaxation term are treated
! with an implicit method, whereas the constant source term is treated
! fully explicit.
!
! The input parameters {\tt Bcup} and {\tt Bcdw} specify the type
! of the upper and lower boundary conditions, which can be either
! Dirichlet or Neumann-type. {\tt Bcup} and {\tt Bcdw} must have integer
! values corresponding to the parameters {\tt Dirichlet} and {\tt Neumann}
! defined in the module {\tt util}, see \sect{sec:utils}.
! {\tt Yup} and {\tt Ydw} are the values of the boundary conditions at
! the surface and the bottom. Depending on the values of {\tt Bcup} and
! {\tt Bcdw}, they represent either fluxes or prescribed values.
! The integer {\tt posconc} indicates if a quantity is
! non-negative by definition ({\tt posconc}=1, such as for concentrations)
! or not ({\tt posconc}=0). For {\tt posconc}=1 and negative
! boundary fluxes, the source term linearisation according to
! \cite{Patankar80} is applied.
!
! Note that fluxes \emph{entering} a boundary cell are counted positive
! by convention. The lower and upper position for prescribing these fluxes
! are located at the lowest und uppermost grid faces with index "0" and
! index "N", respectively. If values are prescribed, they are located at
! the centers with index "1" and index "N", respectively.
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
!  1: non-negative concentration, 0: else
!   integer, intent(in)                 :: posconc
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
!   REALTYPE, intent(in)                :: nuY(0:N)
!
!  linear source term
!  (treated implicitly)
!   REALTYPE, intent(in)                :: Lsour(0:N)
!
!  constant source term
!  (treated explicitly)
!   REALTYPE, intent(in)                :: Qsour(0:N)
!
!  relaxation time (s)
!   REALTYPE, intent(in)                :: Taur(0:N)
!
!  observed value of Y
!   REALTYPE, intent(in)                :: Yobs(0:N)
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
    "diff_center",
    "diff_center_batch",
]


@numba.njit(cache=True)
def diff_center(
    nlev: int,
    dt: float,
    cnpar: float,
    posconc: int,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    tau_r: np.ndarray,
    y_obs: np.ndarray,
    y: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""! !ROUTINE: Diffusion schemes --- grid centers
!
! !DESCRIPTION:
! This subroutine solves the one-dimensional diffusion equation including
! source terms for all variables defined at the centers of the grid cells.
    """

    for i in range(2, nlev):
        c = 2.0 * dt * nu_y[i] / (h[i] + h[i + 1]) / h[i]
        a = 2.0 * dt * nu_y[i - 1] / (h[i] + h[i - 1]) / h[i]
        linear_source = dt * l_sour[i]

        cu[i] = -cnpar * c
        au[i] = -cnpar * a
        bu[i] = 1.0 + cnpar * (a + c) - linear_source
        du[i] = (1.0 - (1.0 - cnpar) * (a + c)) * y[i]
        du[i] += (1.0 - cnpar) * (a * y[i - 1] + c * y[i + 1])
        du[i] += dt * q_sour[i]

    if bc_up == NEUMANN:
        a = 2.0 * dt * nu_y[nlev - 1] / (h[nlev] + h[nlev - 1]) / h[nlev]
        linear_source = dt * l_sour[nlev]

        au[nlev] = -cnpar * a
        if posconc == 1 and y_up < 0.0:
            # Patankar (1980): move negative flux to the implicit diagonal term.
            # Requires y[nlev] > 0 — same assumption as Fortran diff_center.F90.
            # Division by zero if y goes exactly to zero.
            # Guard: callers must ensure y > 0 when posconc=1 and y_up < 0.
            bu[nlev] = 1.0 - au[nlev] - linear_source - dt * y_up / y[nlev] / h[nlev]
            du[nlev] = y[nlev] + dt * q_sour[nlev]
            du[nlev] += (1.0 - cnpar) * a * (y[nlev - 1] - y[nlev])
        else:
            bu[nlev] = 1.0 - au[nlev] - linear_source
            du[nlev] = y[nlev] + dt * (q_sour[nlev] + y_up / h[nlev])
            du[nlev] += (1.0 - cnpar) * a * (y[nlev - 1] - y[nlev])
    else:
        au[nlev] = 0.0
        bu[nlev] = 1.0
        du[nlev] = y_up

    if bc_down == NEUMANN:
        c = 2.0 * dt * nu_y[1] / (h[1] + h[2]) / h[1]
        linear_source = dt * l_sour[1]

        cu[1] = -cnpar * c
        if posconc == 1 and y_down < 0.0:
            # Patankar (1980): move negative flux to the implicit diagonal term.
            # Requires y[1] > 0 — same assumption as Fortran diff_center.F90.
            # Division by zero if y goes exactly to zero.
            # Guard: callers must ensure y > 0 when posconc=1 and y_down < 0.
            bu[1] = 1.0 - cu[1] - linear_source - dt * y_down / y[1] / h[1]
            du[1] = y[1] + dt * q_sour[1]
            du[1] += (1.0 - cnpar) * c * (y[2] - y[1])
        else:
            bu[1] = 1.0 - cu[1] - linear_source
            du[1] = y[1] + dt * (q_sour[1] + y_down / h[1])
            du[1] += (1.0 - cnpar) * c * (y[2] - y[1])
    else:
        cu[1] = 0.0
        bu[1] = 1.0
        du[1] = y_down

    apply_relaxation = 0
    for i in range(1, nlev + 1):
        if tau_r[i] < 1.0e10:
            apply_relaxation = 1

    if apply_relaxation == 1:
        for i in range(1, nlev + 1):
            bu[i] += dt / tau_r[i]
            du[i] += dt / tau_r[i] * y_obs[i]

    tridiagonal(au, bu, cu, du, ru, qu, y, 1, nlev)


@numba.njit(parallel=True, cache=True)
def diff_center_batch(
    batch_size: int,
    nlev: int,
    dt: float,
    cnpar: float,
    posconc: int,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    tau_r: np.ndarray,
    y_obs: np.ndarray,
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
        diff_center(
            nlev,
            dt,
            cnpar,
            posconc,
            h[b],
            bc_up,
            bc_down,
            y_up,
            y_down,
            nu_y[b],
            l_sour[b],
            q_sour[b],
            tau_r[b],
            y_obs[b],
            y[b],
            au[b],
            bu[b],
            cu[b],
            du[b],
            ru[b],
            qu[b],
        )
