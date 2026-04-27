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

import taichi as ti

from pygotm.util.tridiagonal import tridiagonal, tridiagonal_column
from pygotm.util.util import Dirichlet as DIRICHLET
from pygotm.util.util import Neumann as NEUMANN

__all__ = [
    "DIRICHLET",
    "NEUMANN",
    "diff_center",
    "diff_center_column",
]


@ti.func
def diff_center(  # type: ignore[no-untyped-def]
    nlev,
    dt,
    cnpar,
    posconc,
    h,
    bc_up,
    bc_down,
    y_up,
    y_down,
    nu_y,
    l_sour,
    q_sour,
    tau_r,
    y_obs,
    y,
    au,
    bu,
    cu,
    du,
    ru,
    qu,
):
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


@ti.func
def diff_center_column(  # type: ignore[no-untyped-def]
    col,
    nlev,
    dt,
    cnpar,
    posconc,
    h,
    bc_up,
    bc_down,
    y_up,
    y_down,
    nu_y,
    l_sour,
    q_sour,
    tau_r,
    y_obs,
    y,
    au,
    bu,
    cu,
    du,
    ru,
    qu,
):
    r"""! !ROUTINE: Diffusion schemes --- grid centers
!
! !DESCRIPTION:
! This subroutine solves the one-dimensional diffusion equation including
! source terms for all variables defined at the centers of the grid cells.
    """

    for i in range(2, nlev):
        c = 2.0 * dt * nu_y[col, i] / (h[col, i] + h[col, i + 1]) / h[col, i]
        a = 2.0 * dt * nu_y[col, i - 1] / (h[col, i] + h[col, i - 1]) / h[col, i]
        linear_source = dt * l_sour[col, i]

        cu[col, i] = -cnpar * c
        au[col, i] = -cnpar * a
        bu[col, i] = 1.0 + cnpar * (a + c) - linear_source
        du[col, i] = (1.0 - (1.0 - cnpar) * (a + c)) * y[col, i]
        du[col, i] += (1.0 - cnpar) * (a * y[col, i - 1] + c * y[col, i + 1])
        du[col, i] += dt * q_sour[col, i]

    if bc_up == NEUMANN:
        a = (
            2.0
            * dt
            * nu_y[col, nlev - 1]
            / (h[col, nlev] + h[col, nlev - 1])
            / h[col, nlev]
        )
        linear_source = dt * l_sour[col, nlev]

        au[col, nlev] = -cnpar * a
        if posconc == 1 and y_up < 0.0:
            # Patankar (1980): move negative flux to the implicit diagonal term.
            # Requires y[col, nlev] > 0 — same assumption as Fortran diff_center.F90.
            # Division by zero if y goes exactly to zero.
            # Guard: callers must ensure y > 0 when posconc=1 and y_up < 0.
            bu[col, nlev] = (
                1.0
                - au[col, nlev]
                - linear_source
                - dt * y_up / y[col, nlev] / h[col, nlev]
            )
            du[col, nlev] = y[col, nlev] + dt * q_sour[col, nlev]
            du[col, nlev] += (1.0 - cnpar) * a * (y[col, nlev - 1] - y[col, nlev])
        else:
            bu[col, nlev] = 1.0 - au[col, nlev] - linear_source
            du[col, nlev] = y[col, nlev] + dt * (
                q_sour[col, nlev] + y_up / h[col, nlev]
            )
            du[col, nlev] += (1.0 - cnpar) * a * (y[col, nlev - 1] - y[col, nlev])
    else:
        au[col, nlev] = 0.0
        bu[col, nlev] = 1.0
        du[col, nlev] = y_up

    if bc_down == NEUMANN:
        c = 2.0 * dt * nu_y[col, 1] / (h[col, 1] + h[col, 2]) / h[col, 1]
        linear_source = dt * l_sour[col, 1]

        cu[col, 1] = -cnpar * c
        if posconc == 1 and y_down < 0.0:
            # Patankar (1980): move negative flux to the implicit diagonal term.
            # Requires y[col, 1] > 0 — same assumption as Fortran diff_center.F90.
            # Division by zero if y goes exactly to zero.
            # Guard: callers must ensure y > 0 when posconc=1 and y_down < 0.
            bu[col, 1] = (
                1.0
                - cu[col, 1]
                - linear_source
                - dt * y_down / y[col, 1] / h[col, 1]
            )
            du[col, 1] = y[col, 1] + dt * q_sour[col, 1]
            du[col, 1] += (1.0 - cnpar) * c * (y[col, 2] - y[col, 1])
        else:
            bu[col, 1] = 1.0 - cu[col, 1] - linear_source
            du[col, 1] = y[col, 1] + dt * (q_sour[col, 1] + y_down / h[col, 1])
            du[col, 1] += (1.0 - cnpar) * c * (y[col, 2] - y[col, 1])
    else:
        cu[col, 1] = 0.0
        bu[col, 1] = 1.0
        du[col, 1] = y_down

    apply_relaxation = 0
    for i in range(1, nlev + 1):
        if tau_r[col, i] < 1.0e10:
            apply_relaxation = 1

    if apply_relaxation == 1:
        for i in range(1, nlev + 1):
            bu[col, i] += dt / tau_r[col, i]
            du[col, i] += dt / tau_r[col, i] * y_obs[col, i]

    tridiagonal_column(col, au, bu, cu, du, ru, qu, y, 1, nlev)
