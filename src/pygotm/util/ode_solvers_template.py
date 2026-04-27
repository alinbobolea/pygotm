r"""!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: General ODE solver (template version) \label{sec:ode-solver}
!
! !DESCRIPTION:
! This is the template version of the ODE solver (ode_solvers_template.F90).
! In Fortran, this file was designed to be #include'd with preprocessor macros
! (_NAME_, _LOWI_, _ODE_ZEROD_, _SIZE_, _INCC_, _INPP_, _ODE_LOOP_BEGIN_,
! etc.) that parametrize it for zero-dimensional or one-dimensional spatial
! systems and for different array indexing conventions.
!
! The template version differs from ode_solvers.F90 in two solvers:
!
!  runge_kutta_2: uses the explicit midpoint method (not Heun's trapezoidal)
!    c^{(mid)} = c^n + dt/2 * f(c^n)
!    c^{n+1}   = c^n + dt   * f(c^{(mid)})
!
!  runge_kutta_4: uses standard RK4 with half-step intermediates
!    k1 = f(c^n)
!    k2 = f(c^n + dt/2 * k1)
!    k3 = f(c^n + dt/2 * k2)
!    k4 = f(c^n + dt   * k3)
!    c^{n+1} = c^n + dt/3 * (k1/2 + k2 + k3 + k4/2)
!
! All other solvers (Euler, Patankar, Modified Patankar, EMP) are identical
! to ode_solvers.F90 and are re-exported directly.
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard, Karsten Bolding
!
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

# Re-export all shared solvers from ode_solvers verbatim.
from pygotm.util.ode_solvers import (
    PpddCallback,
    RhsCallback,
    emp_1,
    emp_2,
    euler_forward,
    findp_bisection,
    matrix_solve,
    modified_patankar,
    modified_patankar_2,
    modified_patankar_4,
    patankar,
    patankar_runge_kutta_2,
    patankar_runge_kutta_4,
)

__all__ = [
    "ode_solver",
    "euler_forward",
    "runge_kutta_2",
    "runge_kutta_4",
    "patankar",
    "patankar_runge_kutta_2",
    "patankar_runge_kutta_4",
    "modified_patankar",
    "modified_patankar_2",
    "modified_patankar_4",
    "emp_1",
    "emp_2",
    "findp_bisection",
    "matrix_solve",
]


def runge_kutta_2(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_rhs: RhsCallback,
) -> None:
    """Second-order Runge-Kutta — explicit midpoint method (template version).

    !DESCRIPTION:
    ! Here, the second-order Runge-Kutta (RK2) scheme is coded, with two
    ! evaluations of the right hand side per time step (midpoint method):
    !   c^{(mid)} = c^n + dt/2 * f(c^n)
    !   c^{n+1}   = c^n + dt   * f(c^{(mid)})
    !
    ! This is the template (ode_solvers_template.F90) variant.  The
    ! non-template ode_solvers.F90 uses Heun's trapezoidal method instead.
    !
    ! !REVISION HISTORY:
    !  Original author(s): Hans Burchard, Karsten Bolding

    Updates cc in-place. Level 0 (index 0) is never modified.
    """
    rhs = get_rhs(True, numc, nlev, cc)

    cc1 = cc.copy()
    cc1[:, 1:] += dt / 2.0 * rhs[:, 1:]

    rhs_mid = get_rhs(False, numc, nlev, cc1)
    cc[:, 1:] += dt * rhs_mid[:, 1:]


def runge_kutta_4(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_rhs: RhsCallback,
) -> None:
    """Fourth-order Runge-Kutta — standard half-step variant (template version).

    !DESCRIPTION:
    ! Here, the fourth-order Runge-Kutta (RK4) scheme is coded,
    ! with four evaluations of the right hand sides per time step
    ! using the standard half-step intermediate evaluations:
    !   k1 = f(c^n)
    !   k2 = f(c^n + dt/2 * k1)
    !   k3 = f(c^n + dt/2 * k2)
    !   k4 = f(c^n + dt   * k3)
    !   c^{n+1} = c^n + dt/3 * (k1/2 + k2 + k3 + k4/2)
    !           = c^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)   [standard form]
    !
    ! This is the template (ode_solvers_template.F90) variant, which achieves
    ! true 4th-order accuracy.  The non-template ode_solvers.F90 uses full-dt
    ! intermediate steps and does not achieve 4th-order accuracy.
    !
    ! !REVISION HISTORY:
    !  Original author(s): Hans Burchard, Karsten Bolding

    Updates cc in-place. Level 0 (index 0) is never modified.
    """
    k1 = get_rhs(True, numc, nlev, cc)

    # Accumulate weighted sum: rhs_sum = k1/2
    rhs_sum = k1.copy()
    rhs_sum[:, 1:] *= 0.5

    cc1 = cc.copy()
    cc1[:, 1:] += dt / 2.0 * k1[:, 1:]
    k2 = get_rhs(False, numc, nlev, cc1)
    rhs_sum[:, 1:] += k2[:, 1:]  # rhs_sum = k1/2 + k2

    cc1[:] = cc[:]
    cc1[:, 1:] += dt / 2.0 * k2[:, 1:]
    k3 = get_rhs(False, numc, nlev, cc1)
    rhs_sum[:, 1:] += k3[:, 1:]  # rhs_sum = k1/2 + k2 + k3

    cc1[:] = cc[:]
    cc1[:, 1:] += dt * k3[:, 1:]
    k4 = get_rhs(False, numc, nlev, cc1)
    rhs_sum[:, 1:] += 0.5 * k4[:, 1:]  # rhs_sum = k1/2 + k2 + k3 + k4/2

    cc[:, 1:] += dt / 3.0 * rhs_sum[:, 1:]


def ode_solver(
    solver: int,
    numc: int,
    nlev: int,
    dt: float,
    cc: np.ndarray,
    get_rhs: RhsCallback | None = None,
    get_ppdd: PpddCallback | None = None,
) -> None:
    """Dispatch to one of 11 ODE solvers (template version).

    !DESCRIPTION:
    ! Template version of the general ODE solver dispatcher.  Identical to
    ! ode_solvers.ode_solver except solvers 2 (RK2) and 3 (RK4) use the
    ! template-version algorithms (midpoint RK2, standard half-step RK4).
    !
    ! Solver IDs:
    !   1 = Euler-forward (E1)
    !   2 = Runge-Kutta 2 (midpoint — template variant)
    !   3 = Runge-Kutta 4 (standard half-step — template variant)
    !   4-11 = same as ode_solvers.ode_solver
    !
    ! !REVISION HISTORY:
    !  Original author(s): Hans Burchard, Karsten Bolding

    Parameters
    ----------
    solver : int
        Solver identifier (1–11).
    numc : int
        Number of biogeochemical state variables.
    nlev : int
        Number of vertical levels.
    dt : float
        Time step [s].
    cc : np.ndarray, shape (numc, nlev+1)
        State variable array, modified in-place. Level 0 is not updated.
    get_rhs : callable, optional
        Required for solvers 1, 2, 3, 10, 11.
    get_ppdd : callable, optional
        Required for solvers 4–9.
    """
    if solver == 1:
        euler_forward(dt, numc, nlev, cc, get_rhs)  # type: ignore[arg-type]
    elif solver == 2:
        runge_kutta_2(dt, numc, nlev, cc, get_rhs)  # type: ignore[arg-type]
    elif solver == 3:
        runge_kutta_4(dt, numc, nlev, cc, get_rhs)  # type: ignore[arg-type]
    elif solver == 4:
        patankar(dt, numc, nlev, cc, get_ppdd)  # type: ignore[arg-type]
    elif solver == 5:
        patankar_runge_kutta_2(dt, numc, nlev, cc, get_ppdd)  # type: ignore[arg-type]
    elif solver == 6:
        patankar_runge_kutta_4(dt, numc, nlev, cc, get_ppdd)  # type: ignore[arg-type]
    elif solver == 7:
        modified_patankar(dt, numc, nlev, cc, get_ppdd)  # type: ignore[arg-type]
    elif solver == 8:
        modified_patankar_2(dt, numc, nlev, cc, get_ppdd)  # type: ignore[arg-type]
    elif solver == 9:
        modified_patankar_4(dt, numc, nlev, cc, get_ppdd)  # type: ignore[arg-type]
    elif solver == 10:
        emp_1(dt, numc, nlev, cc, get_rhs)  # type: ignore[arg-type]
    elif solver == 11:
        emp_2(dt, numc, nlev, cc, get_rhs)  # type: ignore[arg-type]
    else:
        raise ValueError(
            "ode_solvers_template: no valid solver method specified in gotm.yaml !"
            f" (solver={solver})"
        )
