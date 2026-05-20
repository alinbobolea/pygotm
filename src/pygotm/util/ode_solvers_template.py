"""
ODE solver template — translation of ``ode_solvers_template.F90``.

Alternate implementations of the Runge-Kutta solvers (codes 2 and 3) that
differ from :mod:`~pygotm.util.ode_solvers`:

* **RK2** (code 2): explicit midpoint method.

  .. math::

     c^{\\mathrm{mid}} = c^n + \\tfrac{\\Delta t}{2}\\,f(c^n),\\quad
     c^{n+1} = c^n + \\Delta t\\,f(c^{\\mathrm{mid}})

* **RK4** (code 3): standard four-stage RK with half-step intermediates.

  .. math::

     k_1 = f(c^n),\\quad k_2 = f(c^n + \\tfrac{\\Delta t}{2}k_1),\\quad
     k_3 = f(c^n + \\tfrac{\\Delta t}{2}k_2),\\quad k_4 = f(c^n + \\Delta t\\,k_3)

  .. math::

     c^{n+1} = c^n + \\tfrac{\\Delta t}{3}\\left(\\tfrac{k_1}{2} + k_2 + k_3 + \\tfrac{k_4}{2}\\right)

All other solvers (Euler, Patankar, Modified Patankar, EMP — codes 1, 4–11)
are re-exported unchanged from :mod:`~pygotm.util.ode_solvers`.

Original FORTRAN authors: Hans Burchard, Karsten Bolding.
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

    Two evaluations of the right-hand side per time step:

    .. math::

       c^{\\mathrm{mid}} &= c^n + \\tfrac{\\Delta t}{2}\\,f(c^n) \\\\
       c^{n+1}           &= c^n + \\Delta t\\,f(c^{\\mathrm{mid}})

    This is the ``ode_solvers_template.F90`` variant.  The non-template
    ``ode_solvers.F90`` uses Heun's trapezoidal method instead.
    Updates ``cc`` in-place.  Level 0 is never modified.
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

    Four evaluations of the right-hand side per time step with standard
    half-step intermediates (achieves true 4th-order accuracy):

    .. math::

       k_1 &= f(c^n),\\quad k_2 = f\\!\\left(c^n + \\tfrac{\\Delta t}{2}k_1\\right),\\quad
       k_3 = f\\!\\left(c^n + \\tfrac{\\Delta t}{2}k_2\\right),\\quad
       k_4 = f(c^n + \\Delta t\\,k_3) \\\\
       c^{n+1} &= c^n + \\tfrac{\\Delta t}{3}\\bigl(\\tfrac{k_1}{2} + k_2 + k_3 + \\tfrac{k_4}{2}\\bigr)

    This is the ``ode_solvers_template.F90`` variant.  The non-template
    ``ode_solvers.F90`` uses full-:math:`\\Delta t` intermediate steps instead.
    Updates ``cc`` in-place.  Level 0 is never modified.
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

    Identical to :func:`pygotm.util.ode_solvers.ode_solver` except that
    solvers 2 and 3 use the template-version algorithms: midpoint RK2 and
    standard half-step RK4.  Solvers 4–11 delegate to the shared
    implementations in :mod:`~pygotm.util.ode_solvers`.

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
