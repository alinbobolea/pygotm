"""
ODE solvers for biogeochemical models — translation of ``ode_solvers.F90``.

Provides 11 numerical solvers for the reaction-term ODEs that arise in
biogeochemical models:

.. math::

   \\partial_t c_i = P_i(\\mathbf{c}) - D_i(\\mathbf{c}), \\quad i = 1, \\ldots, I

where :math:`c_i` are species concentrations and :math:`P_i`, :math:`D_i` are
the production and destruction terms (Burchard et al. 2003).

.. list-table:: Available solvers
   :header-rows: 1
   :widths: 10 55 20 15

   * - Code
     - Method
     - Conservative
     - Positive
   * - 1
     - First-order explicit (Euler)
     - No
     - No
   * - 2
     - Second-order explicit Runge-Kutta
     - No
     - No
   * - 3
     - Fourth-order explicit Runge-Kutta
     - No
     - No
   * - 4
     - First-order Patankar
     - No
     - Yes
   * - 5
     - Second-order Patankar-Runge-Kutta
     - No
     - Yes
   * - 6
     - Fourth-order Patankar-Runge-Kutta (**non-functional**)
     - —
     - —
   * - 7
     - First-order Modified Patankar
     - Yes
     - Yes
   * - 8
     - Second-order Modified Patankar-Runge-Kutta
     - Yes
     - Yes
   * - 9
     - Fourth-order Modified Patankar-Runge-Kutta (**non-functional**)
     - —
     - —
   * - 10
     - First-order Extended Modified Patankar (EMP)
     - Stoichiometric
     - Yes
   * - 11
     - Second-order Extended Modified Patankar-Runge-Kutta (EMP)
     - Stoichiometric
     - Yes

Schemes 6 and 9 are not yet developed in Fortran GOTM and are non-functional
here.  EMP schemes (10–11) — developed by Bruggeman et al. (2005) — extend
Modified Patankar to full stoichiometric conservation with multiple limiting
nutrients.

Original FORTRAN authors: Hans Burchard, Karsten Bolding.

Public interface: :func:`ode_solver`, :class:`RhsCallback`,
:class:`PpddCallback`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

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

# Callback type aliases matching Fortran interface signatures.
# get_rhs(first, numc, nlev, cc) -> rhs, where cc and rhs have shape
# (numc, nlev+1).
# get_ppdd(first, numc, nlev, cc) -> (pp, dd), where pp and dd have shape
# (numc, numc, nlev+1).
RhsCallback = Callable[[bool, int, int, np.ndarray], np.ndarray]
PpddCallback = Callable[[bool, int, int, np.ndarray], tuple[np.ndarray, np.ndarray]]


def matrix_solve(n: int, a: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Gaussian forward-elimination + back-substitution solver.

    Solves the n×n system A·c = r.  Modifies working copies of ``a`` and ``r``
    internally; does not alter the caller's arrays.

    Parameters
    ----------
    n : int
        System dimension.
    a : np.ndarray, shape (n, n)
        Coefficient matrix (modified internally; caller's copy is unchanged).
    r : np.ndarray, shape (n,)
        Right-hand side vector (modified internally).

    Returns
    -------
    c : np.ndarray, shape (n,)
        Solution vector.
    """
    a = a.astype(np.float64).copy()
    r = r.astype(np.float64).copy()

    for i in range(n):
        pivot = a[i, i]
        r[i] /= pivot
        for j in range(n - 1, i - 1, -1):  # j from n-1 down to i
            a[i, j] /= pivot
        for k in range(i + 1, n):
            mult = a[k, i]
            r[k] -= mult * r[i]
            for j in range(i + 1, n):
                a[k, j] -= mult * a[i, j]

    c = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        c[i] = r[i]
        for j in range(i + 1, n):
            c[i] -= a[i, j] * c[j]
    return c


def findp_bisection(
    numc: int,
    cc: np.ndarray,
    derivative: np.ndarray,
    dt: float,
    accuracy: float,
) -> float:
    """Find the EMP product term p via bisection.

    Solves the non-linear problem:

    .. code-block:: text

       c^{n+1} = c^n + dt * f(t^n, c^n) * prod_{j in J} (c_j^{n+1} / c_j^n)

    with ``J = {i : f_i < 0}`` by reducing it to a polynomial root problem and
    applying 20 bisection iterations.  It has been proved that there exists
    exactly one ``p`` for which the above is true (Bruggeman et al. 2005).

    Original FORTRAN author: Jorn Bruggeman.

    Parameters
    ----------
    numc : int
        Number of state variables.
    cc : np.ndarray, shape (numc,)
        Current concentrations at one vertical level.
    derivative : np.ndarray, shape (numc,)
        Right-hand side f evaluated at current state.
    dt : float
        Time step [s].
    accuracy : float
        Convergence criterion on the relative pi interval width.

    Returns
    -------
    pi : float
        EMP product term p in (0, 1].
    """
    potnegcount = 0
    piright = 1.0
    rel_deriv: list[float] = []

    for i in range(numc):
        if derivative[i] < 0.0:
            # State variable could become zero or less; include it in J.
            if cc[i] == 0.0:
                print(f"Error: state variable {i} is zero and has negative derivative!")
            rd = dt * derivative[i] / cc[i]
            rel_deriv.append(rd)
            # Negative derivative places an upper bound on pi.
            bound = -1.0 / rd
            if bound < piright:
                piright = bound
            potnegcount += 1

    if potnegcount == 0:
        # All derivatives are positive — pure Euler step is safe.
        return 1.0

    pileft = 0.0  # polynomial(0) = 1

    pi = 0.5 * (piright + pileft)
    for _ in range(20):
        pi = 0.5 * (piright + pileft)
        fnow = 1.0
        for i in range(potnegcount):
            fnow *= 1.0 + rel_deriv[i] * pi

        if fnow > pi:
            pileft = pi
        elif fnow < pi:
            piright = pi
        else:
            break  # exact root found
        if (piright - pileft) / pi < accuracy:
            break

    # Low pi implies stiff or non-positive system; EMP will stall.
    if pi < 1.0e-4:
        print(
            f"Warning: small pi={pi} in Extended Modified Patankar slows down system!"
        )
    return pi


def _build_mp_matrix(
    numc: int,
    ci: int,
    cc: np.ndarray,
    pp: np.ndarray,
    dd: np.ndarray,
    dt: float,
    cc_denom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the Modified Patankar linear system for one vertical level.

    Returns (a, r) where a is (numc, numc) and r is (numc,).
    cc_denom is the denominator state (cc or cc1 depending on the stage).
    """
    a = np.zeros((numc, numc))
    r = np.zeros(numc)
    for i in range(numc):
        diag_sum = 0.0
        for j in range(numc):
            diag_sum += dd[i, j, ci]
            if i != j:
                a[i, j] = -dt * pp[i, j, ci] / cc_denom[j, ci]
        a[i, i] = 1.0 + dt * diag_sum / cc_denom[i, ci]
        r[i] = cc[i, ci] + dt * pp[i, i, ci]
    return a, r


# ---------------------------------------------------------------------------
# Explicit Runge-Kutta solvers (get_rhs interface)
# ---------------------------------------------------------------------------


def euler_forward(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_rhs: RhsCallback,
) -> None:
    """First-order Euler-forward (E1) scheme.

    One evaluation of the right-hand side per time step:

    .. math::

       c_i^{n+1} = c_i^n + \\Delta t\\,\\bigl(P_i(c^n) - D_i(c^n)\\bigr)

    Updates ``cc`` in-place.  Level 0 (index 0) is never modified.
    """
    rhs = get_rhs(True, numc, nlev, cc)
    cc[:, 1:] += dt * rhs[:, 1:]


def runge_kutta_2(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_rhs: RhsCallback,
) -> None:
    """Second-order Runge-Kutta — Heun's method (explicit trapezoidal).

    Two evaluations of the right-hand side per time step:

    .. math::

       c_i^{(1)} &= c_i^n + \\Delta t\\,f_i(c^n) \\\\
       c_i^{n+1} &= c_i^n + \\tfrac{\\Delta t}{2}\\bigl(f_i(c^n) + f_i(c^{(1)})\\bigr)

    where :math:`f_i = P_i - D_i`.  Updates ``cc`` in-place.  Level 0 is
    never modified.
    """
    rhs = get_rhs(True, numc, nlev, cc)

    cc1 = cc.copy()
    cc1[:, 1:] += dt * rhs[:, 1:]

    rhs1 = get_rhs(False, numc, nlev, cc1)

    cc[:, 1:] += dt * 0.5 * (rhs[:, 1:] + rhs1[:, 1:])


def runge_kutta_4(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_rhs: RhsCallback,
) -> None:
    """Fourth-order Runge-Kutta scheme (ode_solvers.F90 variant — full-dt steps).

    Four evaluations of the right-hand side per time step using full-:math:`\\Delta t`
    intermediates (differs from the standard half-step RK4 in
    :mod:`~pygotm.util.ode_solvers_template`):

    .. math::

       c^{(1)} &= c^n + \\Delta t\\,f(c^n) \\\\
       c^{(2)} &= c^n + \\Delta t\\,f(c^{(1)}) \\\\
       c^{(3)} &= c^n + \\Delta t\\,f(c^{(2)}) \\\\
       c^{n+1} &= c^n + \\tfrac{\\Delta t}{3}\\bigl(
           \\tfrac{1}{2}f(c^n) + f(c^{(1)}) + f(c^{(2)}) + \\tfrac{1}{2}f(c^{(3)})\\bigr)

    Updates ``cc`` in-place.  Level 0 is never modified.
    """
    rhs = get_rhs(True, numc, nlev, cc)

    cc1 = cc.copy()
    cc1[:, 1:] += dt * rhs[:, 1:]
    rhs1 = get_rhs(False, numc, nlev, cc1)

    cc1[:] = cc[:]
    cc1[:, 1:] += dt * rhs1[:, 1:]
    rhs2 = get_rhs(False, numc, nlev, cc1)

    cc1[:] = cc[:]
    cc1[:, 1:] += dt * rhs2[:, 1:]
    rhs3 = get_rhs(False, numc, nlev, cc1)

    cc[:, 1:] += (
        dt / 3.0 * (0.5 * rhs[:, 1:] + rhs1[:, 1:] + rhs2[:, 1:] + 0.5 * rhs3[:, 1:])
    )


# ---------------------------------------------------------------------------
# Patankar solvers (get_ppdd interface)
# ---------------------------------------------------------------------------


def patankar(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_ppdd: PpddCallback,
) -> None:
    """First-order Patankar-Euler (PE1) scheme.

    One evaluation of the production/destruction terms per time step.
    Not conservative; unconditionally positive:

    .. math::

       c_i^{n+1} = \\frac{c_i^n + \\Delta t\\,P_i(c^n)}{1 + \\Delta t\\,D_i(c^n)/c_i^n}

    Updates ``cc`` in-place.  Level 0 is never modified.
    """
    pp, dd = get_ppdd(True, numc, nlev, cc)

    for ci in range(1, nlev + 1):
        for i in range(numc):
            ppsum = float(pp[i, :, ci].sum())
            ddsum = float(dd[i, :, ci].sum())
            cc[i, ci] = (cc[i, ci] + dt * ppsum) / (1.0 + dt * ddsum / cc[i, ci])


def patankar_runge_kutta_2(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_ppdd: PpddCallback,
) -> None:
    """Second-order Patankar-Runge-Kutta (PRK2) scheme.

    Two evaluations of the production/destruction terms per time step.
    Not conservative; unconditionally positive (Burchard et al. 2003).

    Stage 1 — Patankar-Euler predictor:

    .. math::

       c_i^{(1)} = \\frac{c_i^n + \\Delta t\\,P_i(c^n)}{1 + \\Delta t\\,D_i(c^n)/c_i^n}

    Stage 2 — second-order corrector:

    .. math::

       c_i^{n+1} = \\frac{c_i^n + \\tfrac{\\Delta t}{2}(P_i(c^n)+P_i(c^{(1)}))}
                        {1 + \\tfrac{\\Delta t}{2}(D_i(c^n)+D_i(c^{(1)}))/c_i^{(1)}}

    Updates ``cc`` in-place.  Level 0 is never modified.
    """
    pp, dd = get_ppdd(True, numc, nlev, cc)

    ppsum = np.zeros((numc, nlev + 1))
    ddsum = np.zeros((numc, nlev + 1))
    cc1 = np.zeros((numc, nlev + 1))

    for ci in range(1, nlev + 1):
        for i in range(numc):
            ppsum[i, ci] = float(pp[i, :, ci].sum())
            ddsum[i, ci] = float(dd[i, :, ci].sum())
            cc1[i, ci] = (cc[i, ci] + dt * ppsum[i, ci]) / (
                1.0 + dt * ddsum[i, ci] / cc[i, ci]
            )

    pp2, dd2 = get_ppdd(False, numc, nlev, cc1)

    for ci in range(1, nlev + 1):
        for i in range(numc):
            ppsum[i, ci] += float(pp2[i, :, ci].sum())
            ddsum[i, ci] += float(dd2[i, :, ci].sum())
            cc[i, ci] = (cc[i, ci] + 0.5 * dt * ppsum[i, ci]) / (
                1.0 + 0.5 * dt * ddsum[i, ci] / cc1[i, ci]
            )


def patankar_runge_kutta_4(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_ppdd: PpddCallback,
) -> None:
    """Fourth-order Patankar-Runge-Kutta (PRK4) scheme — **does not work**.

    This scheme has not yet been developed in GOTM Fortran or here; the
    implementation is a placeholder.  Updates ``cc`` in-place.  Level 0 is
    never modified.
    """
    pp, dd = get_ppdd(True, numc, nlev, cc)

    ppsum = np.zeros((numc, nlev + 1))
    ddsum = np.zeros((numc, nlev + 1))
    ppsum1 = np.zeros((numc, nlev + 1))
    ddsum1 = np.zeros((numc, nlev + 1))
    ppsum2 = np.zeros((numc, nlev + 1))
    ddsum2 = np.zeros((numc, nlev + 1))
    ppsum3 = np.zeros((numc, nlev + 1))
    ddsum3 = np.zeros((numc, nlev + 1))
    cc1 = np.zeros((numc, nlev + 1))

    # Stage 1
    for ci in range(1, nlev + 1):
        for i in range(numc):
            ppsum[i, ci] = float(pp[i, :, ci].sum())
            ddsum[i, ci] = float(dd[i, :, ci].sum())
            cc1[i, ci] = (cc[i, ci] + dt * ppsum[i, ci]) / (
                1.0 + dt * ddsum[i, ci] / cc[i, ci]
            )

    pp, dd = get_ppdd(False, numc, nlev, cc1)

    # Stage 2
    for ci in range(1, nlev + 1):
        for i in range(numc):
            ppsum1[i, ci] = float(pp[i, :, ci].sum())
            ddsum1[i, ci] = float(dd[i, :, ci].sum())
            cc1[i, ci] = (cc[i, ci] + dt * ppsum1[i, ci]) / (
                1.0 + dt * ddsum1[i, ci] / cc1[i, ci]
            )

    pp, dd = get_ppdd(False, numc, nlev, cc1)

    # Stage 3
    for ci in range(1, nlev + 1):
        for i in range(numc):
            ppsum2[i, ci] = float(pp[i, :, ci].sum())
            ddsum2[i, ci] = float(dd[i, :, ci].sum())
            cc1[i, ci] = (cc[i, ci] + dt * ppsum2[i, ci]) / (
                1.0 + dt * ddsum2[i, ci] / cc1[i, ci]
            )

    pp, dd = get_ppdd(False, numc, nlev, cc1)

    # Stage 4 and final update
    for ci in range(1, nlev + 1):
        for i in range(numc):
            ppsum3[i, ci] = float(pp[i, :, ci].sum())
            ddsum3[i, ci] = float(dd[i, :, ci].sum())
            ppsum[i, ci] = (
                0.5 * ppsum[i, ci] + ppsum1[i, ci] + ppsum2[i, ci] + 0.5 * ppsum3[i, ci]
            ) / 3.0
            ddsum[i, ci] = (
                0.5 * ddsum[i, ci] + ddsum1[i, ci] + ddsum2[i, ci] + 0.5 * ddsum3[i, ci]
            ) / 3.0
            cc[i, ci] = (cc[i, ci] + dt * ppsum[i, ci]) / (
                1.0 + dt * ddsum[i, ci] / cc1[i, ci]
            )


# ---------------------------------------------------------------------------
# Modified Patankar solvers (conservative + positive)
# ---------------------------------------------------------------------------


def modified_patankar(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_ppdd: PpddCallback,
) -> None:
    """First-order Modified Patankar-Euler (MPE1) scheme.

    One evaluation of the production/destruction terms per time step.
    Conservative and unconditionally positive (Burchard et al. 2003):

    .. math::

       c_i^{n+1} = c_i^n + \\Delta t\\left[
           \\sum_{j \\neq i} p_{ij}(c^n)\\,\\frac{c_j^{n+1}}{c_j^n}
           + p_{ii}(c^n)
           - \\sum_j d_{ij}(c^n)\\,\\frac{c_i^{n+1}}{c_i^n}
       \\right]

    Solved as a linear system per vertical level using :func:`matrix_solve`.
    Updates ``cc`` in-place.  Level 0 is never modified.
    """
    pp, dd = get_ppdd(True, numc, nlev, cc)

    for ci in range(1, nlev + 1):
        a, r = _build_mp_matrix(numc, ci, cc, pp, dd, dt, cc)
        cc[:, ci] = matrix_solve(numc, a, r)


def modified_patankar_2(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_ppdd: PpddCallback,
) -> None:
    """Second-order Modified Patankar-Runge-Kutta (MPRK2) scheme.

    Two evaluations of the production/destruction terms per time step.
    Conservative and unconditionally positive (Burchard et al. 2003).

    Stage 1: Modified Patankar-Euler step to obtain the predictor
    :math:`c^{(1)}` (same system as :func:`modified_patankar`).

    Stage 2: second-order corrector using averaged production/destruction
    terms :math:`\\frac{1}{2}(pp + pp^{(1)})` and :math:`\\frac{1}{2}(dd + dd^{(1)})`,
    with :math:`c^{(1)}` as the denominator state.

    Updates ``cc`` in-place.  Level 0 is never modified.
    """
    pp, dd = get_ppdd(True, numc, nlev, cc)
    cc1 = np.zeros((numc, nlev + 1))

    # Stage 1: Modified Patankar-Euler
    for ci in range(1, nlev + 1):
        a, r = _build_mp_matrix(numc, ci, cc, pp, dd, dt, cc)
        cc1[:, ci] = matrix_solve(numc, a, r)

    pp1, dd1 = get_ppdd(False, numc, nlev, cc1)

    pp_avg = 0.5 * (pp + pp1)
    dd_avg = 0.5 * (dd + dd1)

    # Stage 2: second-order update using averaged fluxes and cc1 as denominator
    for ci in range(1, nlev + 1):
        a, r = _build_mp_matrix(numc, ci, cc, pp_avg, dd_avg, dt, cc1)
        cc[:, ci] = matrix_solve(numc, a, r)


def modified_patankar_4(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_ppdd: PpddCallback,
) -> None:
    """Fourth-order Modified Patankar-Runge-Kutta (MPRK4) — **does not work**.

    This scheme has not yet been developed in GOTM Fortran or here; the
    implementation is a placeholder.  Updates ``cc`` in-place.  Level 0 is
    never modified.
    """
    pp, dd = get_ppdd(True, numc, nlev, cc)
    cc1 = np.zeros((numc, nlev + 1))

    for ci in range(1, nlev + 1):
        a, r = _build_mp_matrix(numc, ci, cc, pp, dd, dt, cc)
        cc1[:, ci] = matrix_solve(numc, a, r)

    pp1, dd1 = get_ppdd(False, numc, nlev, cc1)

    for ci in range(1, nlev + 1):
        a, r = _build_mp_matrix(numc, ci, cc, pp1, dd1, dt, cc1)
        cc1[:, ci] = matrix_solve(numc, a, r)

    pp2, dd2 = get_ppdd(False, numc, nlev, cc1)

    for ci in range(1, nlev + 1):
        a, r = _build_mp_matrix(numc, ci, cc, pp2, dd2, dt, cc1)
        cc1[:, ci] = matrix_solve(numc, a, r)

    pp3, dd3 = get_ppdd(False, numc, nlev, cc1)

    pp_avg = (pp / 2.0 + pp1 + pp2 + pp3 / 2.0) / 3.0
    dd_avg = (dd / 2.0 + dd1 + dd2 + dd3 / 2.0) / 3.0

    for ci in range(1, nlev + 1):
        a, r = _build_mp_matrix(numc, ci, cc, pp_avg, dd_avg, dt, cc1)
        cc[:, ci] = matrix_solve(numc, a, r)


# ---------------------------------------------------------------------------
# Extended Modified Patankar solvers (stoichiometrically conservative)
# ---------------------------------------------------------------------------


def emp_1(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_rhs: RhsCallback,
) -> None:
    """First-order Extended Modified Patankar (EMP-1) scheme.

    One evaluation of the right-hand side per time step.
    Stoichiometrically conservative and positive (Bruggeman et al. 2005):

    .. math::

       c^{n+1} = c^n + \\Delta t\\,f(t^n, c^n)\\prod_{j \\in J^n}
           \\frac{c_j^{n+1}}{c_j^n},
       \\quad J^n = \\{i : f_i(t^n, c^n) < 0\\}

    The product term :math:`p` is found via :func:`findp_bisection`.
    Updates ``cc`` in-place.  Level 0 is never modified.
    """
    derivative = get_rhs(True, numc, nlev, cc)

    for ci in range(1, nlev + 1):
        pi = findp_bisection(numc, cc[:, ci], derivative[:, ci], dt, 1.0e-9)
        cc[:, ci] += dt * derivative[:, ci] * pi


def emp_2(
    dt: float,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    get_rhs: RhsCallback,
) -> None:
    """Second-order Extended Modified Patankar (EMP-2) scheme.

    Two evaluations of the right-hand side per time step.
    Stoichiometrically conservative and positive (Bruggeman et al. 2005).

    Step 1: identical to EMP-1 — advance ``cc`` to a midpoint ``cc_med``
    using the product term from :func:`findp_bisection`.

    Step 2: average :math:`f(c^n)` and :math:`f(c^{\\mathrm{med}})`, correct
    for state variables in the negative-flux set :math:`J`, then apply a
    second :func:`findp_bisection` step to update ``cc``.

    Updates ``cc`` in-place.  Level 0 is never modified.
    """
    rhs = get_rhs(True, numc, nlev, cc)
    cc_med = cc.copy()

    # First step (identical to EMP-1)
    for ci in range(1, nlev + 1):
        pi = findp_bisection(numc, cc[:, ci], rhs[:, ci], dt, 1.0e-9)
        cc_med[:, ci] = cc[:, ci] + dt * rhs[:, ci] * pi

    rhs_med = get_rhs(False, numc, nlev, cc_med)

    for ci in range(1, nlev + 1):
        rhs[:, ci] = 0.5 * (rhs[:, ci] + rhs_med[:, ci])

        # Correct for state variables included in the J set.
        for i in range(numc):
            if rhs[i, ci] < 0.0:
                rhs[:, ci] *= cc[i, ci] / cc_med[i, ci]

        pi = findp_bisection(numc, cc[:, ci], rhs[:, ci], dt, 1.0e-9)
        cc[:, ci] += dt * rhs[:, ci] * pi


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def ode_solver(
    solver: int,
    numc: int,
    nlev: int,
    dt: float,
    cc: np.ndarray,
    get_rhs: RhsCallback | None = None,
    get_ppdd: PpddCallback | None = None,
) -> None:
    """Dispatch to one of 11 ODE solvers for biogeochemical reaction equations.

    Solvers 1–3 and 10–11 require ``get_rhs``; solvers 4–9 require
    ``get_ppdd``.  See the module docstring for the full solver table.

    Parameters
    ----------
    solver : int
        Solver identifier (1–11).
    numc : int
        Number of biogeochemical state variables.
    nlev : int
        Number of vertical levels (cc has shape (numc, nlev+1)).
    dt : float
        Time step [s].
    cc : np.ndarray, shape (numc, nlev+1)
        State variable array, modified in-place. Level 0 is not updated.
    get_rhs : callable, optional
        RHS callback get_rhs(first, numc, nlev, cc) → rhs. Required for solvers
        1, 2, 3, 10, 11.
    get_ppdd : callable, optional
        Production-destruction callback get_ppdd(first, numc, nlev, cc) →
        (pp, dd). Required for solvers 4–9.
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
            f"bio: no valid solver method specified in gotm.yaml ! (solver={solver})"
        )
