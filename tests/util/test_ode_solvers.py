"""Tests for util/ode_solvers.py — Step 1.9 of GOTM translation plan."""

from __future__ import annotations

import numpy as np
import pytest

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
    ode_solver,
    patankar,
    patankar_runge_kutta_2,
    patankar_runge_kutta_4,
    runge_kutta_2,
    runge_kutta_4,
)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

NLEV = 4
NUMC = 2

# ---------------------------------------------------------------------------
# Callback factories
# ---------------------------------------------------------------------------


def _rhs_decay(k: float) -> RhsCallback:
    """Returns rhs callback for dc/dt = -k*c (all species, all levels)."""

    def get_rhs(first: bool, numc: int, nlev: int, cc: np.ndarray) -> np.ndarray:
        return -k * cc

    return get_rhs


def _rhs_growth(k: float) -> RhsCallback:
    """Returns rhs callback for dc/dt = +k (constant source, all species)."""

    def get_rhs(first: bool, numc: int, nlev: int, cc: np.ndarray) -> np.ndarray:
        return np.full_like(cc, k)

    return get_rhs


def _ppdd_decay(k: float) -> PpddCallback:
    """Returns ppdd callback for pure decay dc/dt = -k*c.

    pp = 0 (no production)
    dd[i,i,ci] = k * cc[i,ci]  absolute destruction flux (concentration/time)
    """

    def get_ppdd(
        first: bool, numc: int, nlev: int, cc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pp = np.zeros((numc, numc, nlev + 1))
        dd = np.zeros((numc, numc, nlev + 1))
        for i in range(numc):
            dd[i, i, 1:] = k * cc[i, 1:]  # absolute flux = rate * concentration
        return pp, dd

    return get_ppdd


def _ppdd_transfer() -> PpddCallback:
    """Returns ppdd callback for a 2-species P→N transfer (conservation test).

    pp and dd are absolute fluxes (concentration/time), consistent with the
    Patankar/Modified Patankar formulation. Total P+N is conserved exactly
    by the Modified Patankar scheme.
    """
    k = 1.0

    def get_ppdd(
        first: bool, numc: int, nlev: int, cc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pp = np.zeros((numc, numc, nlev + 1))
        dd = np.zeros((numc, numc, nlev + 1))
        pp[1, 0, 1:] = k * cc[0, 1:]  # N gains from P (absolute flux)
        dd[0, 0, 1:] = k * cc[0, 1:]  # P self-destructs (absolute flux)
        return pp, dd

    return get_ppdd


def _make_cc(numc: int = NUMC, nlev: int = NLEV, val: float = 1.0) -> np.ndarray:
    """Initial concentration array, shape (numc, nlev+1), all levels set to val.

    Level 0 is set to a different sentinel value to detect if it gets modified.
    """
    cc = np.full((numc, nlev + 1), val)
    cc[:, 0] = 99.0  # sentinel: level 0 must never be modified by ODE solvers
    return cc


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.util.ode_solvers import ode_solver  # noqa: F401


# ---------------------------------------------------------------------------
# 2. matrix_solve — Gaussian elimination
# ---------------------------------------------------------------------------


def test_matrix_solve_2x2_known_solution() -> None:
    """2x+y=5, x+3y=7 → x=1.6, y=1.8."""
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    r = np.array([5.0, 7.0])
    c = matrix_solve(2, a, r)
    assert c[0] == pytest.approx(1.6, rel=1e-12)
    assert c[1] == pytest.approx(1.8, rel=1e-12)


def test_matrix_solve_1x1() -> None:
    a = np.array([[4.0]])
    r = np.array([12.0])
    c = matrix_solve(1, a, r)
    assert c[0] == pytest.approx(3.0, rel=1e-12)


def test_matrix_solve_3x3() -> None:
    """Known 3×3 system."""
    a = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 4.0]])
    x_exact = np.array([1.0, 2.0, 3.0])
    r = a @ x_exact
    c = matrix_solve(3, a.copy(), r.copy())
    np.testing.assert_allclose(c, x_exact, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. findp_bisection
# ---------------------------------------------------------------------------


def test_findp_bisection_all_positive_derivatives() -> None:
    """When all derivatives are non-negative, pi must equal 1.0 (pure Euler)."""
    cc = np.array([1.0, 2.0])
    deriv = np.array([0.5, 0.3])  # positive
    pi = findp_bisection(2, cc, deriv, dt=1.0, accuracy=1e-9)
    assert pi == pytest.approx(1.0, abs=1e-12)


def test_findp_bisection_one_negative() -> None:
    """With a negative derivative, pi must be < 1 and ensure positivity."""
    cc = np.array([1.0])
    deriv = np.array([-2.0])  # without EMP: cc + dt*deriv = 1 - 2 < 0
    dt = 1.0
    pi = findp_bisection(1, cc, deriv, dt=dt, accuracy=1e-9)
    assert pi < 1.0
    assert pi > 0.0
    # Resulting cc must be positive
    cc_new = cc + dt * deriv * pi
    assert cc_new[0] > 0.0


def test_findp_bisection_positivity_guarantee() -> None:
    """All resulting concentrations must be positive after applying pi."""
    numc = 3
    cc = np.array([0.5, 1.0, 2.0])
    deriv = np.array([-3.0, -0.5, 0.2])
    dt = 0.5
    pi = findp_bisection(numc, cc, deriv, dt=dt, accuracy=1e-9)
    cc_new = cc + dt * deriv * pi
    assert np.all(cc_new > 0.0)


def test_findp_bisection_result_in_bounds() -> None:
    """pi must be in (0, 1]."""
    cc = np.array([1.0, 1.0])
    deriv = np.array([-1.0, -0.5])
    pi = findp_bisection(2, cc, deriv, dt=1.0, accuracy=1e-9)
    assert 0.0 < pi <= 1.0


# ---------------------------------------------------------------------------
# 4. euler_forward — analytic verification
# ---------------------------------------------------------------------------


def test_euler_forward_exact_one_step() -> None:
    """Euler: c_new = c + dt*(-k*c) = c*(1 - k*dt). Exact for one step."""
    k = 0.5
    dt = 0.1
    numc, nlev = 1, 3
    cc = _make_cc(numc, nlev, val=2.0)
    cc_expected = cc.copy()
    cc_expected[:, 1:] = cc[:, 1:] * (1 - k * dt)
    cc_expected[:, 0] = 99.0  # level 0 unchanged

    euler_forward(dt, numc, nlev, cc, _rhs_decay(k))

    np.testing.assert_allclose(cc[:, 1:], cc_expected[:, 1:], rtol=1e-14)
    # Level 0 must be untouched
    assert np.all(cc[:, 0] == 99.0)


def test_euler_forward_level_0_not_modified() -> None:
    """Level 0 must never be updated by euler_forward."""
    cc = _make_cc()
    euler_forward(0.1, NUMC, NLEV, cc, _rhs_decay(0.3))
    assert np.all(cc[:, 0] == 99.0)


def test_euler_forward_no_nan_inf() -> None:
    cc = _make_cc(val=1.0)
    cc[:, 1:] = np.random.default_rng(0).uniform(0.1, 10.0, (NUMC, NLEV))
    euler_forward(0.01, NUMC, NLEV, cc, _rhs_decay(1.0))
    assert not np.any(np.isnan(cc))
    assert not np.any(np.isinf(cc))


# ---------------------------------------------------------------------------
# 5. runge_kutta_2 — Heun's method
# ---------------------------------------------------------------------------


def test_runge_kutta_2_analytic_one_step() -> None:
    """Heun: c_new = c + dt/2*(f(c) + f(c+dt*f(c))).
    For dc/dt=-k*c: c_new = c*(1 - k*dt + k^2*dt^2/2).
    """
    k = 0.5
    dt = 0.2
    numc, nlev = 1, 2
    cc = _make_cc(numc, nlev, val=3.0)
    cc[:, 1:] = 3.0

    expected = 3.0 * (1 - k * dt + (k * dt) ** 2 / 2)
    runge_kutta_2(dt, numc, nlev, cc, _rhs_decay(k))

    np.testing.assert_allclose(cc[:, 1:], expected, rtol=1e-13)


def test_runge_kutta_2_second_order_convergence() -> None:
    """Halving dt should reduce error by ~4 (2nd order method)."""
    k = 1.0
    t_end = 0.5
    c0 = 1.0
    exact = c0 * np.exp(-k * t_end)

    def run(dt: float) -> float:
        nsteps = round(t_end / dt)
        numc, nlev = 1, 1
        cc = np.ones((numc, nlev + 1))
        cc[:, 0] = 99.0
        for _ in range(nsteps):
            runge_kutta_2(dt, numc, nlev, cc, _rhs_decay(k))
        return float(cc[0, 1])

    err1 = abs(run(0.01) - exact)
    err2 = abs(run(0.005) - exact)
    ratio = err1 / err2
    assert ratio == pytest.approx(4.0, rel=0.1)


def test_runge_kutta_2_level_0_not_modified() -> None:
    cc = _make_cc()
    runge_kutta_2(0.1, NUMC, NLEV, cc, _rhs_decay(0.2))
    assert np.all(cc[:, 0] == 99.0)


# ---------------------------------------------------------------------------
# 6. runge_kutta_4 — non-standard RK4 from ode_solvers.F90
# ---------------------------------------------------------------------------


def test_runge_kutta_4_produces_finite_output() -> None:
    """The ode_solvers.F90 RK4 uses full-dt intermediate steps (non-standard).
    It is NOT more accurate than RK2 for this reason. Verify it runs and
    produces finite, physically reasonable output.
    """
    k = 0.5
    dt = 0.05
    t_end = 1.0
    nsteps = round(t_end / dt)
    exact = np.exp(-k * t_end)
    numc, nlev = 1, 1
    cc = np.ones((numc, nlev + 1))
    cc[:, 0] = 99.0
    for _ in range(nsteps):
        runge_kutta_4(dt, numc, nlev, cc, _rhs_decay(k))
    assert not np.any(np.isnan(cc))
    assert not np.any(np.isinf(cc))
    # Should be in the right ballpark (within 50% of exact)
    assert abs(float(cc[0, 1]) - exact) < 0.5 * exact


def test_runge_kutta_4_converges_to_analytic() -> None:
    """Non-standard RK4 (ode_solvers.F90) converges to c(t)=c0*exp(-k*t).

    This variant uses full-dt intermediates so it is ~2nd order, not 4th.
    Halving dt should reduce error by at least 1.5x (better than 1st order).
    """
    k = 1.0
    t_end = 0.5
    c0 = 1.0
    exact = c0 * np.exp(-k * t_end)

    def run(dt: float) -> float:
        nsteps = round(t_end / dt)
        numc, nlev = 1, 1
        cc = np.ones((numc, nlev + 1))
        cc[:, 0] = 99.0
        cc[:, 1] = c0
        for _ in range(nsteps):
            runge_kutta_4(dt, numc, nlev, cc, _rhs_decay(k))
        return float(cc[0, 1])

    err1 = abs(run(0.01) - exact)
    err2 = abs(run(0.005) - exact)
    ratio = err1 / err2
    assert ratio > 1.5, f"Expected convergence ratio > 1.5, got {ratio:.3f}"


def test_runge_kutta_4_level_0_not_modified() -> None:
    cc = _make_cc()
    runge_kutta_4(0.1, NUMC, NLEV, cc, _rhs_decay(0.2))
    assert np.all(cc[:, 0] == 99.0)


def test_runge_kutta_4_no_nan_inf() -> None:
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    runge_kutta_4(0.05, NUMC, NLEV, cc, _rhs_decay(0.5))
    assert not np.any(np.isnan(cc))
    assert not np.any(np.isinf(cc))


# ---------------------------------------------------------------------------
# 7. patankar — unconditionally positive
# ---------------------------------------------------------------------------


def test_patankar_stays_positive_large_decay() -> None:
    """Patankar must keep cc > 0 even when dt*k >> 1 (stiff decay)."""
    k = 1000.0  # huge decay rate — explicit Euler would give negative cc
    dt = 1.0
    numc, nlev = 2, 4
    cc = _make_cc(numc, nlev, val=1.0)
    cc[:, 1:] = 1.0

    patankar(dt, numc, nlev, cc, _ppdd_decay(k))

    assert np.all(cc[:, 1:] > 0.0)


def test_patankar_analytic_decay() -> None:
    """For pure decay: cc_new = cc / (1 + dt*k)."""
    k = 2.0
    dt = 0.5
    numc, nlev = 1, 2
    cc = _make_cc(numc, nlev, val=4.0)
    cc[:, 1:] = 4.0
    expected = 4.0 / (1.0 + dt * k)

    patankar(dt, numc, nlev, cc, _ppdd_decay(k))

    np.testing.assert_allclose(cc[:, 1:], expected, rtol=1e-13)


def test_patankar_level_0_not_modified() -> None:
    cc = _make_cc()
    patankar(0.1, NUMC, NLEV, cc, _ppdd_decay(0.5))
    assert np.all(cc[:, 0] == 99.0)


# ---------------------------------------------------------------------------
# 8. patankar_runge_kutta_2
# ---------------------------------------------------------------------------


def test_patankar_runge_kutta_2_stays_positive() -> None:
    """PRK2 must keep cc > 0 for stiff decay."""
    k = 500.0
    dt = 1.0
    numc, nlev = 2, 3
    cc = _make_cc(numc, nlev, val=1.0)
    cc[:, 1:] = 1.0

    patankar_runge_kutta_2(dt, numc, nlev, cc, _ppdd_decay(k))

    assert np.all(cc[:, 1:] > 0.0)


def test_patankar_runge_kutta_2_level_0_not_modified() -> None:
    cc = _make_cc()
    patankar_runge_kutta_2(0.1, NUMC, NLEV, cc, _ppdd_decay(0.5))
    assert np.all(cc[:, 0] == 99.0)


def test_patankar_runge_kutta_2_no_nan_inf() -> None:
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    patankar_runge_kutta_2(0.1, NUMC, NLEV, cc, _ppdd_decay(0.5))
    assert not np.any(np.isnan(cc))
    assert not np.any(np.isinf(cc))


# ---------------------------------------------------------------------------
# 9. patankar_runge_kutta_4 (noted as non-functional in Fortran)
# ---------------------------------------------------------------------------


def test_patankar_runge_kutta_4_runs_without_error() -> None:
    """PRK4 is noted as non-functional in GOTM; verify it at least runs."""
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    patankar_runge_kutta_4(0.05, NUMC, NLEV, cc, _ppdd_decay(0.3))
    assert not np.any(np.isnan(cc))


def test_patankar_runge_kutta_4_level_0_not_modified() -> None:
    cc = _make_cc()
    patankar_runge_kutta_4(0.05, NUMC, NLEV, cc, _ppdd_decay(0.3))
    assert np.all(cc[:, 0] == 99.0)


# ---------------------------------------------------------------------------
# 10. modified_patankar — conservative and positive
# ---------------------------------------------------------------------------


def test_modified_patankar_stays_positive() -> None:
    k = 1000.0
    dt = 1.0
    numc, nlev = 2, 3
    cc = _make_cc(numc, nlev, val=1.0)
    cc[:, 1:] = 1.0

    modified_patankar(dt, numc, nlev, cc, _ppdd_decay(k))

    assert np.all(cc[:, 1:] > 0.0)


def test_modified_patankar_conservation_p_plus_n() -> None:
    """Modified Patankar must conserve P+N exactly for balanced transfer."""
    dt = 0.5
    numc, nlev = 2, 4
    cc = _make_cc(numc, nlev)
    cc[0, 1:] = 3.0  # P
    cc[1, 1:] = 1.0  # N
    total_before = cc[:, 1:].sum(axis=0).copy()

    modified_patankar(dt, numc, nlev, cc, _ppdd_transfer())

    total_after = cc[:, 1:].sum(axis=0)
    np.testing.assert_allclose(total_after, total_before, rtol=1e-13)


def test_modified_patankar_analytic_single_species() -> None:
    """For 1-species Modified Patankar with pure decay: cc_new = cc/(1+dt*k)."""
    k = 2.0
    dt = 0.3
    numc, nlev = 1, 2
    cc = _make_cc(numc, nlev, val=5.0)
    cc[:, 1:] = 5.0
    expected = 5.0 / (1.0 + dt * k)

    modified_patankar(dt, numc, nlev, cc, _ppdd_decay(k))

    np.testing.assert_allclose(cc[:, 1:], expected, rtol=1e-13)


def test_modified_patankar_level_0_not_modified() -> None:
    cc = _make_cc()
    modified_patankar(0.1, NUMC, NLEV, cc, _ppdd_decay(0.5))
    assert np.all(cc[:, 0] == 99.0)


# ---------------------------------------------------------------------------
# 11. modified_patankar_2
# ---------------------------------------------------------------------------


def test_modified_patankar_2_stays_positive() -> None:
    k = 500.0
    dt = 1.0
    numc, nlev = 2, 3
    cc = _make_cc(numc, nlev, val=1.0)
    cc[:, 1:] = 1.0

    modified_patankar_2(dt, numc, nlev, cc, _ppdd_decay(k))

    assert np.all(cc[:, 1:] > 0.0)


def test_modified_patankar_2_conservation() -> None:
    """MPRK2 conserves total concentration for P→N transfer."""
    dt = 0.3
    numc, nlev = 2, 4
    cc = _make_cc(numc, nlev)
    cc[0, 1:] = 2.0
    cc[1, 1:] = 1.0
    total_before = cc[:, 1:].sum(axis=0).copy()

    modified_patankar_2(dt, numc, nlev, cc, _ppdd_transfer())

    total_after = cc[:, 1:].sum(axis=0)
    np.testing.assert_allclose(total_after, total_before, rtol=1e-12)


def test_modified_patankar_2_level_0_not_modified() -> None:
    cc = _make_cc()
    modified_patankar_2(0.1, NUMC, NLEV, cc, _ppdd_decay(0.5))
    assert np.all(cc[:, 0] == 99.0)


# ---------------------------------------------------------------------------
# 12. modified_patankar_4 (noted as non-functional in Fortran)
# ---------------------------------------------------------------------------


def test_modified_patankar_4_runs_without_error() -> None:
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    modified_patankar_4(0.05, NUMC, NLEV, cc, _ppdd_decay(0.3))
    assert not np.any(np.isnan(cc))


def test_modified_patankar_4_level_0_not_modified() -> None:
    cc = _make_cc()
    modified_patankar_4(0.05, NUMC, NLEV, cc, _ppdd_decay(0.3))
    assert np.all(cc[:, 0] == 99.0)


# ---------------------------------------------------------------------------
# 13. emp_1 — first-order EMP
# ---------------------------------------------------------------------------


def test_emp_1_stays_positive_stiff() -> None:
    """EMP-1 must keep cc > 0 for a stiff negative RHS."""

    def get_rhs(first: bool, numc: int, nlev: int, cc: np.ndarray) -> np.ndarray:
        rhs = np.zeros_like(cc)
        rhs[:, 1:] = -100.0 * cc[:, 1:]  # strong decay
        return rhs

    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    emp_1(1.0, NUMC, NLEV, cc, get_rhs)
    assert np.all(cc[:, 1:] > 0.0)


def test_emp_1_positive_rhs_equals_euler() -> None:
    """When all derivatives are positive, EMP-1 reduces to Euler forward."""
    k = 0.3
    dt = 0.1
    numc, nlev = 1, 2

    cc_emp = _make_cc(numc, nlev, val=1.0)
    cc_emp[:, 1:] = 1.0
    cc_euler = cc_emp.copy()

    emp_1(dt, numc, nlev, cc_emp, _rhs_growth(k))
    euler_forward(dt, numc, nlev, cc_euler, _rhs_growth(k))

    np.testing.assert_allclose(cc_emp[:, 1:], cc_euler[:, 1:], rtol=1e-14)


def test_emp_1_level_0_not_modified() -> None:
    cc = _make_cc()
    emp_1(0.1, NUMC, NLEV, cc, _rhs_decay(0.5))
    assert np.all(cc[:, 0] == 99.0)


def test_emp_1_no_nan_inf() -> None:
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    emp_1(0.1, NUMC, NLEV, cc, _rhs_decay(0.5))
    assert not np.any(np.isnan(cc))
    assert not np.any(np.isinf(cc))


# ---------------------------------------------------------------------------
# 14. emp_2 — second-order EMP
# ---------------------------------------------------------------------------


def test_emp_2_stays_positive_stiff() -> None:
    def get_rhs(first: bool, numc: int, nlev: int, cc: np.ndarray) -> np.ndarray:
        rhs = np.zeros_like(cc)
        rhs[:, 1:] = -50.0 * cc[:, 1:]
        return rhs

    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    emp_2(1.0, NUMC, NLEV, cc, get_rhs)
    assert np.all(cc[:, 1:] > 0.0)


def test_emp_2_level_0_not_modified() -> None:
    cc = _make_cc()
    emp_2(0.1, NUMC, NLEV, cc, _rhs_decay(0.5))
    assert np.all(cc[:, 0] == 99.0)


def test_emp_2_no_nan_inf() -> None:
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    emp_2(0.1, NUMC, NLEV, cc, _rhs_decay(0.5))
    assert not np.any(np.isnan(cc))
    assert not np.any(np.isinf(cc))


# ---------------------------------------------------------------------------
# 15. ode_solver dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_id", [1, 2, 3, 10, 11])
def test_ode_solver_rhs_based_solvers(solver_id: int) -> None:
    """Solvers 1,2,3,10,11 use get_rhs callback."""
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    ode_solver(solver_id, NUMC, NLEV, 0.05, cc, get_rhs=_rhs_decay(0.1))
    assert not np.any(np.isnan(cc))
    assert np.all(cc[:, 0] == 99.0)


@pytest.mark.parametrize("solver_id", [4, 5, 6, 7, 8, 9])
def test_ode_solver_ppdd_based_solvers(solver_id: int) -> None:
    """Solvers 4-9 use get_ppdd callback."""
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    ode_solver(solver_id, NUMC, NLEV, 0.05, cc, get_ppdd=_ppdd_decay(0.1))
    assert not np.any(np.isnan(cc))
    assert np.all(cc[:, 0] == 99.0)


def test_ode_solver_invalid_raises() -> None:
    cc = _make_cc()
    with pytest.raises((ValueError, SystemExit)):
        ode_solver(99, NUMC, NLEV, 0.1, cc, get_rhs=_rhs_decay(0.1))


def test_ode_solver_euler_matches_direct() -> None:
    """ode_solver(solver=1) must give identical result to euler_forward()."""
    k, dt = 0.5, 0.1
    cc1 = _make_cc(val=2.0)
    cc1[:, 1:] = 2.0
    cc2 = cc1.copy()

    ode_solver(1, NUMC, NLEV, dt, cc1, get_rhs=_rhs_decay(k))
    euler_forward(dt, NUMC, NLEV, cc2, _rhs_decay(k))

    np.testing.assert_array_equal(cc1, cc2)


# ---------------------------------------------------------------------------
# 16. Edge cases — zero concentration guard
# ---------------------------------------------------------------------------


def test_euler_forward_zero_rhs_no_change() -> None:
    """With zero RHS, concentrations must be unchanged."""
    cc = _make_cc(val=3.0)
    cc[:, 1:] = 3.0
    cc_before = cc.copy()

    def zero_rhs(first: bool, numc: int, nlev: int, cc: np.ndarray) -> np.ndarray:
        return np.zeros_like(cc)

    euler_forward(0.5, NUMC, NLEV, cc, zero_rhs)
    np.testing.assert_array_equal(cc, cc_before)


def test_patankar_zero_production_zero_destruction_no_change() -> None:
    """With pp=dd=0, Patankar must leave concentrations unchanged."""
    cc = _make_cc(val=2.0)
    cc[:, 1:] = 2.0
    cc_before = cc.copy()

    def zero_ppdd(
        first: bool, numc: int, nlev: int, cc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((numc, numc, nlev + 1)), np.zeros((numc, numc, nlev + 1))

    patankar(0.5, NUMC, NLEV, cc, zero_ppdd)
    np.testing.assert_allclose(cc[:, 1:], cc_before[:, 1:], rtol=1e-14)
