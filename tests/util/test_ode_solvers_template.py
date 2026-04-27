"""Tests for util/ode_solvers_template.py — Step 1.9 of GOTM translation plan.

The template module provides midpoint RK2 and standard RK4 (half-step intermediates),
in contrast to ode_solvers.py which uses Heun's RK2 and the non-standard full-step RK4
from ode_solvers.F90.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.util.ode_solvers import PpddCallback, RhsCallback
from pygotm.util.ode_solvers_template import (
    euler_forward,
    modified_patankar,
    ode_solver,
    patankar,
    runge_kutta_2,
    runge_kutta_4,
)

# ---------------------------------------------------------------------------
# Helpers (same as in test_ode_solvers.py)
# ---------------------------------------------------------------------------

NLEV = 4
NUMC = 2


def _rhs_decay(k: float) -> RhsCallback:
    def get_rhs(
        first: bool, numc: int, nlev: int, cc: np.ndarray
    ) -> np.ndarray:
        return -k * cc

    return get_rhs


def _ppdd_decay(k: float) -> PpddCallback:
    def get_ppdd(
        first: bool, numc: int, nlev: int, cc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pp = np.zeros((numc, numc, nlev + 1))
        dd = np.zeros((numc, numc, nlev + 1))
        for i in range(numc):
            dd[i, i, 1:] = k * cc[i, 1:]
        return pp, dd

    return get_ppdd


def _make_cc(numc: int = NUMC, nlev: int = NLEV, val: float = 1.0) -> np.ndarray:
    cc = np.full((numc, nlev + 1), val)
    cc[:, 0] = 99.0
    return cc


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.util.ode_solvers_template import ode_solver  # noqa: F401


# ---------------------------------------------------------------------------
# 2. euler_forward — shared with ode_solvers, just verify it works
# ---------------------------------------------------------------------------


def test_euler_forward_delegates() -> None:
    """Template euler_forward is identical to ode_solvers.euler_forward."""
    k, dt = 0.5, 0.1
    numc, nlev = 1, 2
    cc = _make_cc(numc, nlev, val=3.0)
    cc[:, 1:] = 3.0
    expected = 3.0 * (1 - k * dt)

    euler_forward(dt, numc, nlev, cc, _rhs_decay(k))

    np.testing.assert_allclose(cc[:, 1:], expected, rtol=1e-14)
    assert np.all(cc[:, 0] == 99.0)


# ---------------------------------------------------------------------------
# 3. runge_kutta_2 — template uses midpoint method
# ---------------------------------------------------------------------------


def test_runge_kutta_2_midpoint_analytic() -> None:
    """Midpoint RK2: c_mid = c + dt/2*f(c); c_new = c + dt*f(c_mid).

    For dc/dt = -k*c: c_new = c*(1 - k*dt + k²dt²/2), same Taylor expansion
    as Heun's to O(dt²) — verified numerically.
    """
    k = 0.5
    dt = 0.2
    numc, nlev = 1, 2
    cc = _make_cc(numc, nlev, val=3.0)
    cc[:, 1:] = 3.0

    # Midpoint: k1 = -k*c, c_mid = c + dt/2*k1, k2 = -k*c_mid
    c0 = 3.0
    c_mid = c0 + dt / 2 * (-k * c0)
    expected = c0 + dt * (-k * c_mid)

    runge_kutta_2(dt, numc, nlev, cc, _rhs_decay(k))

    np.testing.assert_allclose(cc[:, 1:], expected, rtol=1e-13)


def test_runge_kutta_2_second_order_convergence() -> None:
    """Template RK2 must show 2nd-order convergence."""
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
    assert err1 / err2 == pytest.approx(4.0, rel=0.1)


def test_runge_kutta_2_level_0_not_modified() -> None:
    cc = _make_cc()
    runge_kutta_2(0.1, NUMC, NLEV, cc, _rhs_decay(0.2))
    assert np.all(cc[:, 0] == 99.0)


# ---------------------------------------------------------------------------
# 4. runge_kutta_4 — standard RK4 with half-step intermediates
# ---------------------------------------------------------------------------


def test_runge_kutta_4_analytic_one_step() -> None:
    """Standard RK4: c_new = c*(1 - k*dt + (k*dt)^2/2 - (k*dt)^3/6 + (k*dt)^4/24)."""
    k = 1.0
    dt = 0.3
    numc, nlev = 1, 1
    cc = np.ones((numc, nlev + 1))
    cc[:, 0] = 99.0

    # Standard RK4 Taylor expansion for dc/dt = -k*c
    kdt = k * dt
    expected = 1.0 - kdt + kdt**2 / 2 - kdt**3 / 6 + kdt**4 / 24

    runge_kutta_4(dt, numc, nlev, cc, _rhs_decay(k))

    np.testing.assert_allclose(cc[0, 1], expected, rtol=1e-12)


def test_runge_kutta_4_fourth_order_convergence() -> None:
    """Standard RK4 must show 4th-order convergence.

    Halving ``dt`` should reduce the error by about 16×.
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
        for _ in range(nsteps):
            runge_kutta_4(dt, numc, nlev, cc, _rhs_decay(k))
        return float(cc[0, 1])

    err1 = abs(run(0.05) - exact)
    err2 = abs(run(0.025) - exact)
    ratio = err1 / err2
    assert ratio == pytest.approx(16.0, rel=0.1)


def test_runge_kutta_4_level_0_not_modified() -> None:
    cc = _make_cc()
    runge_kutta_4(0.1, NUMC, NLEV, cc, _rhs_decay(0.2))
    assert np.all(cc[:, 0] == 99.0)


# ---------------------------------------------------------------------------
# 5. Shared solvers — verify they delegate correctly
# ---------------------------------------------------------------------------


def test_patankar_delegates_correctly() -> None:
    """Template patankar must give the same result as ode_solvers.patankar."""
    from pygotm.util.ode_solvers import patankar as ref_patankar

    cc1 = _make_cc(val=2.0)
    cc1[:, 1:] = 2.0
    cc2 = cc1.copy()
    dt = 0.3

    patankar(dt, NUMC, NLEV, cc1, _ppdd_decay(1.5))
    ref_patankar(dt, NUMC, NLEV, cc2, _ppdd_decay(1.5))

    np.testing.assert_array_equal(cc1, cc2)


def test_modified_patankar_delegates_correctly() -> None:
    """Template modified_patankar must give the same result as ode_solvers version."""
    from pygotm.util.ode_solvers import modified_patankar as ref_mp

    cc1 = _make_cc(val=1.5)
    cc1[:, 1:] = 1.5
    cc2 = cc1.copy()
    dt = 0.2

    modified_patankar(dt, NUMC, NLEV, cc1, _ppdd_decay(2.0))
    ref_mp(dt, NUMC, NLEV, cc2, _ppdd_decay(2.0))

    np.testing.assert_array_equal(cc1, cc2)


# ---------------------------------------------------------------------------
# 6. Dispatcher
# ---------------------------------------------------------------------------


def test_ode_solver_dispatcher_euler() -> None:
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    ode_solver(1, NUMC, NLEV, 0.05, cc, get_rhs=_rhs_decay(0.2))
    assert not np.any(np.isnan(cc))
    assert np.all(cc[:, 0] == 99.0)


def test_ode_solver_dispatcher_rk4() -> None:
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    ode_solver(3, NUMC, NLEV, 0.05, cc, get_rhs=_rhs_decay(0.2))
    assert not np.any(np.isnan(cc))


def test_ode_solver_invalid_raises() -> None:
    cc = _make_cc()
    with pytest.raises((ValueError, SystemExit)):
        ode_solver(99, NUMC, NLEV, 0.1, cc, get_rhs=_rhs_decay(0.1))


# ---------------------------------------------------------------------------
# 7. NaN / Inf guard
# ---------------------------------------------------------------------------


def test_no_nan_inf_rk4() -> None:
    cc = _make_cc(val=1.0)
    cc[:, 1:] = 1.0
    runge_kutta_4(0.05, NUMC, NLEV, cc, _rhs_decay(0.5))
    assert not np.any(np.isnan(cc))
    assert not np.any(np.isinf(cc))
