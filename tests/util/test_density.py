"""Tests for pygotm.util.density — TEOS-10 equation of state."""

from __future__ import annotations

import gsw
import numpy as np
import pytest
from type_helpers import ReadyDensityState, require_density_state

from pygotm.util.density import (
    CP0,
    METHOD_LINEAR_TEOS10,
    METHOD_LINEAR_USER,
    METHOD_TEOS10,
    DensityState,
    clean_density,
    do_density,
    get_alpha,
    get_beta,
    get_rho,
    init_density,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_column(nlev: int, S: float, T: float, p_surface: float = 0.0) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return (S, T, p, pi) arrays for a uniform water column."""
    S_arr = np.full(nlev + 1, S)
    T_arr = np.full(nlev + 1, T)
    p_arr = np.linspace(p_surface, p_surface + nlev * 10.0, nlev + 1)
    pi_arr = np.linspace(p_surface, p_surface + nlev * 10.0, nlev + 1)
    return S_arr, T_arr, p_arr, pi_arr


def _state(method: int = METHOD_TEOS10) -> DensityState:
    state = DensityState()
    state.density_method = method
    return state


def _init_state(state: DensityState, nlev: int) -> ReadyDensityState:
    init_density(state, nlev)
    return require_density_state(state)


# ---------------------------------------------------------------------------
# 1. Import and instantiate
# ---------------------------------------------------------------------------


def test_import_and_instantiate() -> None:
    state = DensityState()
    assert state.density_method == METHOD_TEOS10
    assert state.rho is None
    assert state.alpha is None


# ---------------------------------------------------------------------------
# 2. init_density — array allocation
# ---------------------------------------------------------------------------


def test_init_density_allocates_correct_shapes() -> None:
    state = _state()
    state = _init_state(state, 20)
    assert state.rho is not None and state.rho.shape == (21,)
    assert state.rho_p is not None and state.rho_p.shape == (21,)
    assert state.alpha is not None and state.alpha.shape == (21,)
    assert state.beta is not None and state.beta.shape == (21,)


def test_init_density_method1_sets_cp() -> None:
    state = _state(METHOD_TEOS10)
    state = _init_state(state, 10)
    # TEOS-10 standard specific heat of seawater (Roquet et al. 2015)
    assert state.cp == pytest.approx(CP0, rel=1e-10)


def test_init_density_method2_derives_coeffs_from_teos10() -> None:
    state = _state(METHOD_LINEAR_TEOS10)
    state.T0 = 15.0
    state.S0 = 34.0
    state.p0 = 0.0
    state = _init_state(state, 5)

    expected_rhob = float(gsw.sigma0(34.0, 15.0)) + 1000.0
    assert state._rhob == pytest.approx(expected_rhob, rel=1e-10)
    assert state.alpha0 == pytest.approx(float(gsw.alpha(34.0, 15.0, 0.0)), rel=1e-10)
    assert state.beta0 == pytest.approx(float(gsw.beta(34.0, 15.0, 0.0)), rel=1e-10)
    assert state.cp == pytest.approx(CP0, rel=1e-10)


def test_init_density_method3_uses_rho0() -> None:
    state = _state(METHOD_LINEAR_USER)
    state.rho0 = 1025.0
    state.alpha0 = 2.0e-4
    state.beta0 = 8.0e-4
    state = _init_state(state, 5)

    assert state._rhob == 1025.0
    assert np.all(state.alpha == 2.0e-4)
    assert np.all(state.beta == 8.0e-4)
    assert np.all(state.rho == 0.0)


def test_init_density_arrays_initialised_consistently() -> None:
    state = _state(METHOD_LINEAR_TEOS10)
    state.T0 = 10.0
    state.S0 = 35.0
    state.p0 = 0.0
    state = _init_state(state, 8)

    # alpha and beta arrays are filled with alpha0/beta0 after init
    assert np.allclose(state.alpha, state.alpha0)
    assert np.allclose(state.beta, state.beta0)
    assert np.all(state.rho == 0.0)


# ---------------------------------------------------------------------------
# 3. do_density — method 1 (full TEOS-10)
# ---------------------------------------------------------------------------


def test_do_density_method1_rho_matches_gsw_directly() -> None:
    nlev = 5
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S, T, p, pi = _uniform_column(nlev, 35.0, 20.0)
    do_density(state, nlev, S, T, p, pi)

    expected = gsw.rho(S[1:], T[1:], p[1:])
    assert np.allclose(state.rho[1:], expected, rtol=1e-12)


def test_do_density_method1_rho_p_matches_sigma0() -> None:
    nlev = 4
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S = np.array([0.0, 34.0, 35.0, 35.5, 36.0])
    T = np.array([0.0, 15.0, 20.0, 22.0, 23.0])
    p = np.linspace(0.0, 40.0, nlev + 1)
    pi = np.linspace(0.0, 40.0, nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    expected = gsw.sigma0(S[1:], T[1:]) + 1000.0
    assert np.allclose(state.rho_p[1:], expected, rtol=1e-12)


def test_do_density_method1_alpha_at_interior_interfaces() -> None:
    nlev = 4
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S = np.linspace(34.0, 36.0, nlev + 1)
    T = np.linspace(10.0, 25.0, nlev + 1)
    p = np.linspace(0.0, 40.0, nlev + 1)
    pi = np.linspace(0.0, 40.0, nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    # Reconstruct expected interface S and T
    si = np.empty(nlev + 1)
    ti = np.empty(nlev + 1)
    si[1:nlev] = 0.5 * (S[1:nlev] + S[2 : nlev + 1])
    ti[1:nlev] = 0.5 * (T[1:nlev] + T[2 : nlev + 1])
    si[0] = S[0]
    si[nlev] = S[nlev]
    ti[0] = T[0]
    ti[nlev] = T[nlev]
    expected_alpha = gsw.alpha(si, ti, pi)

    assert np.allclose(state.alpha, expected_alpha, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. do_density — methods 2 and 3 (linear EOS)
# ---------------------------------------------------------------------------


def test_do_density_method2_linear_eos_formula() -> None:
    nlev = 3
    state = _state(METHOD_LINEAR_TEOS10)
    state.T0 = 15.0
    state.S0 = 34.0
    state.p0 = 0.0
    state = _init_state(state, nlev)

    S = np.array([0.0, 34.0, 35.0, 36.0])
    T = np.array([0.0, 15.0, 20.0, 10.0])
    p = np.zeros(nlev + 1)
    pi = np.zeros(nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    expected = state._rhob * (
        1.0 - state.alpha0 * (T[1:] - state.T0) + state.beta0 * (S[1:] - state.S0)
    )
    assert np.allclose(state.rho_p[1:], expected, rtol=1e-12)


def test_do_density_method3_user_coefficients() -> None:
    nlev = 2
    state = _state(METHOD_LINEAR_USER)
    state.T0 = 10.0
    state.S0 = 35.0
    state.rho0 = 1027.0
    state.alpha0 = 1.7e-4
    state.beta0 = 7.6e-4
    state = _init_state(state, nlev)

    S = np.array([35.0, 35.0, 36.0])
    T = np.array([10.0, 10.0, 15.0])
    p = np.zeros(nlev + 1)
    pi = np.zeros(nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    expected = 1027.0 * (1.0 - 1.7e-4 * (T[1:] - 10.0) + 7.6e-4 * (S[1:] - 35.0))
    assert np.allclose(state.rho_p[1:], expected, rtol=1e-12)
    assert np.allclose(state.rho[1:], state.rho_p[1:])


def test_do_density_method2_at_reference_point_matches_teos10_closely() -> None:
    """At the reference point linearised and full EOS should agree within 0.1%."""
    nlev = 1
    T_ref, S_ref = 15.0, 35.0

    state_full = _state(METHOD_TEOS10)
    state_full = _init_state(state_full, nlev)

    state_lin = _state(METHOD_LINEAR_TEOS10)
    state_lin.T0 = T_ref
    state_lin.S0 = S_ref
    state_lin.p0 = 0.0
    state_lin = _init_state(state_lin, nlev)

    S = np.array([S_ref, S_ref])
    T = np.array([T_ref, T_ref])
    p = np.array([0.0, 0.0])
    pi = np.array([0.0, 0.0])

    do_density(state_full, nlev, S, T, p, pi)
    do_density(state_lin, nlev, S, T, p, pi)

    assert np.allclose(state_full.rho[1:], state_lin.rho_p[1:], rtol=1e-3)


# ---------------------------------------------------------------------------
# 5. Boundary conditions: k=0 (seabed) and k=nlev (surface)
# ---------------------------------------------------------------------------


def test_do_density_boundary_alpha_uses_adjacent_cell_value() -> None:
    """Boundary interface α must equal α at the boundary cell, not interpolated."""
    nlev = 4
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S = np.array([28.0, 34.0, 35.0, 35.5, 36.0, 36.5])
    T = np.array([5.0, 10.0, 15.0, 20.0, 22.0, 24.0])
    p = np.linspace(0.0, 40.0, nlev + 1)
    pi = np.linspace(0.0, 40.0, nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    # k=0 bottom boundary: si[0]=S[0], ti[0]=T[0]
    assert state.alpha[0] == pytest.approx(
        float(gsw.alpha(S[0], T[0], pi[0])), rel=1e-12
    )
    # k=nlev surface boundary: si[nlev]=S[nlev], ti[nlev]=T[nlev]
    assert state.alpha[nlev] == pytest.approx(
        float(gsw.alpha(S[nlev], T[nlev], pi[nlev])), rel=1e-12
    )


def test_do_density_nlev_one_runs_without_error() -> None:
    """Single-layer column: no interior interfaces, only the two boundary faces."""
    nlev = 1
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S = np.array([35.0, 35.0])
    T = np.array([20.0, 20.0])
    p = np.array([0.0, 10.0])
    pi = np.array([0.0, 10.0])
    do_density(state, nlev, S, T, p, pi)

    assert state.rho is not None
    assert np.isfinite(state.rho[1])


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


def test_do_density_zero_salinity_gradient() -> None:
    """Uniform salinity — beta contribution should vanish in linear EOS."""
    nlev = 5
    state = _state(METHOD_LINEAR_USER)
    state.T0 = 15.0
    state.S0 = 35.0
    state.rho0 = 1027.0
    state.alpha0 = 1.7e-4
    state.beta0 = 7.6e-4
    state = _init_state(state, nlev)

    S = np.full(nlev + 1, 35.0)
    T = np.linspace(10.0, 20.0, nlev + 1)
    p = np.zeros(nlev + 1)
    pi = np.zeros(nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    expected = 1027.0 * (1.0 - 1.7e-4 * (T[1:] - 15.0))
    assert np.allclose(state.rho_p[1:], expected, rtol=1e-12)


def test_do_density_neutral_stratification_method2() -> None:
    """At the exact reference (T0, S0) point, rho_p should equal _rhob."""
    nlev = 3
    state = _state(METHOD_LINEAR_TEOS10)
    state.T0 = 10.0
    state.S0 = 35.0
    state.p0 = 0.0
    state = _init_state(state, nlev)

    S = np.full(nlev + 1, 35.0)
    T = np.full(nlev + 1, 10.0)
    p = np.zeros(nlev + 1)
    pi = np.zeros(nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    assert np.allclose(state.rho_p[1:], state._rhob, rtol=1e-12)


def test_do_density_freshwater_method1() -> None:
    """Zero salinity (freshwater lake) should return physically valid density."""
    nlev = 3
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S = np.zeros(nlev + 1)
    T = np.full(nlev + 1, 10.0)
    p = np.zeros(nlev + 1)
    pi = np.zeros(nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    # Fresh water at 10°C and 0 dbar ≈ 999.7 kg/m³
    assert np.all(state.rho[1:] > 990.0)
    assert np.all(state.rho[1:] < 1010.0)


# ---------------------------------------------------------------------------
# 7. get_rho / get_alpha / get_beta — scalar point queries
# ---------------------------------------------------------------------------


def test_get_rho_method1_with_pressure_matches_gsw() -> None:
    state = _state(METHOD_TEOS10)
    state = _init_state(state, 5)
    assert get_rho(state, 35.0, 20.0, 100.0) == pytest.approx(
        float(gsw.rho(35.0, 20.0, 100.0)), rel=1e-12
    )


def test_get_rho_method1_without_pressure_returns_potential_density() -> None:
    state = _state(METHOD_TEOS10)
    state = _init_state(state, 5)
    assert get_rho(state, 35.0, 20.0) == pytest.approx(
        float(gsw.sigma0(35.0, 20.0)) + 1000.0, rel=1e-12
    )


def test_get_rho_method3_linear_formula() -> None:
    state = _state(METHOD_LINEAR_USER)
    state.rho0 = 1027.0
    state.T0 = 10.0
    state.S0 = 35.0
    state.alpha0 = 1.7e-4
    state.beta0 = 7.6e-4
    state = _init_state(state, 5)

    S, T = 36.0, 15.0
    expected = 1027.0 * (1.0 - 1.7e-4 * (15.0 - 10.0) + 7.6e-4 * (36.0 - 35.0))
    assert get_rho(state, S, T, 0.0) == pytest.approx(expected, rel=1e-12)


def test_get_alpha_method1_matches_gsw() -> None:
    state = _state(METHOD_TEOS10)
    state = _init_state(state, 5)
    assert get_alpha(state, 35.0, 20.0, 0.0) == pytest.approx(
        float(gsw.alpha(35.0, 20.0, 0.0)), rel=1e-10
    )


def test_get_beta_method1_matches_gsw() -> None:
    state = _state(METHOD_TEOS10)
    state = _init_state(state, 5)
    assert get_beta(state, 35.0, 20.0, 0.0) == pytest.approx(
        float(gsw.beta(35.0, 20.0, 0.0)), rel=1e-10
    )


def test_get_alpha_method3_returns_alpha0() -> None:
    state = _state(METHOD_LINEAR_USER)
    state.alpha0 = 2.5e-4
    state.rho0 = 1027.0
    state = _init_state(state, 5)
    assert get_alpha(state, 35.0, 20.0, 50.0) == 2.5e-4


def test_get_beta_method2_returns_beta0() -> None:
    state = _state(METHOD_LINEAR_TEOS10)
    state.T0 = 10.0
    state.S0 = 35.0
    state.p0 = 0.0
    state = _init_state(state, 5)
    assert get_beta(state, 35.0, 10.0, 0.0) == state.beta0


# ---------------------------------------------------------------------------
# 8. Physical bounds
# ---------------------------------------------------------------------------


def test_rho_physically_reasonable_for_ocean_range() -> None:
    nlev = 10
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S = np.linspace(30.0, 40.0, nlev + 1)
    T = np.linspace(0.0, 30.0, nlev + 1)
    p = np.linspace(0.0, 200.0, nlev + 1)
    pi = np.linspace(0.0, 200.0, nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    # Ocean in-situ densities are in the range ~1020–1035 kg/m³
    assert np.all(state.rho[1:] > 1015.0)
    assert np.all(state.rho[1:] < 1045.0)


def test_alpha_positive_for_warm_saline_water() -> None:
    """Thermal expansion > 0 for T > ~4°C in saline water."""
    nlev = 5
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S, T, p, pi = _uniform_column(nlev, 35.0, 20.0)
    do_density(state, nlev, S, T, p, pi)

    assert np.all(state.alpha > 0.0)


def test_beta_positive_for_typical_seawater() -> None:
    """Haline contraction > 0: adding salt increases density."""
    nlev = 5
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S, T, p, pi = _uniform_column(nlev, 35.0, 15.0)
    do_density(state, nlev, S, T, p, pi)

    assert np.all(state.beta > 0.0)


# ---------------------------------------------------------------------------
# 9. NaN / Inf guard
# ---------------------------------------------------------------------------


def test_no_nan_or_inf_method1() -> None:
    nlev = 10
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S = np.linspace(30.0, 40.0, nlev + 1)
    T = np.linspace(-1.0, 30.0, nlev + 1)
    p = np.linspace(0.0, 500.0, nlev + 1)
    pi = np.linspace(0.0, 500.0, nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    assert np.isfinite(state.rho).all()
    assert np.isfinite(state.rho_p).all()
    assert np.isfinite(state.alpha).all()
    assert np.isfinite(state.beta).all()


def test_no_nan_or_inf_method3() -> None:
    nlev = 5
    state = _state(METHOD_LINEAR_USER)
    state.rho0 = 1027.0
    state.T0 = 10.0
    state.S0 = 35.0
    state.alpha0 = 1.7e-4
    state.beta0 = 7.6e-4
    state = _init_state(state, nlev)

    S = np.linspace(0.0, 40.0, nlev + 1)
    T = np.linspace(-2.0, 35.0, nlev + 1)
    p = np.zeros(nlev + 1)
    pi = np.zeros(nlev + 1)
    do_density(state, nlev, S, T, p, pi)

    assert np.isfinite(state.rho).all()
    assert np.isfinite(state.rho_p).all()


# ---------------------------------------------------------------------------
# 10. Published TEOS-10 reference value (validates cp0 constant)
# ---------------------------------------------------------------------------


def test_cp0_matches_teos10_standard_value() -> None:
    """TEOS-10 standard: cp0 = 3991.86795711963 J/(kg·K) (IOC 2010, Appendix K)."""
    state = _state(METHOD_TEOS10)
    state = _init_state(state, 5)
    assert state.cp == pytest.approx(CP0, rel=1e-10)


def test_rho_reference_value_against_gsw() -> None:
    """Spot-check: S=35 g/kg, CT=20°C, p=0 dbar matches gsw.rho directly."""
    nlev = 1
    state = _state(METHOD_TEOS10)
    state = _init_state(state, nlev)

    S = np.array([35.0, 35.0])
    T = np.array([20.0, 20.0])
    p = np.array([0.0, 0.0])
    pi = np.array([0.0, 0.0])
    do_density(state, nlev, S, T, p, pi)

    expected = float(gsw.rho(35.0, 20.0, 0.0))
    assert state.rho[1] == pytest.approx(expected, rel=1e-12)
    # Physically: warm Atlantic surface water ≈ 1024–1026 kg/m³
    assert 1023.0 < state.rho[1] < 1026.0


# ---------------------------------------------------------------------------
# 11. clean_density
# ---------------------------------------------------------------------------


def test_clean_density_releases_arrays() -> None:
    state = _state(METHOD_TEOS10)
    state = _init_state(state, 5)
    clean_density(state)

    assert state.alpha is None
    assert state.beta is None
    assert state.rho is None
    assert state.rho_p is None


def test_after_clean_density_reinit_works() -> None:
    state = _state(METHOD_TEOS10)
    state = _init_state(state, 5)
    clean_density(state)
    state = _init_state(state, 10)

    assert state.rho is not None and state.rho.shape == (11,)


# ---------------------------------------------------------------------------
# 12. do_density raises if not initialised
# ---------------------------------------------------------------------------


def test_do_density_raises_if_not_initialised() -> None:
    state = DensityState()
    S = np.array([35.0, 35.0])
    T = np.array([20.0, 20.0])
    p = np.zeros(2)
    pi = np.zeros(2)
    with pytest.raises(AssertionError):
        do_density(state, 1, S, T, p, pi)
