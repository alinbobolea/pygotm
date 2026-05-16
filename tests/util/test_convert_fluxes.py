"""Tests for util/convert_fluxes.py — Step 1.6 of GOTM translation plan."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.util.convert_fluxes import convert_fluxes
from pygotm.util.density import (
    METHOD_LINEAR_TEOS10,
    METHOD_LINEAR_USER,
    METHOD_TEOS10,
    DensityState,
    init_density,
)

GRAVITY = 9.81
NLEV = 10


def _make_state(method: int = METHOD_TEOS10, nlev: int = NLEV) -> DensityState:
    """Return an initialised DensityState."""
    from pygotm.util.density import CP0

    state = DensityState()
    state.density_method = method
    if method == METHOD_LINEAR_USER:
        state.rho0 = 1027.0
        state.alpha0 = 2.0e-4
        state.beta0 = 7.5e-4
        state.cp = CP0  # caller must supply cp for method 3 (no TEOS-10 call)
    init_density(state, nlev)
    return state


def _flat_rad(nlev: int, val: float = 100.0) -> np.ndarray:
    return np.full(nlev + 1, val)


# ---------------------------------------------------------------------------
# Import and smoke tests
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.util.convert_fluxes import convert_fluxes  # noqa: F401


def test_smoke_teos10() -> None:
    state = _make_state(METHOD_TEOS10)
    rad = _flat_rad(NLEV)
    result = convert_fluxes(
        state, NLEV, GRAVITY, shf=200.0, ssf=1e-5, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert len(result) == 6


def test_smoke_linear_teos10() -> None:
    state = _make_state(METHOD_LINEAR_TEOS10)
    rad = _flat_rad(NLEV)
    result = convert_fluxes(
        state, NLEV, GRAVITY, shf=200.0, ssf=1e-5, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert len(result) == 6


def test_smoke_linear_user() -> None:
    state = _make_state(METHOD_LINEAR_USER)
    rad = _flat_rad(NLEV)
    result = convert_fluxes(
        state, NLEV, GRAVITY, shf=200.0, ssf=1e-5, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert len(result) == 6


# ---------------------------------------------------------------------------
# Analytic / formula verification (METHOD_LINEAR_USER — known coefficients)
# ---------------------------------------------------------------------------


def test_t_flux_formula() -> None:
    """tFlux = -shf / (rho0 * cp) with known cp from TEOS-10 (method 2 or 3)."""
    from pygotm.util.density import CP0

    state = _make_state(METHOD_LINEAR_USER)
    shf = 400.0
    rad = _flat_rad(NLEV, 0.0)
    t_flux, _, _, _, _, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=shf, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    expected = -shf / (state.rho0 * CP0)
    assert abs(t_flux - expected) < 1e-15 * abs(expected)


def test_t_flux_formula_method2() -> None:
    """tFlux = -shf / (rho0 * cp) with known cp from TEOS-10."""
    from pygotm.util.density import CP0

    state = _make_state(METHOD_LINEAR_TEOS10)
    shf = 400.0
    rad = _flat_rad(NLEV, 0.0)
    t_flux, _, _, _, _, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=shf, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    expected = -shf / (state.rho0 * CP0)
    assert abs(t_flux - expected) < 1e-15 * abs(expected)


def test_bt_flux_formula() -> None:
    """btFlux = gravity * alpha0 * tFlux."""
    from pygotm.util.density import CP0

    state = _make_state(METHOD_LINEAR_TEOS10)
    shf = 400.0
    rad = _flat_rad(NLEV, 0.0)
    t_flux, _, bt_flux, _, _, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=shf, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    expected_t = -shf / (state.rho0 * CP0)
    assert abs(bt_flux - GRAVITY * state.alpha0 * expected_t) < 1e-12


def test_s_flux_formula() -> None:
    """sFlux = -ssf."""
    state = _make_state(METHOD_LINEAR_TEOS10)
    ssf = 2.5e-5
    rad = _flat_rad(NLEV, 0.0)
    _, s_flux, _, _, _, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=0.0, ssf=ssf, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert abs(s_flux - (-ssf)) < 1e-20


def test_bs_flux_formula() -> None:
    """bsFlux = -gravity * beta0 * sFlux."""
    state = _make_state(METHOD_LINEAR_TEOS10)
    ssf = 2.5e-5
    rad = _flat_rad(NLEV, 0.0)
    _, s_flux, _, bs_flux, _, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=0.0, ssf=ssf, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert abs(bs_flux - (-GRAVITY * state.beta0 * s_flux)) < 1e-15


def test_t_rad_formula() -> None:
    """tRad = rad / (rho0 * cp)."""
    from pygotm.util.density import CP0

    state = _make_state(METHOD_LINEAR_TEOS10)
    rad_val = 150.0
    rad = _flat_rad(NLEV, rad_val)
    _, _, _, _, t_rad, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=0.0, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    expected = rad_val / (state.rho0 * CP0)
    assert np.allclose(t_rad, expected, rtol=1e-12)


def test_b_rad_formula() -> None:
    """bRad = gravity * alpha * tRad (profile)."""

    state = _make_state(METHOD_LINEAR_TEOS10)
    rad_val = 150.0
    rad = _flat_rad(NLEV, rad_val)
    _, _, _, _, t_rad, b_rad = convert_fluxes(
        state, NLEV, GRAVITY, shf=0.0, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert state.alpha is not None
    expected = GRAVITY * state.alpha * t_rad
    assert np.allclose(b_rad, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------


def test_positive_shf_gives_negative_t_flux() -> None:
    """Downward heat flux (shf > 0) warms the ocean → positive tFlux convention
    in GOTM is tFlux = -shf/(rho0*cp), so positive shf gives negative tFlux."""
    state = _make_state(METHOD_LINEAR_TEOS10)
    rad = _flat_rad(NLEV, 0.0)
    t_flux, _, _, _, _, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=100.0, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert t_flux < 0.0


def test_positive_ssf_gives_negative_s_flux() -> None:
    """sFlux = -ssf, so positive ssf (precipitation > evaporation) gives negative sFlux."""
    state = _make_state(METHOD_LINEAR_TEOS10)
    rad = _flat_rad(NLEV, 0.0)
    _, s_flux, _, _, _, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=0.0, ssf=1e-5, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert s_flux < 0.0


# ---------------------------------------------------------------------------
# Output shapes and no NaN/Inf
# ---------------------------------------------------------------------------


def test_output_shapes() -> None:
    state = _make_state(METHOD_LINEAR_TEOS10)
    rad = _flat_rad(NLEV)
    t_flux, s_flux, bt_flux, bs_flux, t_rad, b_rad = convert_fluxes(
        state, NLEV, GRAVITY, shf=200.0, ssf=1e-5, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert np.isscalar(t_flux)
    assert np.isscalar(s_flux)
    assert np.isscalar(bt_flux)
    assert np.isscalar(bs_flux)
    assert t_rad.shape == (NLEV + 1,)
    assert b_rad.shape == (NLEV + 1,)


def test_no_nan_inf_teos10() -> None:
    state = _make_state(METHOD_TEOS10)
    rad = _flat_rad(NLEV, 50.0)
    t_flux, s_flux, bt_flux, bs_flux, t_rad, b_rad = convert_fluxes(
        state, NLEV, GRAVITY, shf=300.0, ssf=5e-6, rad=rad, T_srf=15.0, S_srf=33.0
    )
    for val in (t_flux, s_flux, bt_flux, bs_flux):
        assert np.isfinite(val)
    assert np.all(np.isfinite(t_rad))
    assert np.all(np.isfinite(b_rad))


def test_no_nan_inf_linear() -> None:
    state = _make_state(METHOD_LINEAR_TEOS10)
    rad = np.linspace(100.0, 0.0, NLEV + 1)
    t_flux, s_flux, bt_flux, bs_flux, t_rad, b_rad = convert_fluxes(
        state, NLEV, GRAVITY, shf=200.0, ssf=1e-5, rad=rad, T_srf=20.0, S_srf=35.0
    )
    for val in (t_flux, s_flux, bt_flux, bs_flux):
        assert np.isfinite(val)
    assert np.all(np.isfinite(t_rad))
    assert np.all(np.isfinite(b_rad))


# ---------------------------------------------------------------------------
# Zero-flux edge cases
# ---------------------------------------------------------------------------


def test_zero_shf() -> None:
    state = _make_state(METHOD_LINEAR_TEOS10)
    rad = _flat_rad(NLEV, 0.0)
    t_flux, _, bt_flux, _, _, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=0.0, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert t_flux == 0.0
    assert bt_flux == 0.0


def test_zero_ssf() -> None:
    state = _make_state(METHOD_LINEAR_TEOS10)
    rad = _flat_rad(NLEV, 0.0)
    _, s_flux, _, bs_flux, _, _ = convert_fluxes(
        state, NLEV, GRAVITY, shf=0.0, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert s_flux == 0.0
    assert bs_flux == 0.0


def test_zero_rad() -> None:
    state = _make_state(METHOD_LINEAR_TEOS10)
    rad = _flat_rad(NLEV, 0.0)
    _, _, _, _, t_rad, b_rad = convert_fluxes(
        state, NLEV, GRAVITY, shf=0.0, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert np.all(t_rad == 0.0)
    assert np.all(b_rad == 0.0)


# ---------------------------------------------------------------------------
# Boundary levels k=0 and k=nlev
# ---------------------------------------------------------------------------


def test_boundary_levels_rad_profile() -> None:
    """Ensure t_rad and b_rad at k=0 and k=nlev are computed correctly."""
    state = _make_state(METHOD_LINEAR_TEOS10)
    from pygotm.util.density import CP0

    rad = np.zeros(NLEV + 1)
    rad[0] = 50.0
    rad[NLEV] = 80.0
    _, _, _, _, t_rad, b_rad = convert_fluxes(
        state, NLEV, GRAVITY, shf=0.0, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
    )
    assert np.isclose(t_rad[0], rad[0] / (state.rho0 * CP0), rtol=1e-12)
    assert np.isclose(t_rad[NLEV], rad[NLEV] / (state.rho0 * CP0), rtol=1e-12)
    assert state.alpha is not None
    assert np.isclose(b_rad[0], GRAVITY * state.alpha[0] * t_rad[0], rtol=1e-12)
    assert np.isclose(
        b_rad[NLEV], GRAVITY * state.alpha[NLEV] * t_rad[NLEV], rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Physical bounds
# ---------------------------------------------------------------------------


def test_buoyancy_fluxes_finite_realistic() -> None:
    """Typical ocean surface forcing should give physically reasonable fluxes."""
    state = _make_state(METHOD_TEOS10)
    rad = np.exp(-np.linspace(0, 5, NLEV + 1)) * 200.0  # decaying SW
    t_flux, s_flux, bt_flux, bs_flux, t_rad, b_rad = convert_fluxes(
        state, NLEV, GRAVITY, shf=150.0, ssf=1e-5, rad=rad, T_srf=25.0, S_srf=35.0
    )
    # Buoyancy flux magnitudes should be small (~O(1e-8) to O(1e-5)) m²/s³
    assert abs(bt_flux) < 1e-3
    assert abs(bs_flux) < 1e-3
    assert np.all(np.abs(b_rad) < 1e-3)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_assert_before_init() -> None:
    """convert_fluxes requires init_density to have been called."""
    state = DensityState()
    rad = _flat_rad(NLEV)
    with pytest.raises(AssertionError):
        convert_fluxes(
            state, NLEV, GRAVITY, shf=100.0, ssf=0.0, rad=rad, T_srf=20.0, S_srf=35.0
        )
