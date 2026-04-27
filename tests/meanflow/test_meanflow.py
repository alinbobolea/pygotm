"""Tests for pygotm.meanflow.meanflow — state fields and dispatcher."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygotm.meanflow.meanflow import (
    MeanflowState,
    clean_meanflow,
    init_meanflow,
    post_init_meanflow,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_NLEV = 10
_DEPTH = 25.0
_SIDEREAL_DAY = 86164.0  # default rotation period [s]
_PI = math.pi


def _make_state(nlev: int = _NLEV, depth: float = _DEPTH) -> MeanflowState:
    """Return a fully initialised MeanflowState for testing."""
    state = MeanflowState()
    init_meanflow(state)
    state.depth = depth
    post_init_meanflow(state, nlev, latitude=0.0)
    return state


# ---------------------------------------------------------------------------
# 1. Import and instantiation
# ---------------------------------------------------------------------------


def test_import_and_instantiate() -> None:
    state = MeanflowState()
    assert state is not None


# ---------------------------------------------------------------------------
# 2. Default scalar configuration values
# ---------------------------------------------------------------------------


def test_default_config_values() -> None:
    state = MeanflowState()
    assert state.h0b == pytest.approx(0.05)
    assert state.calc_bottom_stress is True
    assert state.z0s_min == pytest.approx(0.02)
    assert state.charnock is False
    assert state.charnock_val == pytest.approx(1400.0)
    assert state.gravity == pytest.approx(9.81)
    assert state.rotation_period == pytest.approx(_SIDEREAL_DAY)
    assert state.avmolu == pytest.approx(1.3e-6)
    assert state.avmolT == pytest.approx(1.4e-7)
    assert state.avmolS == pytest.approx(1.1e-9)
    assert state.MaxItz0b == 1


def test_default_arrays_none_before_post_init() -> None:
    state = MeanflowState()
    for name in ("ga", "z", "zi", "h", "ho", "u", "v", "w", "uo", "vo",
                 "T", "S", "Tp", "Sp", "Ti", "Tobs", "Sobs",
                 "NN", "NNT", "NNS", "SS", "SSU", "SSV",
                 "SSCSTK", "SSSTK", "buoy", "rad", "xP", "avh",
                 "fric", "drag", "bioshade"):
        assert getattr(state, name) is None, f"{name} should be None before post_init"


# ---------------------------------------------------------------------------
# 3. init_meanflow applies configuration
# ---------------------------------------------------------------------------


def test_init_meanflow_default() -> None:
    state = MeanflowState()
    init_meanflow(state)
    assert state.h0b == pytest.approx(0.05)
    assert state.gravity == pytest.approx(9.81)


def test_init_meanflow_custom() -> None:
    state = MeanflowState()
    init_meanflow(
        state,
        calc_bottom_stress=False,
        h0b=0.1,
        max_it_z0b=3,
        charnock=True,
        charnock_val=800.0,
        z0s_min=0.01,
        gravity=9.80665,
        rotation_period=86400.0,
        avmolu=1.5e-6,
        avmolT=1.6e-7,
        avmolS=1.2e-9,
    )
    assert state.calc_bottom_stress is False
    assert state.h0b == pytest.approx(0.1)
    assert state.MaxItz0b == 3
    assert state.charnock is True
    assert state.charnock_val == pytest.approx(800.0)
    assert state.z0s_min == pytest.approx(0.01)
    assert state.gravity == pytest.approx(9.80665)
    assert state.rotation_period == pytest.approx(86400.0)
    assert state.avmolu == pytest.approx(1.5e-6)
    assert state.avmolT == pytest.approx(1.6e-7)
    assert state.avmolS == pytest.approx(1.2e-9)


# ---------------------------------------------------------------------------
# 4. post_init_meanflow — array shapes and initialization
# ---------------------------------------------------------------------------


def test_array_shapes() -> None:
    state = _make_state(nlev=_NLEV)
    expected_shape = (_NLEV + 1,)
    for name in ("ga", "z", "zi", "h", "ho", "u", "v", "w", "uo", "vo",
                 "T", "S", "Tp", "Sp", "Ti", "Tobs", "Sobs",
                 "NN", "NNT", "NNS", "SS", "SSU", "SSV",
                 "SSCSTK", "SSSTK", "buoy", "rad", "xP", "avh",
                 "fric", "drag", "bioshade"):
        arr = getattr(state, name)
        assert arr is not None, f"{name} must not be None after post_init"
        assert arr.shape == expected_shape, (
            f"{name}: expected shape {expected_shape}, got {arr.shape}"
        )


def test_arrays_zero_initialized() -> None:
    """All arrays except bioshade must be zero on post_init."""
    state = _make_state()
    zero_arrays = (
        "ga", "z", "zi", "h", "ho", "u", "v", "w", "uo", "vo",
        "T", "S", "Tp", "Sp", "Ti", "Tobs", "Sobs",
        "NN", "NNT", "NNS", "SS", "SSU", "SSV",
        "SSCSTK", "SSSTK", "buoy", "rad", "xP", "avh",
        "fric", "drag",
    )
    for name in zero_arrays:
        arr = getattr(state, name)
        assert arr is not None
        assert np.all(arr == 0.0), f"{name} must be zero after post_init"


def test_bioshade_initialized_to_one() -> None:
    """bioshade starts at 1 (no biological shading = full light transmission)."""
    state = _make_state()
    assert state.bioshade is not None
    assert np.all(state.bioshade == 1.0)


# ---------------------------------------------------------------------------
# 5. Coriolis parameter at known latitudes
# ---------------------------------------------------------------------------


def _expected_cori(latitude_deg: float, rotation_period: float = _SIDEREAL_DAY) -> float:
    """f = 2*Omega*sin(lat), Omega = 2*pi/T."""
    return 4.0 * _PI / rotation_period * math.sin(_PI * latitude_deg / 180.0)


@pytest.mark.parametrize("lat,expected", [
    (0.0, 0.0),
    (90.0, 4.0 * _PI / _SIDEREAL_DAY),
    (-90.0, -4.0 * _PI / _SIDEREAL_DAY),
    (45.0, _expected_cori(45.0)),
    (30.0, _expected_cori(30.0)),
    (-30.0, _expected_cori(-30.0)),
])
def test_coriolis_parameter(lat: float, expected: float) -> None:
    state = MeanflowState()
    init_meanflow(state)
    state.depth = _DEPTH
    post_init_meanflow(state, nlev=5, latitude=lat)
    assert state.cori == pytest.approx(expected, rel=1e-10)


def test_coriolis_at_45n_physical_value() -> None:
    """At 45°N with sidereal day, f ≈ 1.031e-4 rad/s."""
    state = MeanflowState()
    init_meanflow(state)
    state.depth = _DEPTH
    post_init_meanflow(state, nlev=5, latitude=45.0)
    # 2 * Omega * sin(45°) where Omega = 2*pi/86164
    omega = 2.0 * _PI / _SIDEREAL_DAY
    expected = 2.0 * omega * math.sin(_PI / 4.0)
    assert state.cori == pytest.approx(expected, rel=1e-10)
    assert abs(state.cori - 1.031e-4) < 1e-6


# ---------------------------------------------------------------------------
# 6. Roughness length initialization
# ---------------------------------------------------------------------------


def test_z0b_from_h0b() -> None:
    """z0b = 0.03 * h0b on post_init."""
    state = MeanflowState()
    init_meanflow(state, h0b=0.1)
    state.depth = _DEPTH
    post_init_meanflow(state, nlev=5, latitude=0.0)
    assert state.z0b == pytest.approx(0.03 * 0.1)


def test_z0s_equals_z0s_min() -> None:
    """z0s is set to z0s_min on post_init (lu initialization guard)."""
    state = MeanflowState()
    init_meanflow(state, z0s_min=0.005)
    state.depth = _DEPTH
    post_init_meanflow(state, nlev=5, latitude=0.0)
    assert state.z0s == pytest.approx(0.005)


def test_za_zero() -> None:
    state = _make_state()
    assert state.za == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. depth0 mirrors depth at post_init
# ---------------------------------------------------------------------------


def test_depth0_set_from_depth() -> None:
    """post_init stores depth as depth0 (initial depth)."""
    state = MeanflowState()
    init_meanflow(state)
    state.depth = 42.5
    post_init_meanflow(state, nlev=5, latitude=0.0)
    assert state.depth0 == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# 8. grid_ready is False after post_init
# ---------------------------------------------------------------------------


def test_grid_ready_false_after_post_init() -> None:
    state = _make_state()
    assert state.grid_ready is False


# ---------------------------------------------------------------------------
# 9. Scalar state reset after post_init
# ---------------------------------------------------------------------------


def test_friction_velocities_zero() -> None:
    state = _make_state()
    assert state.u_taub == pytest.approx(0.0)
    assert state.u_taubo == pytest.approx(0.0)
    assert state.u_taus == pytest.approx(0.0)
    assert state.taub == pytest.approx(0.0)


def test_runtimeu_runtimev_zero() -> None:
    state = _make_state()
    assert state.runtimeu == pytest.approx(0.0)
    assert state.runtimev == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 10. clean_meanflow resets all arrays to None
# ---------------------------------------------------------------------------


def test_clean_meanflow_sets_arrays_none() -> None:
    state = _make_state()
    clean_meanflow(state)
    for name in ("ga", "z", "zi", "h", "ho", "u", "v", "w", "uo", "vo",
                 "T", "S", "Tp", "Sp", "Ti", "Tobs", "Sobs",
                 "NN", "NNT", "NNS", "SS", "SSU", "SSV",
                 "SSCSTK", "SSSTK", "buoy", "rad", "xP", "avh",
                 "fric", "drag", "bioshade"):
        assert getattr(state, name) is None, f"{name} should be None after clean"


def test_clean_meanflow_scalars_unchanged() -> None:
    """clean_meanflow must not reset scalar config fields."""
    state = _make_state()
    state.gravity = 9.80665
    clean_meanflow(state)
    assert state.gravity == pytest.approx(9.80665)


# ---------------------------------------------------------------------------
# 11. NaN / Inf guard — no NaN or Inf in any array after post_init
# ---------------------------------------------------------------------------


def test_no_nan_or_inf_in_arrays() -> None:
    state = _make_state(nlev=100)
    for name in ("ga", "z", "zi", "h", "ho", "u", "v", "w", "uo", "vo",
                 "T", "S", "Tp", "Sp", "Ti", "Tobs", "Sobs",
                 "NN", "NNT", "NNS", "SS", "SSU", "SSV",
                 "SSCSTK", "SSSTK", "buoy", "rad", "xP", "avh",
                 "fric", "drag", "bioshade"):
        arr = getattr(state, name)
        assert arr is not None
        assert not np.any(np.isnan(arr)), f"{name} contains NaN"
        assert not np.any(np.isinf(arr)), f"{name} contains Inf"


# ---------------------------------------------------------------------------
# 12. Physical bounds on default molecular constants
# ---------------------------------------------------------------------------


def test_molecular_diffusivity_physical_bounds() -> None:
    state = MeanflowState()
    assert state.avmolu > 0.0, "molecular viscosity for momentum must be positive"
    assert state.avmolT > 0.0, "molecular diffusivity for temperature must be positive"
    assert state.avmolS > 0.0, "molecular diffusivity for salinity must be positive"
    # momentum > heat > salt (Le_S >> Le_T >> 1 in seawater)
    assert state.avmolu > state.avmolT > state.avmolS


def test_gravity_positive() -> None:
    state = MeanflowState()
    assert state.gravity > 0.0


# ---------------------------------------------------------------------------
# 13. Different nlev values (boundary conditions at k=0 and k=nlev)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nlev", [1, 5, 50, 100])
def test_array_shape_various_nlev(nlev: int) -> None:
    state = MeanflowState()
    init_meanflow(state)
    state.depth = 10.0
    post_init_meanflow(state, nlev=nlev, latitude=0.0)
    assert state.ga is not None
    assert state.ga.shape == (nlev + 1,)
    assert state.bioshade is not None
    assert state.bioshade.shape == (nlev + 1,)


def test_boundary_indices_accessible() -> None:
    """Explicitly access k=0 and k=nlev to verify boundary slots exist."""
    nlev = 10
    state = _make_state(nlev=nlev)
    assert state.u is not None
    _ = state.u[0]       # seabed
    _ = state.u[nlev]    # surface


# ---------------------------------------------------------------------------
# 14. Reinitialisation — can call post_init twice
# ---------------------------------------------------------------------------


def test_reinitialise_different_nlev() -> None:
    """Calling post_init_meanflow again must replace arrays with new size."""
    state = MeanflowState()
    init_meanflow(state)
    state.depth = _DEPTH
    post_init_meanflow(state, nlev=5, latitude=0.0)
    assert state.u is not None
    assert state.u.shape == (6,)

    state.depth = _DEPTH
    post_init_meanflow(state, nlev=20, latitude=45.0)
    assert state.u is not None
    assert state.u.shape == (21,)
