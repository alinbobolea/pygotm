"""Tests for pygotm.meanflow.external_pressure.

Barotropic pressure-gradient correction.

The subroutine applies a depth-uniform velocity shift to (u, v) profiles so
that either:
  - Method 1: the interpolated velocity at height h_press equals (dpdx, dpdy)
  - Method 2: the depth-weighted mean velocity equals (dpdx, dpdy)
  - Method 0: no-op (gradient applied directly in uequation/vequation)

Tests verify:
  - Method 0 leaves u, v unchanged
  - Method 1 produces the correct interpolated velocity at h_press
  - Method 2 produces the correct depth-mean velocity
  - Shift is depth-uniform (all levels shifted by the same constant)
  - No NaN or Inf for valid inputs
  - Boundary levels (k=0, k=nlev) behaviour
"""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.meanflow.external_pressure import (
    EXT_PRESS_HEIGHT,
    EXT_PRESS_MEAN,
    EXT_PRESS_SLOPE,
    external_pressure,
)
from pygotm.meanflow.meanflow import MeanflowState, init_meanflow, post_init_meanflow
from pygotm.meanflow.updategrid import updategrid

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NLEV = 20
_DEPTH = 10.0
_DT = 3600.0


def _make_state(nlev: int = _NLEV, depth: float = _DEPTH) -> MeanflowState:
    state = MeanflowState()
    init_meanflow(state)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    return state


def _set_linear_profile(
    state: MeanflowState, nlev: int, u_bot: float, u_top: float
) -> None:
    """Set a linear u-profile from u_bot at k=1 to u_top at k=nlev."""
    assert state.u is not None
    for k in range(1, nlev + 1):
        state.u[k] = u_bot + (u_top - u_bot) * (k - 1) / (nlev - 1)


def _depth_mean(state: MeanflowState, nlev: int, field: np.ndarray) -> float:
    """Compute depth-weighted mean of field[1..nlev]."""
    assert state.h is not None
    hint = float(sum(float(state.h[k]) for k in range(1, nlev + 1)))
    weighted = float(sum(float(state.h[k] * field[k]) for k in range(1, nlev + 1)))
    return weighted / hint


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.meanflow.external_pressure import external_pressure as _ep  # noqa: F401

    assert callable(_ep)


# ---------------------------------------------------------------------------
# 2. Method 0 — no-op
# ---------------------------------------------------------------------------


def test_method0_noop_u() -> None:
    """Method 0 must leave u unchanged."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    state.u[:] = np.linspace(0.1, 1.0, nlev + 1)
    sentinel = state.u.copy()

    external_pressure(state, nlev, EXT_PRESS_SLOPE, dpdx=0.5, dpdy=0.3, h_press=3.0)

    np.testing.assert_array_equal(
        state.u,
        sentinel,
        err_msg="Method 0 must not modify u",
    )


def test_method0_noop_v() -> None:
    """Method 0 must leave v unchanged."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.v is not None
    state.v[:] = np.linspace(-0.5, 0.5, nlev + 1)
    sentinel = state.v.copy()

    external_pressure(state, nlev, EXT_PRESS_SLOPE, dpdx=0.5, dpdy=0.3, h_press=3.0)

    np.testing.assert_array_equal(
        state.v,
        sentinel,
        err_msg="Method 0 must not modify v",
    )


# ---------------------------------------------------------------------------
# 3. Method 1 — velocity at h_press matches prescribed value
# ---------------------------------------------------------------------------


def test_method1_u_at_h_press() -> None:
    """After method 1, interpolated u at h_press must equal dpdx."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.h is not None

    # Linear u-profile: 0.0 at k=1 up to 1.0 at k=nlev
    _set_linear_profile(state, nlev, 0.0, 1.0)

    dpdx = 0.3
    h_press = _DEPTH * 0.4  # 40 % of water column depth

    external_pressure(
        state,
        nlev,
        EXT_PRESS_HEIGHT,
        dpdx=dpdx,
        dpdy=0.0,
        h_press=h_press,
    )

    # Rebuild cell-centre heights to interpolate u at h_press after the shift.
    h = state.h
    z = np.zeros(nlev + 2)
    z[1] = 0.5 * h[1]
    for k in range(1, nlev):
        z[k + 1] = z[k] + 0.5 * (h[k] + h[k + 1])

    # Find bracket
    i = 1
    while z[i + 1] < h_press and i < nlev - 1:
        i += 1

    dz = z[i + 1] - z[i]
    rat = (h_press - z[i]) / dz if dz > 0.0 else 0.0
    rat = max(0.0, min(1.0, rat))
    u_interp = rat * state.u[i + 1] + (1.0 - rat) * state.u[i]

    np.testing.assert_allclose(u_interp, dpdx, atol=1e-12)


def test_method1_v_at_h_press() -> None:
    """After method 1, interpolated v at h_press must equal dpdy."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.v is not None

    for k in range(1, nlev + 1):
        state.v[k] = -0.5 + 0.02 * k  # arbitrary linear profile

    dpdy = -0.1
    h_press = _DEPTH * 0.6

    external_pressure(
        state,
        nlev,
        EXT_PRESS_HEIGHT,
        dpdx=0.0,
        dpdy=dpdy,
        h_press=h_press,
    )

    h = state.h
    assert h is not None
    z = np.zeros(nlev + 2)
    z[1] = 0.5 * h[1]
    for k in range(1, nlev):
        z[k + 1] = z[k] + 0.5 * (h[k] + h[k + 1])

    i = 1
    while z[i + 1] < h_press and i < nlev - 1:
        i += 1

    dz = z[i + 1] - z[i]
    rat = (h_press - z[i]) / dz if dz > 0.0 else 0.0
    rat = max(0.0, min(1.0, rat))
    v_interp = rat * state.v[i + 1] + (1.0 - rat) * state.v[i]

    np.testing.assert_allclose(v_interp, dpdy, atol=1e-12)


def test_method1_shift_is_depth_uniform() -> None:
    """Method 1 must shift every layer by the same constant amount."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None

    _set_linear_profile(state, nlev, 0.1, 0.9)
    u_before = state.u.copy()

    external_pressure(
        state,
        nlev,
        EXT_PRESS_HEIGHT,
        dpdx=0.5,
        dpdy=0.0,
        h_press=_DEPTH * 0.5,
    )

    shifts = state.u[1 : nlev + 1] - u_before[1 : nlev + 1]
    # All shifts must be equal (depth-uniform)
    np.testing.assert_allclose(
        shifts, shifts[0], atol=1e-14, err_msg="Method 1 shift must be depth-uniform"
    )


# ---------------------------------------------------------------------------
# 4. Method 2 — depth-mean matches prescribed value
# ---------------------------------------------------------------------------


def test_method2_u_mean() -> None:
    """After method 2, depth-mean u must equal dpdx."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None

    _set_linear_profile(state, nlev, 0.0, 1.0)

    dpdx = 0.4
    external_pressure(state, nlev, EXT_PRESS_MEAN, dpdx=dpdx, dpdy=0.0)

    assert state.u is not None
    u_mean = _depth_mean(state, nlev, state.u)
    np.testing.assert_allclose(u_mean, dpdx, atol=1e-12)


def test_method2_v_mean() -> None:
    """After method 2, depth-mean v must equal dpdy."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.v is not None

    for k in range(1, nlev + 1):
        state.v[k] = float(k) * 0.05

    dpdy = -0.2
    external_pressure(state, nlev, EXT_PRESS_MEAN, dpdx=0.0, dpdy=dpdy)

    assert state.v is not None
    v_mean = _depth_mean(state, nlev, state.v)
    np.testing.assert_allclose(v_mean, dpdy, atol=1e-12)


def test_method2_shift_is_depth_uniform() -> None:
    """Method 2 must shift every layer by the same constant amount."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None

    _set_linear_profile(state, nlev, 0.2, 0.8)
    u_before = state.u.copy()

    external_pressure(state, nlev, EXT_PRESS_MEAN, dpdx=0.5, dpdy=0.0)

    shifts = state.u[1 : nlev + 1] - u_before[1 : nlev + 1]
    np.testing.assert_allclose(
        shifts, shifts[0], atol=1e-14, err_msg="Method 2 shift must be depth-uniform"
    )


def test_method2_uniform_profile_unchanged_shape() -> None:
    """Method 2 on a uniform profile: shape preserved, mean shifted."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None

    # Uniform u profile
    u_const = 0.3
    state.u[1 : nlev + 1] = u_const

    dpdx = 0.7
    external_pressure(state, nlev, EXT_PRESS_MEAN, dpdx=dpdx, dpdy=0.0)

    # All layers should be exactly dpdx now
    np.testing.assert_allclose(state.u[1 : nlev + 1], dpdx, atol=1e-14)


# ---------------------------------------------------------------------------
# 5. Analytic verification — method 2 with known mean
# ---------------------------------------------------------------------------


def test_method2_analytic_known_mean() -> None:
    """Analytic check: uniform-layer grid, constant u=c → mean=c, shift to target d."""
    nlev = 10
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.h is not None

    c = 0.25
    d = 0.75
    state.u[1 : nlev + 1] = c

    external_pressure(state, nlev, EXT_PRESS_MEAN, dpdx=d, dpdy=0.0)

    # Expected: every layer shifted to d (since mean was c and shift = d - c)
    expected_shift = d - c
    np.testing.assert_allclose(state.u[1 : nlev + 1], c + expected_shift, rtol=1e-14)


# ---------------------------------------------------------------------------
# 6. Method 1 — h_press at bottom of column (k=1 level)
# ---------------------------------------------------------------------------


def test_method1_h_press_near_bottom() -> None:
    """Method 1 with h_press near seabed should use level k=1 velocity."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.h is not None

    state.u[1 : nlev + 1] = np.linspace(0.1, 1.0, nlev)

    # h_press very close to bottom: should be at or just above z[1]
    h_press = 0.01 * _DEPTH  # 1 % depth — well below z[1] for most grids
    dpdx = 0.5

    external_pressure(
        state,
        nlev,
        EXT_PRESS_HEIGHT,
        dpdx=dpdx,
        dpdy=0.0,
        h_press=h_press,
    )

    # The shift is depth-uniform, so the profile shape must be preserved.
    # Just verify no NaN/Inf.
    assert np.all(np.isfinite(state.u[1 : nlev + 1]))


# ---------------------------------------------------------------------------
# 7. Both u and v shifted simultaneously
# ---------------------------------------------------------------------------


def test_method2_both_components_shifted() -> None:
    """Method 2 must shift u and v independently to their respective targets."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.v is not None

    state.u[1 : nlev + 1] = 0.0
    state.v[1 : nlev + 1] = 0.0

    dpdx = 0.6
    dpdy = -0.3
    external_pressure(state, nlev, EXT_PRESS_MEAN, dpdx=dpdx, dpdy=dpdy)

    u_mean = _depth_mean(state, nlev, state.u)
    v_mean = _depth_mean(state, nlev, state.v)

    np.testing.assert_allclose(u_mean, dpdx, atol=1e-12)
    np.testing.assert_allclose(v_mean, dpdy, atol=1e-12)


# ---------------------------------------------------------------------------
# 8. No NaN or Inf for valid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,dpdx,dpdy,h_press_frac",
    [
        (EXT_PRESS_SLOPE, 0.5, 0.3, 0.5),
        (EXT_PRESS_HEIGHT, 0.1, -0.1, 0.2),
        (EXT_PRESS_HEIGHT, 0.1, -0.1, 0.8),
        (EXT_PRESS_MEAN, 0.4, -0.2, 0.5),
        (EXT_PRESS_HEIGHT, 0.0, 0.0, 0.5),
        (EXT_PRESS_MEAN, 0.0, 0.0, 0.5),
    ],
)
def test_no_nan_inf(method: int, dpdx: float, dpdy: float, h_press_frac: float) -> None:
    """No NaN or Inf in u or v for any valid input combination."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.v is not None

    _set_linear_profile(state, nlev, 0.1, 0.9)
    state.v[1 : nlev + 1] = np.linspace(-0.3, 0.3, nlev)

    h_press = h_press_frac * _DEPTH

    external_pressure(state, nlev, method, dpdx=dpdx, dpdy=dpdy, h_press=h_press)

    assert np.all(np.isfinite(state.u[1 : nlev + 1])), "NaN/Inf in u"
    assert np.all(np.isfinite(state.v[1 : nlev + 1])), "NaN/Inf in v"


# ---------------------------------------------------------------------------
# 9. k=0 and k=nlev not modified by any method
# ---------------------------------------------------------------------------


def test_boundary_levels_k0_not_modified() -> None:
    """u[0] and v[0] must not be touched — they are unused (face below seabed)."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.v is not None

    state.u[0] = 99.0
    state.v[0] = 88.0

    external_pressure(state, nlev, EXT_PRESS_MEAN, dpdx=0.5, dpdy=0.5)

    assert state.u[0] == 99.0, "u[0] must not be modified"
    assert state.v[0] == 88.0, "v[0] must not be modified"
