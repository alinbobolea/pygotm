"""Tests for pygotm.meanflow.wequation — vertical velocity equation.

The subroutine builds a tent-shaped (piecewise-linear) vertical velocity
profile:
  - w = 0 at the seabed (zi[0]) and at the surface (zi[nlev])
  - w = w_adv at the peak height w_height
  - Linear interpolation between these anchors

Tests verify:
  - Method 0 leaves w unchanged (no-op)
  - Method 1 and 2 produce the tent profile
  - w(0) = w(nlev) = 0 for active methods
  - w_height clamping stays within 1 % margins
  - Peak location and value are correct
  - Sign of w follows sign of w_adv
  - No NaN or Inf for valid inputs
"""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.meanflow.meanflow import MeanflowState, init_meanflow, post_init_meanflow
from pygotm.meanflow.updategrid import updategrid
from pygotm.meanflow.wequation import W_ADV_NONE, W_ADV_PROFILE, wequation

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


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.meanflow.wequation import wequation as _w  # noqa: F401

    assert callable(_w)


# ---------------------------------------------------------------------------
# 2. Method 0 — no-op, w unchanged
# ---------------------------------------------------------------------------


def test_method0_noop() -> None:
    """Method 0 must leave state.w unchanged."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    sentinel = np.linspace(-0.1, 0.1, nlev + 1)
    state.w[:] = sentinel

    wequation(state, nlev, _DT, W_ADV_NONE, w_adv=0.5, w_height=_DEPTH * 0.5)

    np.testing.assert_array_equal(
        state.w, sentinel, err_msg="Method 0 must not modify w"
    )


# ---------------------------------------------------------------------------
# 3. Boundary conditions: w(0) = w(nlev) = 0 for active methods
# ---------------------------------------------------------------------------


def test_boundary_conditions_zero() -> None:
    """w at seabed and surface must be exactly zero for method 1."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    state.w[:] = 99.0

    wequation(state, nlev, _DT, W_ADV_PROFILE, w_adv=0.1, w_height=_DEPTH * 0.3)

    assert state.w[0] == 0.0, "w at seabed (k=0) must be zero"
    assert state.w[nlev] == 0.0, "w at surface (k=nlev) must be zero"


# ---------------------------------------------------------------------------
# 4. Peak location — w closest to w_height has maximum |w|
# ---------------------------------------------------------------------------


def test_peak_is_at_w_height() -> None:
    """The interior interface closest to w_height should have the largest |w|."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    assert state.zi is not None

    w_adv = 0.05
    w_height = _DEPTH * 0.4  # 40 % up from bottom

    wequation(state, nlev, _DT, W_ADV_PROFILE, w_adv=w_adv, w_height=w_height)

    # Find the interface level closest to w_height
    zi_arr = np.array(state.zi[1:nlev])
    closest = (
        int(np.argmin(np.abs(zi_arr - w_height))) + 1
    )  # offset back to original index

    w_inner = np.abs(state.w[1:nlev])
    peak_idx = int(np.argmax(w_inner)) + 1

    # Within one layer of the expected peak
    assert abs(peak_idx - closest) <= 1, (
        f"Peak at level {peak_idx} but expected near level {closest}"
    )


# ---------------------------------------------------------------------------
# 5. Sign of w follows sign of w_adv
# ---------------------------------------------------------------------------


def test_positive_w_adv_gives_positive_w() -> None:
    """All interior w must have the same sign as w_adv."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    wequation(state, nlev, _DT, W_ADV_PROFILE, w_adv=0.03, w_height=_DEPTH * 0.5)
    assert np.all(state.w[1:nlev] >= 0.0), (
        "Positive w_adv must give non-negative interior w"
    )


def test_negative_w_adv_gives_negative_w() -> None:
    """All interior w must be non-positive when w_adv < 0."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    wequation(state, nlev, _DT, W_ADV_PROFILE, w_adv=-0.03, w_height=_DEPTH * 0.5)
    assert np.all(state.w[1:nlev] <= 0.0), (
        "Negative w_adv must give non-positive interior w"
    )


# ---------------------------------------------------------------------------
# 6. Zero w_adv → all w = 0
# ---------------------------------------------------------------------------


def test_zero_w_adv_gives_zero_w() -> None:
    """w_adv = 0 must produce a zero w profile."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    state.w[:] = 99.0

    wequation(state, nlev, _DT, W_ADV_PROFILE, w_adv=0.0, w_height=_DEPTH * 0.5)

    np.testing.assert_allclose(state.w, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 7. w_height clamping — extreme heights are pulled to 1 % margin
# ---------------------------------------------------------------------------


def test_clamping_top() -> None:
    """w_height above 99 % of depth must be clamped to z_top - 1 %."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    assert state.zi is not None

    w_adv = 0.05
    # Request peak at 99.9 % of depth (above the 99 % clamp threshold).
    w_height_in = float(state.zi[0]) + 0.999 * (
        float(state.zi[nlev]) - float(state.zi[0])
    )
    returned_height = wequation(
        state, nlev, _DT, W_ADV_PROFILE, w_adv=w_adv, w_height=w_height_in
    )

    col_depth = float(state.zi[nlev]) - float(state.zi[0])
    expected_max = float(state.zi[nlev]) - 0.01 * col_depth
    assert returned_height <= expected_max + 1e-12, (
        "Returned w_height must be clamped at top"
    )


def test_clamping_bottom() -> None:
    """w_height below 1 % of depth must be clamped to z_bot + 1 %."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    assert state.zi is not None

    w_height_in = float(state.zi[0]) + 0.001 * (
        float(state.zi[nlev]) - float(state.zi[0])
    )
    returned_height = wequation(
        state, nlev, _DT, W_ADV_PROFILE, w_adv=0.05, w_height=w_height_in
    )

    col_depth = float(state.zi[nlev]) - float(state.zi[0])
    expected_min = float(state.zi[0]) + 0.01 * col_depth
    assert returned_height >= expected_min - 1e-12, (
        "Returned w_height must be clamped at bottom"
    )


# ---------------------------------------------------------------------------
# 8. Analytic check — midpoint peak
# ---------------------------------------------------------------------------


def test_analytic_midpoint_peak() -> None:
    """With w_height at z_mid, profile should be symmetric and peak near mid-column."""
    nlev = 10
    state = _make_state(nlev=nlev)
    assert state.w is not None
    assert state.zi is not None

    z_bot = float(state.zi[0])
    z_top = float(state.zi[nlev])
    z_mid = 0.5 * (z_bot + z_top)
    w_adv = 0.1

    wequation(state, nlev, _DT, W_ADV_PROFILE, w_adv=w_adv, w_height=z_mid)

    # Profile should be symmetric about mid-column; check linear increase
    # from bottom half and decrease in top half.
    for i in range(1, nlev):
        zi_i = float(state.zi[i])
        if zi_i > z_mid:
            expected = (z_top - zi_i) / (z_top - z_mid) * w_adv
        else:
            expected = (z_bot - zi_i) / (z_bot - z_mid) * w_adv
        np.testing.assert_allclose(
            state.w[i],
            expected,
            rtol=1e-12,
            err_msg=f"Mismatch at interface k={i}",
        )


# ---------------------------------------------------------------------------
# 9. Method 2 produces the same result as method 1
# ---------------------------------------------------------------------------


def test_method2_same_as_method1() -> None:
    """Methods 1 and 2 use the same tent-profile formula."""
    nlev = _NLEV
    state1 = _make_state(nlev=nlev)
    state2 = _make_state(nlev=nlev)
    assert state1.w is not None and state2.w is not None

    w_adv = 0.07
    w_height = _DEPTH * 0.35

    wequation(state1, nlev, _DT, 1, w_adv=w_adv, w_height=w_height)
    wequation(state2, nlev, _DT, 2, w_adv=w_adv, w_height=w_height)

    np.testing.assert_array_equal(
        state1.w, state2.w, err_msg="Methods 1 and 2 must give identical w profiles"
    )


# ---------------------------------------------------------------------------
# 10. Returned w_height matches clamped value used internally
# ---------------------------------------------------------------------------


def test_returned_height_matches_clamped() -> None:
    """The returned w_height must equal the internally clamped value."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.zi is not None

    # Request exactly at 50 % — no clamping expected.
    w_height_in = float(state.zi[0]) + 0.5 * (
        float(state.zi[nlev]) - float(state.zi[0])
    )
    returned = wequation(
        state, nlev, _DT, W_ADV_PROFILE, w_adv=0.1, w_height=w_height_in
    )
    np.testing.assert_allclose(returned, w_height_in, rtol=1e-12)


# ---------------------------------------------------------------------------
# 11. No NaN or Inf for a range of valid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "w_adv,frac",
    [
        (0.0, 0.5),
        (0.1, 0.1),
        (0.1, 0.9),
        (-0.05, 0.5),
        (1.0, 0.5),
    ],
)
def test_no_nan_inf(w_adv: float, frac: float) -> None:
    """No NaN or Inf in w for any valid (w_adv, w_height) combination."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    assert state.zi is not None

    z_bot = float(state.zi[0])
    z_top = float(state.zi[nlev])
    w_height = z_bot + frac * (z_top - z_bot)

    wequation(state, nlev, _DT, W_ADV_PROFILE, w_adv=w_adv, w_height=w_height)

    assert np.all(np.isfinite(state.w)), f"NaN/Inf in w for w_adv={w_adv}, frac={frac}"


# ---------------------------------------------------------------------------
# 12. Boundary levels k=0 and k=nlev always zero regardless of clamping
# ---------------------------------------------------------------------------


def test_boundary_levels_always_zero() -> None:
    """Even with clamped w_height, boundaries must remain zero."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.w is not None
    assert state.zi is not None
    state.w[:] = 99.0

    # Force clamping at top
    z_top = float(state.zi[nlev])
    wequation(state, nlev, _DT, W_ADV_PROFILE, w_adv=0.1, w_height=z_top * 2.0)

    assert state.w[0] == 0.0
    assert state.w[nlev] == 0.0
