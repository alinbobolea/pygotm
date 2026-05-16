"""Tests for pygotm.meanflow.updategrid — vertical grid update."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pygotm.meanflow.meanflow import (
    MeanflowState,
    init_meanflow,
    post_init_meanflow,
)
from pygotm.meanflow.updategrid import updategrid

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_NLEV = 10
_DEPTH = 25.0
_DT = 3600.0


def _make_state(
    nlev: int = _NLEV,
    depth: float = _DEPTH,
    grid_method: int = 0,
    ddu: float = 0.0,
    ddl: float = 0.0,
    grid_file: str = "",
) -> MeanflowState:
    state = MeanflowState()
    init_meanflow(state)
    state.depth = depth
    state.grid_method = grid_method
    state.ddu = ddu
    state.ddl = ddl
    state.grid_file = grid_file
    post_init_meanflow(state, nlev, latitude=0.0)
    return state


def _call(state: MeanflowState, nlev: int = _NLEV, zeta: float = 0.0) -> None:
    updategrid(state, nlev, _DT, zeta)


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.meanflow.updategrid import updategrid as _ug  # noqa: F401

    assert callable(_ug)


# ---------------------------------------------------------------------------
# 2. Smoke test — basic call completes without error
# ---------------------------------------------------------------------------


def test_smoke_method0() -> None:
    state = _make_state()
    _call(state)
    assert state.grid_ready is True


# ---------------------------------------------------------------------------
# 3. Equidistant grid (method 0, no zooming)
# ---------------------------------------------------------------------------


def test_equidistant_all_layers_equal() -> None:
    """All layer thicknesses must be depth/nlev for equidistant grid."""
    nlev = 20
    depth = 100.0
    state = _make_state(nlev=nlev, depth=depth)
    _call(state, nlev=nlev)

    assert state.h is not None
    expected_h = depth / nlev
    assert np.allclose(state.h[1:], expected_h, rtol=1e-12)


def test_equidistant_h0_zero() -> None:
    """h[0] is never set by updategrid (seabed index, always zero)."""
    state = _make_state()
    _call(state)
    assert state.h is not None
    assert state.h[0] == pytest.approx(0.0)


def test_equidistant_depth_sum() -> None:
    """Sum of all layer thicknesses must equal current depth."""
    nlev = 15
    depth = 50.0
    state = _make_state(nlev=nlev, depth=depth)
    _call(state, nlev=nlev)

    assert state.h is not None
    assert np.sum(state.h[1:]) == pytest.approx(depth, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Depth update with sea surface elevation
# ---------------------------------------------------------------------------


def test_depth_updates_with_zeta() -> None:
    """state.depth must equal depth0 + zeta after updategrid."""
    depth0 = 30.0
    zeta = 0.5
    state = _make_state(depth=depth0)
    _call(state, zeta=zeta)

    assert state.depth == pytest.approx(depth0 + zeta, rel=1e-12)


def test_depth0_unchanged_by_updategrid() -> None:
    """updategrid must not modify depth0."""
    depth0 = 30.0
    state = _make_state(depth=depth0)
    _call(state, zeta=1.0)
    assert state.depth0 == pytest.approx(depth0)


def test_depth_sum_with_positive_zeta() -> None:
    """Layer thicknesses must sum to depth0 + zeta."""
    nlev = 10
    depth0 = 20.0
    zeta = 2.0
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev, zeta=zeta)

    assert state.h is not None
    assert np.sum(state.h[1:]) == pytest.approx(depth0 + zeta, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Interface and layer-centre depths
# ---------------------------------------------------------------------------


def test_zi_seabed() -> None:
    """zi[0] must equal -depth0 (GOTM convention: seabed is at -depth0)."""
    depth0 = 20.0
    state = _make_state(depth=depth0)
    _call(state)

    assert state.zi is not None
    assert state.zi[0] == pytest.approx(-depth0, rel=1e-12)


def test_zi_surface_zeta_zero() -> None:
    """zi[nlev] == 0 when zeta == 0 (surface at mean sea level)."""
    nlev = 10
    depth0 = 20.0
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev, zeta=0.0)

    assert state.zi is not None
    assert state.zi[nlev] == pytest.approx(0.0, abs=1e-12)


def test_zi_surface_with_zeta() -> None:
    """zi[nlev] == zeta when depth = depth0 + zeta."""
    nlev = 10
    depth0 = 20.0
    zeta = 1.5
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev, zeta=zeta)

    assert state.zi is not None
    assert state.zi[nlev] == pytest.approx(zeta, rel=1e-12)


def test_zi_monotonically_increasing() -> None:
    """zi must be strictly increasing from seabed to surface."""
    nlev = 20
    state = _make_state(nlev=nlev, depth=40.0)
    _call(state, nlev=nlev)

    assert state.zi is not None
    diffs = np.diff(state.zi[: nlev + 1])
    assert np.all(diffs > 0), "zi must be monotonically increasing"


def test_z_layer_centers_equidistant() -> None:
    """For equidistant grid, z[i] = -depth0 + (i - 0.5) * h_layer."""
    nlev = 10
    depth0 = 10.0
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev)

    assert state.z is not None
    h_layer = depth0 / nlev
    for i in range(1, nlev + 1):
        expected = -depth0 + (i - 0.5) * h_layer
        assert state.z[i] == pytest.approx(
            expected, rel=1e-12
        ), f"z[{i}] = {state.z[i]}, expected {expected}"


def test_z_midpoint_of_layer() -> None:
    """z[i] must be midpoint between zi[i-1] and zi[i]."""
    nlev = 15
    state = _make_state(nlev=nlev, depth=30.0)
    _call(state, nlev=nlev)

    assert state.z is not None
    assert state.zi is not None
    assert state.h is not None
    for i in range(1, nlev + 1):
        midpoint = state.zi[i - 1] + 0.5 * state.h[i]
        assert state.z[i] == pytest.approx(midpoint, rel=1e-12)


def test_z0_not_set_by_updategrid() -> None:
    """z[0] is never written by updategrid; remains 0 from post_init."""
    state = _make_state()
    _call(state)
    assert state.z is not None
    assert state.z[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. grid_ready flag
# ---------------------------------------------------------------------------


def test_grid_ready_true_after_first_call() -> None:
    state = _make_state()
    assert state.grid_ready is False
    _call(state)
    assert state.grid_ready is True


def test_ga_not_reinitialised_on_second_call() -> None:
    """ga must not change on subsequent calls (grid initialised once)."""
    nlev = 5
    state = _make_state(nlev=nlev)
    _call(state, nlev=nlev)

    assert state.ga is not None
    ga_after_first = state.ga.copy()
    _call(state, nlev=nlev)
    np.testing.assert_array_equal(state.ga, ga_after_first)


# ---------------------------------------------------------------------------
# 7. ho tracks previous h (history tracking)
# ---------------------------------------------------------------------------


def test_ho_tracks_previous_h_on_second_call() -> None:
    """After second call, ho[i] must equal h[i] from the first call."""
    nlev = 10
    depth0 = 20.0
    state = _make_state(nlev=nlev, depth=depth0)

    # First call (zeta=0): h[1:] = depth/nlev
    _call(state, nlev=nlev, zeta=0.0)
    assert state.h is not None
    h_after_first = state.h.copy()

    # Second call (zeta=1): depth changes, ho should hold previous h
    _call(state, nlev=nlev, zeta=1.0)
    assert state.ho is not None
    np.testing.assert_allclose(state.ho[1:], h_after_first[1:], rtol=1e-12)


# ---------------------------------------------------------------------------
# 8. Zoomed grid (method 0, ddu/ddl > 0)
# ---------------------------------------------------------------------------


def test_zoomed_surface_layers_thinner() -> None:
    """With ddu > 0, surface layers must be thinner than bottom layers."""
    nlev = 20
    state = _make_state(nlev=nlev, depth=40.0, ddu=2.0, ddl=0.0)
    _call(state, nlev=nlev)

    assert state.h is not None
    # Surface layer (nlev) should be thinner than bottom layer (1)
    assert state.h[nlev] < state.h[1]


def test_zoomed_bottom_layers_thinner() -> None:
    """With ddl > 0, bottom layers must be thinner than surface layers."""
    nlev = 20
    state = _make_state(nlev=nlev, depth=40.0, ddu=0.0, ddl=2.0)
    _call(state, nlev=nlev)

    assert state.h is not None
    assert state.h[1] < state.h[nlev]


def test_zoomed_depth_sum_preserved() -> None:
    """Zoomed grid: sum of layer thicknesses must still equal depth."""
    nlev = 20
    depth = 40.0
    state = _make_state(nlev=nlev, depth=depth, ddu=2.0, ddl=2.0)
    _call(state, nlev=nlev)

    assert state.h is not None
    assert np.sum(state.h[1:]) == pytest.approx(depth, rel=1e-10)


def test_ga_range_method0() -> None:
    """For method 0, ga must span [0, 1] (sigma coordinate from 0 to 1)."""
    nlev = 10
    state = _make_state(nlev=nlev, ddu=1.0, ddl=1.0)
    _call(state, nlev=nlev)

    assert state.ga is not None
    assert state.ga[0] == pytest.approx(0.0, abs=1e-15)
    assert state.ga[nlev] == pytest.approx(1.0, rel=1e-12)
    # ga must be monotonically increasing
    assert np.all(np.diff(state.ga) >= 0)


# ---------------------------------------------------------------------------
# 9. Method 1 — external sigma from file
# ---------------------------------------------------------------------------


def test_method1_sigma_from_file() -> None:
    """Method 1: layer fractions from file produce correct thicknesses."""
    nlev = 4
    depth0 = 20.0
    fractions = [0.1, 0.2, 0.3, 0.4]  # surface first in file, sum = 1.0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(f"{nlev}\n")
        for frac in fractions:
            f.write(f"{frac}\n")
        grid_file = f.name

    try:
        state = _make_state(nlev=nlev, depth=depth0, grid_method=1, grid_file=grid_file)
        _call(state, nlev=nlev)

        assert state.h is not None
        # fractions are read surface-first: ga[4]=0.1, ga[3]=0.2, ga[2]=0.3, ga[1]=0.4
        # h[i] = ga[i] * depth0
        assert state.h[4] == pytest.approx(fractions[0] * depth0, rel=1e-12)
        assert state.h[3] == pytest.approx(fractions[1] * depth0, rel=1e-12)
        assert state.h[2] == pytest.approx(fractions[2] * depth0, rel=1e-12)
        assert state.h[1] == pytest.approx(fractions[3] * depth0, rel=1e-12)
    finally:
        Path(grid_file).unlink(missing_ok=True)


def test_method1_depth_sum() -> None:
    """Method 1: sum of layer thicknesses equals depth0."""
    nlev = 5
    depth0 = 50.0
    frac = 1.0 / nlev  # equal fractions

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(f"{nlev}\n")
        for _ in range(nlev):
            f.write(f"{frac}\n")
        grid_file = f.name

    try:
        state = _make_state(nlev=nlev, depth=depth0, grid_method=1, grid_file=grid_file)
        _call(state, nlev=nlev)

        assert state.h is not None
        assert np.sum(state.h[1:]) == pytest.approx(depth0, rel=1e-10)
    finally:
        Path(grid_file).unlink(missing_ok=True)


def test_method1_sigma_bad_sum_raises() -> None:
    """Method 1: fractions not summing to 1 must raise ValueError."""
    nlev = 3
    depth0 = 10.0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(f"{nlev}\n")
        for _ in range(nlev):
            f.write("0.4\n")  # sum = 1.2 != 1.0
        grid_file = f.name

    try:
        state = _make_state(nlev=nlev, depth=depth0, grid_method=1, grid_file=grid_file)
        with pytest.raises(ValueError, match="sigma fractions"):
            _call(state, nlev=nlev)
    finally:
        Path(grid_file).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 10. Method 2 — external Cartesian from file
# ---------------------------------------------------------------------------


def test_method2_cartesian_from_file() -> None:
    """Method 2: layer thicknesses read directly from file."""
    nlev = 4
    depth0 = 20.0
    thicknesses = [2.0, 4.0, 6.0, 8.0]  # surface first, sum = 20 = depth0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(f"{nlev}\n")
        for t in thicknesses:
            f.write(f"{t}\n")
        grid_file = f.name

    try:
        state = _make_state(nlev=nlev, depth=depth0, grid_method=2, grid_file=grid_file)
        _call(state, nlev=nlev)

        assert state.h is not None
        # thicknesses read surface-first: h[4]=2, h[3]=4, h[2]=6, h[1]=8
        assert state.h[4] == pytest.approx(thicknesses[0], rel=1e-12)
        assert state.h[3] == pytest.approx(thicknesses[1], rel=1e-12)
        assert state.h[2] == pytest.approx(thicknesses[2], rel=1e-12)
        assert state.h[1] == pytest.approx(thicknesses[3], rel=1e-12)
    finally:
        Path(grid_file).unlink(missing_ok=True)


def test_method2_ho_tracks_h() -> None:
    """Method 2: ho == h after second call (h is fixed; only ho updates)."""
    nlev = 3
    depth0 = 9.0
    thicknesses = [3.0, 3.0, 3.0]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(f"{nlev}\n")
        for t in thicknesses:
            f.write(f"{t}\n")
        grid_file = f.name

    try:
        state = _make_state(nlev=nlev, depth=depth0, grid_method=2, grid_file=grid_file)
        _call(state, nlev=nlev)
        assert state.h is not None
        h_first = state.h.copy()

        _call(state, nlev=nlev)
        assert state.ho is not None
        np.testing.assert_array_equal(state.ho[1:], h_first[1:])
    finally:
        Path(grid_file).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 11. Method 3 — adaptive grid (always raises in update phase)
# ---------------------------------------------------------------------------


def test_method3_raises_on_first_call() -> None:
    """Method 3 raises ValueError even on the first call.

    In GOTM, method 3 (adaptive grid) is managed by a separate mechanism;
    updategrid.F90 initialises the grid but its update select-case has no
    case(3), so the Fortran 'stop' is hit immediately after initialisation.
    We faithfully translate this as a ValueError.
    """
    nlev = 5
    state = _make_state(nlev=nlev, depth=10.0, grid_method=3)
    with pytest.raises(ValueError, match="grid_method"):
        updategrid(state, nlev, _DT, 0.0)


# ---------------------------------------------------------------------------
# 12. Invalid grid_method
# ---------------------------------------------------------------------------


def test_invalid_method_raises_during_init() -> None:
    """Unknown grid_method raises ValueError during initialization."""
    state = _make_state()
    state.grid_method = 99
    with pytest.raises(ValueError, match="grid_method"):
        _call(state)


def test_invalid_method_raises_after_grid_ready() -> None:
    """Unknown grid_method raises ValueError in update phase too."""
    state = _make_state()
    _call(state)  # method 0 init succeeds
    state.grid_method = 99  # corrupt after init
    with pytest.raises(ValueError, match="grid_method"):
        _call(state)


# ---------------------------------------------------------------------------
# 13. Physical bounds
# ---------------------------------------------------------------------------


def test_layer_thicknesses_positive() -> None:
    """All h[1:nlev+1] must be strictly positive after updategrid."""
    nlev = 20
    state = _make_state(nlev=nlev, depth=40.0, ddu=1.5, ddl=1.5)
    _call(state, nlev=nlev)

    assert state.h is not None
    assert np.all(state.h[1:] > 0), "all layer thicknesses must be positive"


def test_z_within_column() -> None:
    """All z[1:nlev+1] must lie within [-depth0, 0] for zeta=0."""
    nlev = 20
    depth0 = 50.0
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev)

    assert state.z is not None
    assert np.all(state.z[1:] >= -depth0)
    assert np.all(state.z[1:] <= 0.0)


def test_zi_within_column() -> None:
    """All zi[0:nlev+1] must lie within [-depth0, 0] for zeta=0."""
    nlev = 20
    depth0 = 50.0
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev)

    assert state.zi is not None
    assert np.all(state.zi[: nlev + 1] >= -depth0)
    assert np.all(state.zi[: nlev + 1] <= 0.0)


# ---------------------------------------------------------------------------
# 14. Boundary conditions: k=0 and k=nlev
# ---------------------------------------------------------------------------


def test_boundary_k0_h_zero() -> None:
    """h[0] (seabed ghost) must remain zero — updategrid only writes h[1:]."""
    state = _make_state()
    _call(state)
    assert state.h is not None
    assert state.h[0] == pytest.approx(0.0)


def test_boundary_knlev_accessible() -> None:
    """h[nlev] and zi[nlev] must be set and finite."""
    nlev = 10
    state = _make_state(nlev=nlev)
    _call(state, nlev=nlev)
    assert state.h is not None
    assert state.zi is not None
    assert math.isfinite(state.h[nlev])
    assert math.isfinite(state.zi[nlev])


# ---------------------------------------------------------------------------
# 15. Edge cases
# ---------------------------------------------------------------------------


def test_single_layer() -> None:
    """nlev=1: single layer spanning full depth."""
    nlev = 1
    depth0 = 10.0
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev)

    assert state.h is not None
    assert state.zi is not None
    assert state.z is not None
    assert state.h[1] == pytest.approx(depth0)
    assert state.zi[0] == pytest.approx(-depth0)
    assert state.zi[1] == pytest.approx(0.0, abs=1e-12)
    assert state.z[1] == pytest.approx(-0.5 * depth0)


def test_zeta_zero() -> None:
    """zeta=0 gives zi[nlev]==0 and depth==depth0."""
    nlev = 5
    depth0 = 20.0
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev, zeta=0.0)

    assert state.zi is not None
    assert state.zi[nlev] == pytest.approx(0.0, abs=1e-12)
    assert state.depth == pytest.approx(depth0)


def test_very_shallow_column() -> None:
    """Very thin column (0.1 m) must produce valid grid."""
    nlev = 5
    depth0 = 0.1
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev)

    assert state.h is not None
    assert np.all(state.h[1:] > 0)
    assert np.sum(state.h[1:]) == pytest.approx(depth0, rel=1e-10)


def test_many_layers() -> None:
    """nlev=100 equidistant grid produces equal layers."""
    nlev = 100
    depth0 = 100.0
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev)

    assert state.h is not None
    assert np.allclose(state.h[1:], depth0 / nlev, rtol=1e-12)


def test_method2_multicolumn_file() -> None:
    """Method 2: extra tab-separated columns (like asics_med grid_z.dat) are ignored.

    Fortran list-directed read(*) reads only the first value per record;
    the Python parser must match by taking split()[0].
    """
    nlev = 4
    depth0 = 20.0
    thicknesses = [2.0, 4.0, 6.0, 8.0]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(f"{nlev}\n")
        for t in thicknesses:
            # Extra columns separated by tabs — must be ignored
            f.write(f"{t}\t13.16\t38.44\n")
        grid_file = f.name

    try:
        state = _make_state(nlev=nlev, depth=depth0, grid_method=2, grid_file=grid_file)
        _call(state, nlev=nlev)

        assert state.h is not None
        assert state.h[4] == pytest.approx(thicknesses[0], rel=1e-12)
        assert state.h[3] == pytest.approx(thicknesses[1], rel=1e-12)
        assert state.h[2] == pytest.approx(thicknesses[2], rel=1e-12)
        assert state.h[1] == pytest.approx(thicknesses[3], rel=1e-12)
    finally:
        Path(grid_file).unlink(missing_ok=True)


def test_method1_multicolumn_file() -> None:
    """Method 1: extra tab-separated columns in sigma file are ignored."""
    nlev = 4
    depth0 = 20.0
    fractions = [0.1, 0.2, 0.3, 0.4]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(f"{nlev}\n")
        for frac in fractions:
            f.write(f"{frac}\textra_column\n")
        grid_file = f.name

    try:
        state = _make_state(nlev=nlev, depth=depth0, grid_method=1, grid_file=grid_file)
        _call(state, nlev=nlev)

        assert state.h is not None
        assert state.h[4] == pytest.approx(fractions[0] * depth0, rel=1e-12)
        assert state.h[1] == pytest.approx(fractions[3] * depth0, rel=1e-12)
    finally:
        Path(grid_file).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 16. NaN / Inf guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nlev", [5, 20, 100])
def test_no_nan_or_inf(nlev: int) -> None:
    state = _make_state(nlev=nlev, depth=float(nlev) * 2.0)
    _call(state, nlev=nlev)

    for name, arr in [
        ("h", state.h),
        ("ho", state.ho),
        ("z", state.z),
        ("zi", state.zi),
        ("ga", state.ga),
    ]:
        assert arr is not None
        assert not np.any(np.isnan(arr)), f"{name} contains NaN (nlev={nlev})"
        assert not np.any(np.isinf(arr)), f"{name} contains Inf (nlev={nlev})"


# ---------------------------------------------------------------------------
# 17. Analytic verification
# ---------------------------------------------------------------------------


def test_analytic_zi_equidistant() -> None:
    """Analytic check: zi[i] = -depth0 + i * h_layer for equidistant grid."""
    nlev = 10
    depth0 = 30.0
    state = _make_state(nlev=nlev, depth=depth0)
    _call(state, nlev=nlev)

    assert state.zi is not None
    h_layer = depth0 / nlev
    for i in range(nlev + 1):
        expected = -depth0 + i * h_layer
        assert state.zi[i] == pytest.approx(
            expected, rel=1e-12
        ), f"zi[{i}] = {state.zi[i]}, expected {expected}"


def test_analytic_ga_equidistant() -> None:
    """For equidistant method 0, ga[i] == i/nlev exactly."""
    nlev = 10
    state = _make_state(nlev=nlev)
    _call(state, nlev=nlev)

    assert state.ga is not None
    for i in range(nlev + 1):
        assert state.ga[i] == pytest.approx(i / nlev, rel=1e-12)
