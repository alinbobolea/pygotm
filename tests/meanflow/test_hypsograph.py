"""Tests for GOTM lake hypsography geometry."""

from pathlib import Path

import numpy as np
import pytest

from pygotm.meanflow.hypsograph import (
    HypsographState,
    read_hypsograph,
    vc2zi,
    zi2vc,
)


def _rectangular_state(depth: float, area: float) -> HypsographState:
    zi_input = np.asarray([-depth, 0.0], dtype=np.float64)
    af_input = np.asarray([area, area], dtype=np.float64)
    sqrt_af_input = np.sqrt(af_input)
    v_input = np.asarray([0.0, depth * area], dtype=np.float64)
    return HypsographState(
        nlev_input=1,
        zi_input=zi_input,
        af_input=af_input,
        sqrt_af_input=sqrt_af_input,
        v_input=v_input,
    )


def test_read_lake_erken_hypsograph_surface_first() -> None:
    state = read_hypsograph("validation/runs/lake_erken/hypsograph.dat", 21.0)

    assert state.nlev_input == 11
    assert state.zi_input[0] == pytest.approx(-21.0)
    assert state.zi_input[state.nlev_input] == pytest.approx(0.0)
    assert state.af_input[0] == pytest.approx(0.0)
    assert state.af_input[state.nlev_input] == pytest.approx(23670000.0)
    assert np.all(np.diff(state.zi_input) > 0.0)


def test_rectangular_basin_zi2vc_matches_area_times_thickness() -> None:
    nlev = 5
    depth = 10.0
    area = 42.0
    state = _rectangular_state(depth, area)
    zi = np.linspace(-depth, 0.0, nlev + 1, dtype=np.float64)
    af = np.zeros(nlev + 1, dtype=np.float64)
    vc = np.zeros(nlev + 1, dtype=np.float64)

    zi2vc(state, nlev, zi, af, vc)

    np.testing.assert_allclose(af, area, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(vc[1:], area * depth / nlev, rtol=1.0e-14)
    assert vc[0] == pytest.approx(0.0)


def test_vc2zi_round_trips_rectangular_basin() -> None:
    nlev = 8
    depth = 12.0
    area = 150.0
    state = _rectangular_state(depth, area)
    zi = np.linspace(-depth, 0.0, nlev + 1, dtype=np.float64)
    af = np.zeros(nlev + 1, dtype=np.float64)
    vc = np.zeros(nlev + 1, dtype=np.float64)
    out_zi = np.zeros(nlev + 1, dtype=np.float64)

    zi2vc(state, nlev, zi, af, vc)
    vc2zi(state, nlev, depth, vc, out_zi)

    np.testing.assert_allclose(out_zi, zi, rtol=1.0e-14, atol=1.0e-14)


def test_vc2zi_round_trips_lake_erken_geometry() -> None:
    nlev = 42
    depth = 21.0
    state = read_hypsograph("validation/runs/lake_erken/hypsograph.dat", depth)
    zi = np.linspace(-depth, 0.0, nlev + 1, dtype=np.float64)
    af = np.zeros(nlev + 1, dtype=np.float64)
    vc = np.zeros(nlev + 1, dtype=np.float64)
    out_zi = np.zeros(nlev + 1, dtype=np.float64)

    zi2vc(state, nlev, zi, af, vc)
    vc2zi(state, nlev, depth, vc, out_zi)

    np.testing.assert_allclose(out_zi, zi, rtol=1.0e-12, atol=1.0e-11)
    assert np.all(af >= 0.0)
    assert np.all(vc[1:] > 0.0)


def test_read_hypsograph_rejects_unknown_order(tmp_path: Path) -> None:
    path = tmp_path / "hypsograph.dat"
    path.write_text("2 99\n-1 1\n0 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="read order"):
        read_hypsograph(path, 1.0)
