"""Tests for GOTM lake water balance."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.meanflow.meanflow import MeanflowState
from pygotm.meanflow.water_balance import (
    WATER_BALANCE_ALLLAYERS,
    WATER_BALANCE_SURFACE,
    water_balance,
)
from pygotm.observations.streams import StreamsState


def _meanflow(
    *,
    lake: bool,
    method: int = WATER_BALANCE_SURFACE,
) -> MeanflowState:
    state = MeanflowState()
    state.lake = lake
    state.water_balance_method = method
    state.Af = np.asarray([1.0, 6.0, 8.0, 10.0], dtype=np.float64)
    state.Vc = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    return state


def test_lake_surface_residual_closes_net_water_balance() -> None:
    meanflow = _meanflow(lake=True, method=WATER_BALANCE_SURFACE)
    streams = StreamsState(int_inflow=100.0, int_outflow=-20.0)
    Qlayer = np.asarray([0.0, 0.5, 0.5, 1.0], dtype=np.float64)
    Qres = np.zeros(4, dtype=np.float64)

    new_zeta = water_balance(
        meanflow,
        streams,
        3,
        60.0,
        precip=0.2,
        evap=-0.1,
        Qlayer=Qlayer,
        Qres=Qres,
    )

    assert new_zeta is None
    assert meanflow.net_water_balance == pytest.approx(3.0)
    assert meanflow.int_water_balance == pytest.approx(180.0)
    assert meanflow.int_flows == pytest.approx(8.0)
    assert meanflow.int_fwf == pytest.approx(8.0)
    np.testing.assert_allclose(Qres, [0.0, 0.0, 0.0, -3.0])


def test_lake_all_layers_residual_scales_with_cell_volume() -> None:
    meanflow = _meanflow(lake=True, method=WATER_BALANCE_ALLLAYERS)
    streams = StreamsState()
    Qlayer = np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    Qres = np.zeros(4, dtype=np.float64)

    water_balance(
        meanflow,
        streams,
        3,
        1.0,
        precip=0.0,
        evap=0.0,
        Qlayer=Qlayer,
        Qres=Qres,
    )

    np.testing.assert_allclose(Qres, [0.0, -0.5, -1.0, -1.5])


def test_ocean_water_balance_tracks_surface_flux_without_residual_streams() -> None:
    meanflow = _meanflow(lake=False)
    streams = StreamsState()
    Qlayer = np.zeros(4, dtype=np.float64)
    Qres = np.ones(4, dtype=np.float64)

    water_balance(
        meanflow,
        streams,
        3,
        10.0,
        precip=0.05,
        evap=-0.02,
        Qlayer=Qlayer,
        Qres=Qres,
    )

    assert meanflow.net_water_balance == pytest.approx(0.03)
    assert meanflow.int_water_balance == pytest.approx(0.3)
    np.testing.assert_allclose(Qres, 0.0)
