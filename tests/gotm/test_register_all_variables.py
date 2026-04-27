"""Tests for pygotm.gotm.register_all_variables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pygotm.gotm.diagnostics import DiagnosticsState, init_diagnostics
from pygotm.gotm.register_all_variables import (
    do_register_all_variables,
    snapshot_registry,
)


@dataclass
class _Input:
    value: float = 0.0
    data: np.ndarray | None = None
    method: int = 0


@dataclass
class _Observations:
    tprof_input: _Input
    sprof_input: _Input
    uprof_input: _Input
    vprof_input: _Input
    epsprof_input: _Input
    zeta_input: _Input
    dpdx_input: _Input
    dpdy_input: _Input


def test_register_all_variables_collects_expected_fields() -> None:
    meanflow = type(
        "Meanflow",
        (),
        {
            "z": np.array([0.0, -2.0, -1.0]),
            "zi": np.array([-3.0, -1.5, 0.0]),
            "h": np.array([0.0, 1.5, 1.5]),
            "u": np.array([0.0, 1.0, 2.0]),
            "v": np.array([0.0, 0.5, 1.5]),
            "w": np.array([0.0, 0.0, 0.0]),
            "T": np.array([0.0, 10.0, 11.0]),
            "S": np.array([0.0, 35.0, 35.2]),
            "NN": np.array([0.0, 1.0e-4, 2.0e-4]),
            "SS": np.array([0.0, 2.0e-4, 3.0e-4]),
            "buoy": np.array([0.0, 0.1, 0.2]),
            "rad": np.array([0.0, 50.0, 100.0]),
            "zeta": 0.1,
            "depth": 3.0,
            "cori": 1.0e-4,
        },
    )()
    observations = _Observations(
        tprof_input=_Input(data=np.array([0.0, 8.0, 9.0])),
        sprof_input=_Input(data=np.array([0.0, 34.0, 35.0])),
        uprof_input=_Input(data=np.array([0.0, 0.2, 0.3])),
        vprof_input=_Input(data=np.array([0.0, 0.1, 0.2])),
        epsprof_input=_Input(data=np.array([0.0, 1.0e-6, 2.0e-6]), method=1),
        zeta_input=_Input(value=0.2),
        dpdx_input=_Input(value=0.01),
        dpdy_input=_Input(value=-0.02),
    )
    diagnostics = DiagnosticsState()
    init_diagnostics(diagnostics, 2)
    diagnostics.ekin = 1.0
    diagnostics.mld_surf = 2.5
    registry = do_register_all_variables(
        55.0,
        12.0,
        2,
        observations=observations,
        diagnostics=diagnostics,
        meanflow=meanflow,
    )
    assert "lon" in registry.fields
    assert "temp_obs" in registry.fields
    assert "ekin" in registry.fields
    snapshot = snapshot_registry(registry)
    assert snapshot["lon"] == 12.0
    assert np.allclose(snapshot["temp_obs"], np.array([8.0, 9.0]))


def test_snapshot_registry_returns_copies_for_arrays() -> None:
    meanflow = type(
        "Meanflow",
        (),
        {
            "z": np.array([0.0, -1.0]),
            "zi": np.array([-2.0, 0.0]),
            "h": np.array([0.0, 2.0]),
            "u": np.array([0.0, 1.0]),
            "v": np.array([0.0, 0.0]),
            "w": np.array([0.0, 0.0]),
            "T": np.array([0.0, 10.0]),
            "S": np.array([0.0, 35.0]),
            "NN": np.array([0.0, 1.0e-4]),
            "SS": np.array([0.0, 2.0e-4]),
            "buoy": np.array([0.0, 0.1]),
            "rad": np.array([0.0, 10.0]),
            "zeta": 0.0,
            "depth": 2.0,
            "cori": 0.0,
        },
    )()
    registry = do_register_all_variables(0.0, 0.0, 1, meanflow=meanflow)
    snap = snapshot_registry(registry)
    meanflow.T[1] = 99.0
    assert np.allclose(snap["T"], np.array([0.0, 10.0]))
