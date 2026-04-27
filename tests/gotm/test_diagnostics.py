"""Tests for pygotm.gotm.diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.gotm.diagnostics import (
    DiagnosticsState,
    clean_diagnostics,
    do_diagnostics,
    init_diagnostics,
)


def test_init_and_clean_diagnostics_allocate_arrays() -> None:
    state = DiagnosticsState()
    init_diagnostics(state, 4)
    assert state.taux is not None
    assert state.tauy is not None
    clean_diagnostics(state)
    assert state.taux is None
    assert state.tauy is None


def test_do_diagnostics_computes_stress_energy_and_mld() -> None:
    state = DiagnosticsState(mld_method=2, Ri_crit=0.5)
    init_diagnostics(state, 3)
    h = np.array([0.0, 2.0, 2.0, 2.0], dtype=np.float64)
    u = np.array([0.0, 1.0, 1.5, 2.0], dtype=np.float64)
    v = np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float64)
    NN = np.array([0.0, 0.1, 0.2, 0.1], dtype=np.float64)
    SS = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    buoy = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    tke = np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float64)
    num = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    nucl = np.zeros(4, dtype=np.float64)
    drag = np.array([0.0, 0.002, 0.0, 0.0], dtype=np.float64)
    do_diagnostics(
        state,
        3,
        tx=0.5,
        ty=-0.25,
        drag=drag,
        h=h,
        u=u,
        v=v,
        NN=NN,
        SS=SS,
        buoy=buoy,
        tke=tke,
        num=num,
        nucl=nucl,
    )
    assert state.taux is not None
    assert state.tauy is not None
    assert state.taux[3] == pytest.approx(-0.5)
    assert state.tauy[3] == pytest.approx(0.25)
    assert state.mld_surf == pytest.approx(6.0)
    assert state.ekin > 0.0
    assert state.eturb > 0.0
    assert state.epot < 0.0


def test_mld_method_one_returns_zero_for_cvmix_mode() -> None:
    state = DiagnosticsState(mld_method=1)
    init_diagnostics(state, 2)
    zeros = np.zeros(3, dtype=np.float64)
    do_diagnostics(
        state,
        2,
        tx=0.0,
        ty=0.0,
        drag=zeros,
        h=np.array([0.0, 1.0, 1.0]),
        u=zeros,
        v=zeros,
        NN=zeros,
        SS=np.ones(3),
        buoy=zeros,
        tke=np.ones(3),
        num=np.ones(3),
        nucl=zeros,
        turb_method=100,
    )
    assert state.mld_surf == pytest.approx(0.0)
    assert state.mld_bott == pytest.approx(0.0)
