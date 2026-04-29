"""Tests for the Stokes drift dispatcher and state."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.stokes_drift.stokes_drift import (
    EXPONENTIAL,
    FROMUS,
    THEORYWAVE,
    StokesDriftState,
    do_stokes_drift,
    init_stokes_drift_yaml,
    langmuir_number,
    post_init_stokes_drift,
)


def _grid(nlev: int) -> tuple[np.ndarray, np.ndarray]:
    zi = np.linspace(-12.0, 0.0, nlev + 1)
    z = np.zeros(nlev + 1, dtype=np.float64)
    z[1:] = 0.5 * (zi[:-1] + zi[1:])
    return z, zi


def test_state_allocates_fortran_sized_arrays() -> None:
    state = StokesDriftState()
    post_init_stokes_drift(state, 7)

    assert state.usprof is not None
    assert state.vsprof is not None
    assert state.dusdz is not None
    assert state.usprof.shape == (8,)
    assert state.La_Turb > 1.0e10


def test_yaml_method_selection() -> None:
    state = StokesDriftState()
    init_stokes_drift_yaml(
        state,
        {
            "us": {"method": "exponential"},
            "vs": {"method": "empirical"},
            "dusdz": {"method": "us"},
            "dvsdz": {"method": "vs"},
        },
    )
    assert state.usprof_method == EXPONENTIAL
    assert state.vsprof_method == THEORYWAVE
    assert state.dusdz_method == FROMUS
    assert state.dvsdz_method == FROMUS


def test_dispatcher_computes_exponential_profile_and_shear() -> None:
    nlev = 5
    z, zi = _grid(nlev)
    state = StokesDriftState(
        usprof_method=EXPONENTIAL,
        vsprof_method=EXPONENTIAL,
        dusdz_method=FROMUS,
        dvsdz_method=FROMUS,
        us0=0.2,
        vs0=-0.1,
        ds=4.0,
    )
    post_init_stokes_drift(state, nlev)

    do_stokes_drift(state, nlev, z, zi, 9.81, 0.0, 0.0)

    assert state.usprof is not None
    assert state.dusdz is not None
    assert state.usprof[nlev] > state.usprof[1]
    assert np.isfinite(state.dusdz).all()
    assert state.dusdz[0] == pytest.approx(state.dusdz[1])
    assert state.dusdz[nlev] == pytest.approx(state.dusdz[nlev - 1])


def test_dispatcher_computes_transport_depth_for_prescribed_profile() -> None:
    nlev = 3
    z, zi = _grid(nlev)
    state = StokesDriftState(us0=0.1, vs0=0.0)
    post_init_stokes_drift(state, nlev)
    assert state.usprof is not None
    state.usprof[1:] = 0.1

    do_stokes_drift(state, nlev, z, zi, 9.81, 0.0, 0.0)

    assert state.ds == pytest.approx(abs(zi[0]))


def test_langmuir_number_outputs_finite_enhancement_factors() -> None:
    nlev = 6
    _z, zi = _grid(nlev)
    state = StokesDriftState(us0=0.12, vs0=0.0)
    post_init_stokes_drift(state, nlev)
    assert state.usprof is not None
    state.usprof[:] = np.linspace(0.02, 0.12, nlev + 1)

    langmuir_number(state, nlev, zi, hsw=1.0, u_taus=0.01, hbl=6.0, u10=8.0, v10=0.0)

    assert 0.0 < state.La_Turb < 10.0
    assert state.EFactor_LWF16 <= 2.0
    assert state.EFactor_RWH16 <= 2.25
    assert np.isfinite(state.theta_WW)
