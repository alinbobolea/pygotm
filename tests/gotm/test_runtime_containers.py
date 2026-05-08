"""Tests for compiled GOTM runtime container allocation."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.gotm.runtime_builder import (
    UnsupportedConfigurationError,
    build_runtime,
)
from pygotm.gotm.runtime_forcing import allocate_runtime_forcing
from pygotm.gotm.runtime_output import allocate_runtime_output
from pygotm.gotm.runtime_params import RuntimeParams, make_runtime_params
from pygotm.gotm.runtime_state import allocate_runtime_state
from pygotm.gotm.runtime_work import allocate_runtime_work
from pygotm.turbulence.turbulence import Constant, first_order, omega_eq, tke_keps


def _assert_float64_c_profile(array: np.ndarray, nlev: int) -> None:
    assert array.dtype == np.float64
    assert array.shape == (nlev + 1,)
    assert array.flags.c_contiguous
    assert np.all(array == 0.0)


def test_runtime_state_allocates_float64_column_profiles() -> None:
    nlev = 4
    state = allocate_runtime_state(nlev)

    names = []
    for name, array in state.iter_profile_arrays():
        names.append(name)
        _assert_float64_c_profile(array, nlev)

    assert "u" in names
    assert "T" in names
    assert "tke" in names
    assert "num" in names
    assert len(names) >= 70


def test_runtime_work_allocates_persistent_solver_arrays() -> None:
    nlev = 5
    work = allocate_runtime_work(nlev)

    names = []
    for name, array in work.iter_arrays():
        names.append(name)
        if name in {
            "vel_relax_tau",
            "vel_relax_tau_eff",
            "s_relax_tau",
            "t_relax_tau",
        }:
            assert array.dtype == np.float64
            assert array.shape == (nlev + 1,)
            assert array.flags.c_contiguous
            assert np.all(array == 1.0e15)
        else:
            _assert_float64_c_profile(array, nlev)

    assert names == [
        "au",
        "bu",
        "cu",
        "du",
        "ru",
        "qu",
        "avh",
        "q_sour",
        "l_sour",
        "sig_eff",
        "adv_cu",
        "idpdx",
        "idpdy",
        "dusdz",
        "dvsdz",
        "vel_relax_tau",
        "vel_relax_tau_eff",
        "s_relax_tau",
        "t_relax_tau",
        "uprof",
        "vprof",
        "q2l",
        "seagrass_z",
        "seagrass_exc",
        "seagrass_vfric",
        "seagrass_xx",
        "seagrass_yy",
        "seagrass_xxP",
        "seagrass_excur",
        "seagrass_grassfric",
    ]


def test_runtime_output_allocates_initial_periodic_and_final_buffers() -> None:
    output = allocate_runtime_output(nlev=3, nt=5, output_every=2)

    assert output.enabled
    assert output.nout == 4
    assert output.output_step.tolist() == [-1, -1, -1, -1]
    assert output.time.shape == (4,)
    assert np.isnan(output.time).all()
    assert output.u.shape == (4, 4)
    assert output.u.dtype == np.float64
    assert output.u.flags.c_contiguous


def test_runtime_output_disabled_allocates_empty_buffers() -> None:
    output = allocate_runtime_output(nlev=3, nt=5, enabled=False)

    assert not output.enabled
    assert output.nout == 0
    assert output.output_step.shape == (0,)
    assert output.u.shape == (0, 4)


def test_runtime_forcing_allocates_scalar_and_profile_series() -> None:
    forcing = allocate_runtime_forcing(nlev=3, nt=5)

    assert forcing.yearday.dtype == np.int64
    assert forcing.yearday.shape == (6,)
    assert forcing.time.shape == (6,)
    assert forcing.airp.dtype == np.float64
    assert forcing.h_press.shape == (6,)
    assert forcing.w_adv.shape == (6,)
    assert forcing.w_height.shape == (6,)
    assert forcing.us0.shape == (6,)
    assert forcing.dtdx.shape == (6, 4)
    assert forcing.dsdx.shape == (6, 4)
    assert forcing.us.shape == (6, 4)
    assert forcing.vs.shape == (6, 4)
    assert forcing.Tobs.shape == (6, 4)
    assert forcing.Sobs.flags.c_contiguous
    assert np.isnan(forcing.sst_obs).all()


def test_runtime_params_validate_basic_shape_controls() -> None:
    with pytest.raises(ValueError, match="nlev"):
        make_runtime_params(nlev=0, nt=1, dt=1.0)
    with pytest.raises(ValueError, match="nt"):
        make_runtime_params(nlev=1, nt=-1, dt=1.0)
    with pytest.raises(ValueError, match="dt"):
        make_runtime_params(nlev=1, nt=1, dt=0.0)


def _supported_params(
    *,
    calc_bottom_stress: int = 1,
    stokes_active: int = 0,
    w_adv_active: int = 0,
    int_press_type: int = 0,
    ext_press_mode: int = 0,
) -> RuntimeParams:
    return make_runtime_params(
        nlev=3,
        nt=1,
        dt=1.0,
        calc_bottom_stress=calc_bottom_stress,
        stokes_active=stokes_active,
        w_adv_active=w_adv_active,
        int_press_type=int_press_type,
        ext_press_mode=ext_press_mode,
        turb_method=first_order,
        tke_method=tke_keps,
        len_scale_method=omega_eq,
        stab_method=Constant,
    )


def test_runtime_builder_accepts_supported_stokes_and_external_pressure() -> None:
    bundle = build_runtime(
        _supported_params(
            stokes_active=1,
            w_adv_active=2,
            int_press_type=1,
            ext_press_mode=2,
        )
    )

    assert getattr(bundle.runner, "__name__", "") == "run_compiled_time_loop"


def test_runtime_builder_reports_bottom_stress_gate() -> None:
    params = _supported_params(calc_bottom_stress=0)

    with pytest.raises(UnsupportedConfigurationError, match="calc_bottom_stress"):
        build_runtime(params)
