"""Tests for the compiled central GOTM time-loop."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gsw
import numpy as np
import pytest

from pygotm.gotm.gotm import (
    _integrate_gotm_python,
    finalize_gotm,
    initialize_gotm,
    integrate_gotm_compiled,
)
from pygotm.gotm.runtime_builder import (
    _copy_absolute_salinity_from_practical,
    _make_salinity_conversion_cache,
    build_runtime,
    build_runtime_from_run,
    runtime_output_to_dataset,
)
from pygotm.gotm.runtime_params import make_runtime_params
from pygotm.gotm.runtime_state import RuntimeState
from pygotm.gotm.time_loop import (
    _compute_mld_single,
    _gsw_t_from_ct_surface_compiled,
    run_compiled_time_loop,
    time_loop_compiled,
    warmup_couette_step_routines,
)
from pygotm.meanflow.friction import step_friction_single
from pygotm.meanflow.shear import step_shear_single
from pygotm.meanflow.uequation import step_uequation_single
from pygotm.meanflow.vequation import step_vequation_single
from pygotm.turbulence.alpha_mnb import step_alpha_mnb_single
from pygotm.turbulence.cmue_c import step_cmue_c_single
from pygotm.turbulence.epsbalgebraic import step_epsbalgebraic_single
from pygotm.turbulence.kbalgebraic import step_kbalgebraic_single
from pygotm.turbulence.omegaeq import step_omegaeq_single
from pygotm.turbulence.production import step_production_single
from pygotm.turbulence.tkeeq import step_tkeeq_single
from pygotm.turbulence.turbulence import Constant, first_order, omega_eq, tke_keps
from pygotm.util.gsw import gsw_sa_from_sp, gsw_sp_from_sa
from pygotm.validation.reference import open_reference_dataset
from tests.dataset_assertions import assert_dataset_variables_allclose
from tests.fixtures import bundled_case, bundled_case_path

_COUETTE_CONFIG = bundled_case_path("couette")
_SEAGRASS_CONFIG = bundled_case_path("seagrass")
_WAVE_BREAKING_CONFIG = bundled_case_path("wave_breaking")
_AIRSEA_FIRST_SLOT_VARIABLES = (
    "es",
    "ea",
    "qs",
    "qa",
    "rhoa",
    "qh",
    "qe",
    "ql",
    "heat",
    "tx",
    "ty",
    "sst",
    "sst_obs",
    "sss",
    "I_0",
    "albedo",
)


def _write_vertical_advection_config(path: Path) -> None:
    path.write_text(
        """
version: 7
location:
  latitude: 55.0
  longitude: 12.0
  depth: 10.0
time:
  start: 2000-01-01 00:00:00
  stop: 2000-01-01 00:01:00
  dt: 60.0
grid:
  nlev: 8
temperature:
  method: off
salinity:
  method: off
mimic_3d:
  w:
    max:
      method: constant
      constant_value: 1.0e-4
    height:
      method: constant
      constant_value: -5.0
""".lstrip(),
        encoding="utf-8",
    )


def test_cached_practical_salinity_conversion_matches_gsw() -> None:
    practical = np.asarray([34.0, 35.0, 36.0], dtype=np.float64)
    pressure = np.asarray([0.0, 50.0, 100.0], dtype=np.float64)
    target = np.zeros_like(practical)
    cache = _make_salinity_conversion_cache(
        practical.size,
        longitude=14.5,
        latitude=43.0,
    )

    _copy_absolute_salinity_from_practical(
        practical,
        pressure,
        14.5,
        43.0,
        target,
        cache,
    )
    expected = np.asarray(gsw_sa_from_sp(practical, pressure, 14.5, 43.0))
    np.testing.assert_array_equal(target, expected)

    updated_practical = practical + 0.25
    _copy_absolute_salinity_from_practical(
        updated_practical,
        pressure,
        14.5,
        43.0,
        target,
        cache,
    )
    expected = np.asarray(gsw_sa_from_sp(updated_practical, pressure, 14.5, 43.0))
    np.testing.assert_array_equal(target, expected)


def test_cached_practical_salinity_conversion_matches_gsw_for_baltic() -> None:
    practical = np.asarray([4.0, 7.0, 10.0], dtype=np.float64)
    pressure = np.asarray([0.0, 10.0, 20.0], dtype=np.float64)
    target = np.zeros_like(practical)
    cache = _make_salinity_conversion_cache(
        practical.size,
        longitude=20.0,
        latitude=58.0,
    )

    _copy_absolute_salinity_from_practical(
        practical,
        pressure,
        20.0,
        58.0,
        target,
        cache,
    )
    expected = np.asarray(gsw_sa_from_sp(practical, pressure, 20.0, 58.0))
    np.testing.assert_array_equal(target, expected)


def test_runtime_dataset_derives_teos10_temperature_and_salinity_outputs() -> None:
    """TEOS-10 diagnostic outputs must follow saved T/S/z, not stale buffers."""

    params = make_runtime_params(
        nlev=2,
        nt=1,
        dt=3600.0,
        calc_bottom_stress=1,
        density_method=1,
        turb_method=first_order,
        tke_method=tke_keps,
        len_scale_method=omega_eq,
        stab_method=Constant,
    )
    runtime = build_runtime(params, output=True, output_every=1)
    runtime.output.output_step[:] = [0, 1]
    runtime.output.time[:] = [0.0, 3600.0]
    runtime.output.T[:, 1:] = [[10.0, 12.0], [14.0, 16.0]]
    runtime.output.S[:, 1:] = [[18.0, 19.0], [20.0, 21.0]]
    runtime.output.z[:, 1:] = [[-5.0, -1.0], [-6.0, -2.0]]
    runtime.output.Tp[:, 1:] = -999.0
    runtime.output.Ti[:, 1:] = -999.0
    runtime.output.Sp[:, 1:] = -999.0
    run = SimpleNamespace(
        time=SimpleNamespace(start="2000-01-01 00:00:00"),
        output_schedule=SimpleNamespace(k_start=1, k1_start=1),
        latitude=43.177,
        longitude=32.625,
        settings=SimpleNamespace(title="test"),
        yaml_path=Path("gotm.yaml"),
        observations=SimpleNamespace(epsprof_input=None),
        document={},
    )

    dataset = runtime_output_to_dataset(run, runtime)

    temp = dataset["temp"].values[:, :, 0, 0]
    salt = dataset["salt"].values[:, :, 0, 0]
    pressure = -dataset["z"].values[:, :, 0, 0]
    np.testing.assert_allclose(
        dataset["temp_p"].values[:, :, 0, 0],
        gsw.pt_from_CT(salt, temp),
    )
    np.testing.assert_allclose(
        dataset["temp_i"].values[:, :, 0, 0],
        gsw.t_from_CT(salt, temp, pressure),
    )
    np.testing.assert_allclose(
        dataset["salt_p"].values[:, :, 0, 0],
        gsw_sp_from_sa(salt, pressure, run.longitude, run.latitude),
    )


def _seed_state(state: RuntimeState) -> None:
    values = np.arange(state.nlev + 1, dtype=np.float64)
    state.u[:] = values
    state.v[:] = values + 10.0
    state.T[:] = values + 20.0
    state.S[:] = values + 30.0
    state.tke[:] = values + 40.0
    state.eps[:] = values + 50.0
    state.num[:] = values + 60.0
    state.nuh[:] = values + 70.0
    state.h[:] = values + 80.0
    state.SS[:] = values + 90.0
    state.P[:] = values + 100.0
    state.kb[:] = values + 110.0
    state.z[:] = -values
    state.zi[:] = -values - 0.5


def test_time_loop_file_does_not_use_postponed_annotations() -> None:
    source = Path("src/pygotm/gotm/time_loop.py").read_text(encoding="utf-8")

    assert "from __future__ import annotations" not in source


def test_compiled_initial_output_preserves_derived_profiles() -> None:
    params = make_runtime_params(
        nlev=3,
        nt=0,
        dt=1.0,
        calc_bottom_stress=1,
        turb_method=first_order,
        tke_method=tke_keps,
        len_scale_method=omega_eq,
        stab_method=Constant,
    )
    runtime = build_runtime(params, output=True)
    values = np.arange(params.nlev + 1, dtype=np.float64)
    runtime.state.rho_p[:] = values + 1000.0
    runtime.state.rho[:] = values + 2000.0
    runtime.state.Tp[:] = values + 10.0
    runtime.state.Ti[:] = values + 20.0
    runtime.state.Sp[:] = values + 30.0

    written = run_compiled_time_loop(
        runtime.params,
        runtime.state,
        runtime.work,
        runtime.forcing,
        runtime.output,
    )

    assert written == 1
    np.testing.assert_array_equal(runtime.output.rho_p[0], runtime.state.rho_p)
    np.testing.assert_array_equal(runtime.output.rho[0], runtime.state.rho)
    np.testing.assert_array_equal(runtime.output.Tp[0], runtime.state.Tp)
    np.testing.assert_array_equal(runtime.output.Ti[0], runtime.state.Ti)
    np.testing.assert_array_equal(runtime.output.Sp[0], runtime.state.Sp)


def test_compute_mld_single_matches_critical_ri_depth() -> None:
    nlev = 4
    h = np.zeros(nlev + 1, dtype=np.float64)
    nn = np.zeros(nlev + 1, dtype=np.float64)
    ss = np.zeros(nlev + 1, dtype=np.float64)
    tke = np.zeros(nlev + 1, dtype=np.float64)
    h[1:] = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    nn[2] = 0.6
    ss[2] = 1.0
    nn[3] = 0.1
    ss[3] = 1.0

    mld_surf, mld_bott = _compute_mld_single(nlev, 2, 1.0e-5, 0.5, 0, h, nn, ss, tke)

    assert mld_surf == pytest.approx(7.0)
    assert mld_bott == pytest.approx(0.0)


def test_surface_teos_sst_conversion_matches_gsw() -> None:
    import gsw

    for salinity, conservative_temperature in (
        (21.80334896153364, 9.071539544191076),
        (32.889328536942, 5.375383827497782),
        (35.0, 10.0),
    ):
        expected = float(gsw.t_from_CT(salinity, conservative_temperature, 0.0))
        actual = _gsw_t_from_ct_surface_compiled(
            salinity,
            conservative_temperature,
        )

        assert actual == pytest.approx(expected, abs=1.0e-8)


def test_compiled_initial_output_preserves_zero_diagnostics_and_writes_stokes() -> None:
    params = make_runtime_params(
        nlev=4,
        nt=0,
        dt=1.0,
        calc_bottom_stress=1,
        turb_method=first_order,
        tke_method=tke_keps,
        len_scale_method=omega_eq,
        stab_method=Constant,
        mld_method=2,
        mld_ri_crit=0.5,
    )
    runtime = build_runtime(params, output=True)
    runtime.forcing.us0[0] = 0.12
    runtime.forcing.vs0[0] = -0.03
    runtime.forcing.ds[0] = 5.0
    runtime.forcing.us[0] = np.linspace(0.0, 0.12, params.nlev + 1)
    runtime.forcing.vs[0] = np.linspace(0.0, -0.03, params.nlev + 1)

    written = run_compiled_time_loop(
        runtime.params,
        runtime.state,
        runtime.work,
        runtime.forcing,
        runtime.output,
    )

    assert written == 1
    assert runtime.output.mld_surf[0] == pytest.approx(0.0)
    assert runtime.output.mld_bott[0] == pytest.approx(0.0)
    assert np.all(runtime.output.Rig[0] == 0.0)
    assert runtime.output.us0[0] == pytest.approx(0.12)
    assert runtime.output.vs0[0] == pytest.approx(-0.03)
    assert runtime.output.ds[0] == pytest.approx(5.0)
    np.testing.assert_allclose(runtime.output.us[0], runtime.forcing.us[0])
    np.testing.assert_allclose(runtime.output.vs[0], runtime.forcing.vs[0])


def test_compiled_initial_output_populates_simple_ice_tf_reference() -> None:
    params = make_runtime_params(
        nlev=3,
        nt=0,
        dt=1.0,
        calc_bottom_stress=1,
        turb_method=first_order,
        tke_method=tke_keps,
        len_scale_method=omega_eq,
        stab_method=Constant,
    )
    runtime = build_runtime(params, output=True)
    runtime.state.S[params.nlev] = 32.8

    written = run_compiled_time_loop(
        runtime.params,
        runtime.state,
        runtime.work,
        runtime.forcing,
        runtime.output,
    )

    assert written == 1
    assert runtime.output.reference_scalars["Tf"][0] == pytest.approx(-0.0575 * 32.8)


def test_runtime_forcing_precomputes_vertical_advection_series(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_vertical_advection_config(config_path)
    run = initialize_gotm(config_path)
    try:
        runtime = integrate_gotm_compiled(run, max_steps=1, output=False)
    finally:
        finalize_gotm(run)

    assert runtime.params.w_adv_active != 0
    assert np.any(runtime.forcing.w_adv != 0.0)
    assert np.any(runtime.forcing.w_height != 0.0)
    assert np.any(runtime.state.w[1 : runtime.params.nlev] != 0.0)


def test_compiled_zeta_forcing_matches_reference_initial_slot() -> None:
    case = bundled_case("seagrass")
    run = initialize_gotm(_SEAGRASS_CONFIG)
    reference = open_reference_dataset(case)
    actual = None
    try:
        runtime = integrate_gotm_compiled(run, max_steps=1, output=True)
        actual = runtime_output_to_dataset(run, runtime)

        assert runtime.params.zeta_input_active == 1
        np.testing.assert_allclose(
            actual["zeta"].values[0],
            reference["zeta"].values.squeeze()[0],
            rtol=5.0e-6,
            atol=1.0e-12,
        )
        assert "h" in actual.data_vars
        assert np.isfinite(actual["h"].values).all()
        for name in ("z", "zi"):
            assert name in actual.coords
            assert np.isfinite(actual.coords[name].values).all()
    finally:
        if actual is not None:
            actual.close()
        reference.close()
        finalize_gotm(run)


def test_compiled_first_order_turbulence_emits_variances() -> None:
    run = initialize_gotm(_WAVE_BREAKING_CONFIG)
    try:
        runtime = integrate_gotm_compiled(run, max_steps=1, output=True)
    finally:
        finalize_gotm(run)

    assert runtime.params.turb_method == first_order
    assert np.any(runtime.state.uu != 0.0)
    assert np.any(runtime.state.vv != 0.0)
    assert np.any(runtime.state.ww != 0.0)
    assert np.any(runtime.state.at != 0.0)


def test_major_couette_step_routines_have_nopython_signatures() -> None:
    run = initialize_gotm(_COUETTE_CONFIG)
    try:
        runtime = build_runtime_from_run(run, max_steps=1, output=False)
        warmup_couette_step_routines(runtime.params, runtime.state, runtime.work)
    finally:
        finalize_gotm(run)

    for step_routine in (
        step_uequation_single,
        step_vequation_single,
        step_friction_single,
        step_shear_single,
        step_production_single,
        step_alpha_mnb_single,
        step_cmue_c_single,
        step_tkeeq_single,
        step_kbalgebraic_single,
        step_omegaeq_single,
        step_epsbalgebraic_single,
    ):
        assert step_routine.nopython_signatures, step_routine


def test_compiled_couette_two_steps_matches_python_path(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    config_text = _COUETTE_CONFIG.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "stop: 2005-01-02 00:00:00", "stop: 2005-01-01 00:00:20", 1
    )
    config_text = config_text.replace("nlev: 100", "nlev: 8", 1)
    config_path.write_text(config_text, encoding="utf-8")

    compiled_run = initialize_gotm(config_path)
    python_run = initialize_gotm(config_path)
    try:
        runtime = integrate_gotm_compiled(compiled_run, max_steps=2, output=False)
        _integrate_gotm_python(python_run, max_steps=2)

        assert runtime.output.nout == 0
        for name in (
            "u",
            "uo",
            "v",
            "vo",
            "SS",
            "SSU",
            "SSV",
            "tke",
            "tkeo",
            "eps",
            "omega",
            "L",
            "kb",
            "epsb",
            "P",
            "B",
            "Pb",
            "Px",
            "PSTK",
            "num",
            "nuh",
            "nus",
            "nucl",
            "as_",
            "an",
            "at",
            "cmue1",
            "cmue2",
            "cmue3",
        ):
            expected_owner = (
                python_run.turbulence
                if hasattr(python_run.turbulence, name)
                else python_run.meanflow
            )
            expected = getattr(expected_owner, name)
            actual = getattr(runtime.state, name)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)
            copied_owner = (
                compiled_run.turbulence
                if hasattr(compiled_run.turbulence, name)
                else compiled_run.meanflow
            )
            copied = getattr(copied_owner, name)
            np.testing.assert_allclose(copied, expected, rtol=0.0, atol=1.0e-12)

        assert runtime.state.u_taub[0] == python_run.meanflow.u_taub
        assert runtime.state.u_taus[0] == python_run.meanflow.u_taus
        assert compiled_run.meanflow.u_taub == python_run.meanflow.u_taub
        assert compiled_run.meanflow.u_taus == python_run.meanflow.u_taus
        assert time_loop_compiled.nopython_signatures
    finally:
        finalize_gotm(compiled_run)
        finalize_gotm(python_run)


@pytest.mark.parametrize("case_name", ("asics_med",))
def test_compiled_airsea_first_slot_matches_fortran(case_name: str) -> None:
    case = bundled_case(case_name)
    compiled_run = initialize_gotm(case.yaml_path)
    if compiled_run.fabm_config is not None and compiled_run.fabm_config.use:
        finalize_gotm(compiled_run)
        pytest.skip(
            "FABM cases need a separate IC-slot test; airsea-only comparison skipped"
        )

    reference = open_reference_dataset(case)
    actual = None
    try:
        runtime = integrate_gotm_compiled(compiled_run, max_steps=0, output=True)
        actual = runtime_output_to_dataset(compiled_run, runtime)

        assert actual.sizes["time"] == 1
        for variable in _AIRSEA_FIRST_SLOT_VARIABLES:
            assert variable in actual.data_vars
            assert variable in reference.data_vars
            actual_value = float(np.asarray(actual[variable].isel(time=0)).squeeze())
            expected_value = float(
                np.asarray(reference[variable].isel(time=0)).squeeze()
            )
            assert actual_value == pytest.approx(expected_value, abs=1.0e-4)
    finally:
        if actual is not None:
            actual.close()
        reference.close()
        finalize_gotm(compiled_run)


@pytest.mark.parametrize("case_name", ("couette", "channel"))
def test_compiled_reference_emitted_variables_match_fortran(case_name: str) -> None:
    case = bundled_case(case_name)
    compiled_run = initialize_gotm(case.yaml_path)
    reference = open_reference_dataset(case)
    actual = None
    try:
        runtime = integrate_gotm_compiled(compiled_run, output=True)
        actual = runtime_output_to_dataset(compiled_run, runtime)

        assert runtime.output.nout == runtime.output.time.shape[0]
        assert actual.sizes["time"] == reference.sizes["time"]

        variables = (
            "u",
            "v",
            "temp",
            "salt",
            "tke",
            "eps",
            "num",
            "nuh",
            "h",
            "xP",
            "fric",
            "drag",
            "bioshade",
            "ga",
            "SS",
            "P",
            "G",
            "Pb",
            "kb",
            "epsb",
            "L",
            "PSTK",
            "cmue2",
            "an",
            "nus",
            "nucl",
        )
        assert_dataset_variables_allclose(actual, reference, variables)
    finally:
        if actual is not None:
            actual.close()
        reference.close()
        finalize_gotm(compiled_run)


@pytest.mark.parametrize(
    ("case_name", "max_steps", "sprof_active", "tprof_active"),
    (
        ("channel", 360, 0, 0),
        ("entrainment", 24, 1, 1),
    ),
)
def test_compiled_avh_matches_fortran_for_transport_paths(
    case_name: str,
    max_steps: int,
    sprof_active: int,
    tprof_active: int,
) -> None:
    case = bundled_case(case_name)
    compiled_run = initialize_gotm(case.yaml_path)
    reference = open_reference_dataset(case)
    actual = None
    try:
        runtime = integrate_gotm_compiled(
            compiled_run,
            max_steps=max_steps,
            output=True,
        )
        actual = runtime_output_to_dataset(compiled_run, runtime)
        reference_slice = reference.isel(time=slice(0, actual.sizes["time"])).squeeze(
            drop=True
        )

        assert runtime.params.sprof_input_active == sprof_active
        assert runtime.params.tprof_input_active == tprof_active
        assert "avh" in actual.data_vars
        assert "avh" in reference_slice.data_vars
        assert_dataset_variables_allclose(
            actual,
            reference_slice,
            ("avh",),
            atol=2.0e-9,
        )
    finally:
        if actual is not None:
            actual.close()
        reference.close()
        finalize_gotm(compiled_run)


@pytest.mark.parametrize("case_name", ("entrainment",))
def test_compiled_profile_cases_emit_parity_comparable_output(case_name: str) -> None:
    case = bundled_case(case_name)
    compiled_run = initialize_gotm(case.yaml_path)
    if compiled_run.fabm_config is not None and compiled_run.fabm_config.use:
        finalize_gotm(compiled_run)
        pytest.skip(
            "FABM cases run the full FABM loop; profile parity tested separately"
        )

    reference = open_reference_dataset(case)
    actual = None
    try:
        runtime = integrate_gotm_compiled(compiled_run, max_steps=24, output=True)
        actual = runtime_output_to_dataset(compiled_run, runtime)
        reference_slice = reference.isel(time=slice(0, actual.sizes["time"])).squeeze(
            drop=True
        )

        assert getattr(runtime.runner, "__name__", "") == "run_compiled_time_loop"
        assert runtime.output.output_step[0] == 0
        assert runtime.output.output_step[-1] == 24
        assert actual.sizes["time"] == runtime.output.nout
        for variable in ("u", "v", "temp", "salt", "tke", "eps", "num", "nuh"):
            assert variable in actual.data_vars
            assert np.isfinite(actual[variable].values).all()

        variables = (
            "u",
            "v",
            "temp",
            "salt",
            "tke",
            "eps",
            "num",
            "nuh",
            "h",
            "SS",
            "P",
            "G",
            "Pb",
            "kb",
            "epsb",
            "L",
            "cmue2",
            "an",
            "nus",
            "nucl",
        )
        shared = tuple(
            name
            for name in variables
            if name in actual.data_vars and name in reference_slice.data_vars
        )
        assert_dataset_variables_allclose(actual, reference_slice, shared)
    finally:
        if actual is not None:
            actual.close()
        reference.close()
        finalize_gotm(compiled_run)

    assert shared
    assert time_loop_compiled.nopython_signatures
