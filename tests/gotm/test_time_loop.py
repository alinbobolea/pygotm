"""Tests for the compiled central GOTM time-loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pygotm.gotm.gotm import (
    _integrate_gotm_python,
    finalize_gotm,
    initialize_gotm,
    integrate_gotm_compiled,
)
from pygotm.gotm.runtime_builder import (
    build_runtime_forcing_from_run,
    build_runtime_from_run,
    runtime_output_to_dataset,
)
from pygotm.gotm.runtime_state import RuntimeState
from pygotm.gotm.time_loop import (
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
from pygotm.validate import (
    compare_datasets,
    open_reference_dataset,
    resolve_reference_case,
)

_COUETTE_CONFIG = Path("gotm-model/cases-runs/couette/gotm.yaml")
_MEDSEA_WEST_CONFIG = Path("gotm-model/cases-runs/medsea_west/gotm.yaml")


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



@pytest.mark.slow
def test_runtime_forcing_precomputes_medsea_profile_and_airsea_series() -> None:
    run = initialize_gotm(_MEDSEA_WEST_CONFIG)
    try:
        forcing = build_runtime_forcing_from_run(run, max_steps=1)
    finally:
        finalize_gotm(run)

    assert forcing.nt == 1
    assert forcing.Tobs.shape == (2, 401)
    assert forcing.Sobs.shape == (2, 401)
    assert np.isfinite(forcing.Tobs).all()
    assert np.isfinite(forcing.Sobs).all()
    assert forcing.airp[0] > 90_000.0
    assert forcing.airp[1] > 90_000.0
    assert forcing.u10[0] != 0.0
    assert forcing.v10[0] != 0.0
    assert 0.0 <= forcing.cloud[0] <= 1.0
    assert forcing.yearday.tolist() == [15, 15]


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.parametrize("case_name", ("couette", "channel"))
def test_compiled_reference_emitted_variables_match_fortran(case_name: str) -> None:
    case = resolve_reference_case(case_name)
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
        comparison = compare_datasets(
            actual,
            reference,
            rtol=5.0e-6,
            atol=1.0e-12,
            variables=variables,
        )
    finally:
        if actual is not None:
            actual.close()
        reference.close()
        finalize_gotm(compiled_run)

    if comparison.failures:
        details = "; ".join(
            f"{f.variable} index={f.index} rel={f.max_rel_error:.2e} "
            f"abs={f.max_abs_error:.2e}"
            for f in comparison.failures
        )
        pytest.fail(f"compiled {case_name} core validation failed: {details}")
    assert comparison.ok


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_name",
    (
        "blacksea",
        "medsea_west",
        "entrainment",
        "flex",
        "gotland",
        "lago_maggiore",
    ),
)
def test_compiled_profile_cases_emit_parity_comparable_output(case_name: str) -> None:
    case = resolve_reference_case(case_name)
    compiled_run = initialize_gotm(case.yaml_path)
    reference = open_reference_dataset(case)
    actual = None
    try:
        runtime = integrate_gotm_compiled(compiled_run, max_steps=24, output=True)
        actual = runtime_output_to_dataset(compiled_run, runtime)
        reference_slice = reference.isel(time=slice(0, actual.sizes["time"])).squeeze(
            drop=True
        )

        assert (
            getattr(runtime.runner, "__name__", "")
            == "run_compiled_time_loop"
        )
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
        comparison = compare_datasets(
            actual,
            reference_slice,
            rtol=5.0e-6,
            atol=1.0e-12,
            variables=shared,
        )
    finally:
        if actual is not None:
            actual.close()
        reference.close()
        finalize_gotm(compiled_run)

    assert comparison.checked_variables
    assert time_loop_compiled.nopython_signatures
