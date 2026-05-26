from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pygotm.config import load_config
from pygotm.driver import GotmDriver
from pygotm.gotm.gotm import (
    initialize_gotm_from_settings,
    integrate_gotm_compiled,
)
from pygotm.gotm.runtime_builder import (
    runtime_output_to_dataset,
)
from pygotm.validation.compare import compare_nc
from pygotm.validation.reference import open_reference_dataset
from tests.dataset_assertions import assert_dataset_variables_allclose
from tests.fixtures import BUNDLED_CASE_NAMES, bundled_case

_COMPILED_VALIDATED_VARIABLES = (
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


@pytest.mark.parametrize("case_name", BUNDLED_CASE_NAMES)
def test_case_assets_are_present(case_name: str) -> None:
    case = bundled_case(case_name)
    assert case.directory.is_dir()
    assert case.yaml_path.is_file()
    assert case.reference_path.is_file()


@pytest.mark.parametrize("case_name", BUNDLED_CASE_NAMES)
def test_reference_dataset_self_comparison_is_exact(case_name: str) -> None:
    case = bundled_case(case_name)
    results = compare_nc(case.reference_path, case.reference_path, case_name=case.name)
    failures = [result for result in results if result.status != "PASS"]

    assert not failures


def test_couette_driver_matches_reference_for_full_hourly_slice() -> None:
    case = bundled_case("couette")
    actual = GotmDriver(case.yaml_path).run(max_steps=360)
    expected = open_reference_dataset(case)
    try:
        expected_slice = expected.isel(time=slice(0, 2)).squeeze(drop=True)
        variables = tuple(
            name
            for name in _COMPILED_VALIDATED_VARIABLES
            if name in actual.data_vars and name in expected_slice.data_vars
        )
        assert_dataset_variables_allclose(actual, expected_slice, variables)
    finally:
        expected.close()
        actual.close()


def test_couette_driver_advances_velocity_and_turbulence() -> None:
    case = bundled_case("couette")
    dataset = GotmDriver(case.yaml_path).run(max_steps=360)
    try:
        assert dataset.sizes["time"] == 2
        assert dataset["u"].dims == ("time", "z", "lat", "lon")
        assert dataset["tke"].dims == ("time", "zi", "lat", "lon")
        assert dataset.coords["z"].dims == ("time", "z", "lat", "lon")
        assert dataset.coords["lat"].shape == (1,)
        assert dataset.coords["lon"].shape == (1,)
        velocity_change = np.max(
            np.abs(dataset["u"].values[1] - dataset["u"].values[0])
        )
        assert float(velocity_change) > 0.0
        assert float(np.max(dataset["num"].values[1])) > 0.0
    finally:
        dataset.close()


@pytest.mark.parametrize("case_name", BUNDLED_CASE_NAMES)
def test_full_case_matches_reference(case_name: str, tmp_path: Path) -> None:
    """Release gate: full simulation must pass Frechet validation.

    Compares every numeric variable in the Fortran reference. Missing pyGOTM
    outputs are validation failures because release parity requires matching
    NetCDF structure and content.
    """
    from pygotm.driver import GotmDriver

    case = bundled_case(case_name)
    py_path = tmp_path / f"{case.run_name}.nc"
    actual = GotmDriver(case.yaml_path).run(output_path=py_path)
    actual.close()

    results = compare_nc(py_path, case.reference_path, case_name=case.run_name)
    failures = [result for result in results if result.status != "PASS"]

    if failures:
        details = "; ".join(
            f"{result.name} status={result.status} score={result.score} "
            f"d_raw={result.d_raw:.2e} d_norm={result.d_norm:.2e}"
            for result in failures[:5]
        )
        if len(failures) > 5:
            details += f"; ... ({len(failures) - 5} more)"
        pytest.fail(
            f"Case {case_name!r}: {len(failures)}/{len(results)} "
            f"variable(s) failed Frechet validation. First failures: {details}"
        )


# ---------------------------------------------------------------------------
# Chunked interleaving tests
# ---------------------------------------------------------------------------


def test_non_fabm_case_unaffected_by_chunk_size() -> None:
    """Non-FABM cases must produce identical output regardless of chunk_size."""
    case = bundled_case("couette")
    cfg = load_config(case.yaml_path)

    def _run(chunk_size: int | None) -> xr.Dataset:
        run = initialize_gotm_from_settings(
            cfg.resolved_settings(),
            yaml_path=case.yaml_path,
            document=cfg.resolved_document(),
        )
        bundle = integrate_gotm_compiled(run, max_steps=360, chunk_size=chunk_size)
        return runtime_output_to_dataset(run, bundle)

    ds_default = _run(chunk_size=None)
    ds_24 = _run(chunk_size=24)

    try:
        for var in ("temp", "salt", "u", "v", "tke"):
            if var in ds_default.data_vars and var in ds_24.data_vars:
                np.testing.assert_array_equal(
                    ds_default[var].values,
                    ds_24[var].values,
                    err_msg=f"chunk_size=24 changed non-FABM variable {var!r}",
                )
    finally:
        ds_default.close()
        ds_24.close()
