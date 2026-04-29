from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pygotm.driver import GotmDriver
from pygotm.validate import (
    REFERENCE_CASE_NAMES,
    ValidationCase,
    compare_datasets,
    discover_reference_cases,
    open_reference_dataset,
    resolve_reference_case,
    run_case_validation,
)

# ---------------------------------------------------------------------------
# pytest marks
# ---------------------------------------------------------------------------
# slow: full-run case validation — deselect with:  pytest -m "not slow"
# Run the full gate with:  pytest -m slow tests/integration/

def test_reference_case_inventory_matches_declared_suite() -> None:
    discovered = tuple(case.name for case in discover_reference_cases())
    assert discovered == REFERENCE_CASE_NAMES


@pytest.mark.parametrize("case_name", REFERENCE_CASE_NAMES)
def test_case_assets_are_present(case_name: str) -> None:
    case = resolve_reference_case(case_name)
    assert case.directory.is_dir()
    assert case.yaml_path.is_file()
    assert case.reference_path.is_file()


@pytest.mark.parametrize("case_name", REFERENCE_CASE_NAMES)
def test_reference_dataset_self_comparison_is_exact(case_name: str) -> None:
    case = resolve_reference_case(case_name)
    dataset = open_reference_dataset(case)
    try:
        comparison = compare_datasets(dataset, dataset)
    finally:
        dataset.close()

    assert comparison.ok
    assert comparison.failures == ()


def test_compare_datasets_squeezes_singleton_lat_lon_dimensions() -> None:
    case = resolve_reference_case("couette")
    dataset = open_reference_dataset(case)
    try:
        squeezed = dataset.squeeze(drop=True)
        comparison = compare_datasets(squeezed, dataset, variables=("temp", "salt"))
    finally:
        dataset.close()

    assert comparison.ok


def test_compare_datasets_uses_range_aware_tolerance() -> None:
    expected = xr.Dataset(
        {"signal": ("z", np.array([0.0, 100.0], dtype=np.float64))}
    )
    within = xr.Dataset(
        {"signal": ("z", np.array([9.0e-6, 100.0], dtype=np.float64))}
    )
    outside = xr.Dataset(
        {"signal": ("z", np.array([1.1e-5, 100.0], dtype=np.float64))}
    )

    assert compare_datasets(within, expected, variables=("signal",)).ok

    comparison = compare_datasets(outside, expected, variables=("signal",))
    assert not comparison.ok
    assert comparison.failures[0].variable == "signal"


def test_run_case_validation_accepts_custom_runner() -> None:
    def runner(case: ValidationCase) -> xr.Dataset:
        return xr.open_dataset(case.reference_path, engine="scipy")

    comparison = run_case_validation("couette", runner=runner)

    assert comparison.ok
    assert comparison.failures == ()


def test_couette_driver_matches_reference_for_full_hourly_slice() -> None:
    case = resolve_reference_case("couette")
    actual = GotmDriver(case.yaml_path).run(max_steps=360)
    expected = open_reference_dataset(case)
    try:
        expected_slice = expected.isel(time=slice(0, 2)).squeeze(drop=True)
        comparison = compare_datasets(actual, expected_slice)
    finally:
        expected.close()
        actual.close()

    assert comparison.ok
    assert comparison.failures == ()


def test_couette_driver_advances_velocity_and_turbulence() -> None:
    case = resolve_reference_case("couette")
    dataset = GotmDriver(case.yaml_path).run(max_steps=360)
    try:
        assert dataset.sizes["time"] == 2
        assert float(np.max(np.abs(dataset["u"].values[1] - dataset["u"].values[0]))) > 0.0
        assert float(np.max(dataset["num"].values[1])) > 0.0
    finally:
        dataset.close()


@pytest.mark.slow
@pytest.mark.parametrize("case_name", REFERENCE_CASE_NAMES)
def test_full_case_matches_reference(case_name: str) -> None:
    """Release gate: full simulation must pass rtol=1e-6 against Fortran GOTM.

    Compares all variables that pyGOTM outputs against the Fortran reference.
    Variables present only in the reference (unimplemented features such as
    FABM biogeochemistry or ice model) are skipped and reported separately.

    Run with:  uv run pytest -m slow tests/integration/
    """
    from pygotm.driver import GotmDriver

    case = resolve_reference_case(case_name)
    actual = GotmDriver(case.yaml_path).run()
    expected = open_reference_dataset(case)

    try:
        # Compare only the intersection: variables that pyGOTM actually outputs.
        # Variables absent from pyGOTM (FABM, ice model) are unimplemented features,
        # not wrong answers — do not count them as failures.
        actual_var_names = {
            str(n)
            for n, da in actual.data_vars.items()
            if np.issubdtype(da.dtype, np.number)
        }
        expected_var_names = tuple(
            str(n)
            for n, da in expected.data_vars.items()
            if np.issubdtype(da.dtype, np.number)
        )
        shared_vars = tuple(v for v in expected_var_names if v in actual_var_names)
        unimplemented = [v for v in expected_var_names if v not in actual_var_names]

        comparison = compare_datasets(
            actual, expected, rtol=1e-6, atol=1e-12, variables=shared_vars
        )
    finally:
        actual.close()
        expected.close()

    if unimplemented:
        # Not a failure — just informational
        print(
            f"\n[INFO] {case_name}: {len(unimplemented)} reference variable(s) not compared "
            f"(unimplemented): {', '.join(unimplemented[:5])}"
            + (f" ... ({len(unimplemented) - 5} more)" if len(unimplemented) > 5 else "")
        )

    if comparison.failures:
        details = "; ".join(
            f"{f.variable} rel={f.max_rel_error:.2e} abs={f.max_abs_error:.2e}"
            for f in comparison.failures[:5]
        )
        if len(comparison.failures) > 5:
            details += f"; ... ({len(comparison.failures) - 5} more)"
        pytest.fail(
            f"Case {case_name!r}: {len(comparison.failures)}/{len(comparison.checked_variables)} "
            f"variable(s) exceeded rtol=1e-6. First failures: {details}"
        )
    assert comparison.ok
