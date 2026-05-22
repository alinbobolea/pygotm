"""Focused dataset assertions for tests that do not need Frechet reports."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


def assert_dataset_variables_allclose(
    actual: xr.Dataset,
    expected: xr.Dataset,
    variables: tuple[str, ...],
    *,
    rtol: float = 5.0e-6,
    atol: float = 1.0e-12,
) -> None:
    """Assert selected variables are pointwise close in test-only checks."""

    failures: list[str] = []
    for name in variables:
        actual_values = np.squeeze(actual[name].values)
        expected_values = np.squeeze(expected[name].values)
        try:
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=rtol,
                atol=atol,
                err_msg=f"variable {name!r}",
            )
        except AssertionError as exc:
            message = str(exc).strip().splitlines()[0]
            failures.append(message or f"variable {name!r} differs")

    if failures:
        pytest.fail("; ".join(failures))
