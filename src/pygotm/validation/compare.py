"""NetCDF comparison utilities for pyGOTM validation — three-indicator system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from pygotm.validate import numeric_variable_names, open_validation_dataset
from pygotm.validation.tolerances import classify_section, get_tolerance

__all__ = ["VarResult", "ValidationError", "compare_nc"]


class ValidationError(Exception):
    """Raised for unrecoverable comparison failures (e.g. empty valid-data mask)."""


@dataclass
class VarResult:
    """Per-variable result for the three-indicator validation system."""

    name: str
    section: Literal["pygotm", "pyfabm"]

    status: Literal["PASS", "MARGINAL", "DISCREPANT", "BROKEN"]
    color: Literal["green", "yellow", "orange", "red"]

    reference_at_worst: float
    calculated_at_worst: float

    primary_score: float
    birge_ratio: float
    normalized_signed_bias: float

    plot_html: str | None  # Plotly div for MARGINAL/DISCREPANT; None otherwise


def _classify_status(
    primary_score: float,
) -> tuple[
    Literal["PASS", "MARGINAL", "DISCREPANT", "BROKEN"],
    Literal["green", "yellow", "orange", "red"],
]:
    if primary_score <= 1.0:
        return "PASS", "green"
    if primary_score <= 3.0:
        return "MARGINAL", "yellow"
    if primary_score <= 10.0:
        return "DISCREPANT", "orange"
    return "BROKEN", "red"


def _make_plot_html(
    ref_arr: np.ndarray,
    calc_arr: np.ndarray,
    *,
    case_name: str,
    section: str,
    variable: str,
    status: str,
    primary_score: float,
    birge_ratio: float,
    normalized_signed_bias: float,
) -> str:
    section_label = "PyGOTM" if section == "pygotm" else "PyFABM"
    title = (
        f"Case: {case_name} | Section: {section_label} | Variable: {variable}<br>"
        f"Status: {status} | Primary score: {primary_score:.2f} | "
        f"Birge: {birge_ratio:.2f} | Bias: {normalized_signed_bias:.2f}"
    )
    x = list(range(len(ref_arr.ravel())))
    fig = go.Figure(
        [
            go.Scatter(x=x, y=ref_arr.ravel().tolist(), mode="lines", name="Reference"),
            go.Scatter(
                x=x,
                y=calc_arr.ravel().tolist(),
                mode="lines",
                name="Calculated",
            ),
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Simulation step index",
        yaxis_title=variable,
        height=300,
    )
    return str(pio.to_html(fig, include_plotlyjs=False, full_html=False))


def _broken_result(name: str) -> VarResult:
    section = classify_section(name)
    return VarResult(
        name=name,
        section=section,
        status="BROKEN",
        color="red",
        reference_at_worst=float("nan"),
        calculated_at_worst=float("nan"),
        primary_score=float("inf"),
        birge_ratio=float("inf"),
        normalized_signed_bias=float("nan"),
        plot_html=None,
    )


def _compute_var_result(
    name: str,
    ref_arr: np.ndarray,
    calc_arr: np.ndarray,
    case_name: str,
) -> VarResult:
    """Compute three-indicator metrics for one variable."""
    section = classify_section(name)
    tol = get_tolerance(name)
    atol, rtol, sf = tol.atol, tol.rtol, tol.scale_floor

    valid = np.isfinite(ref_arr) & np.isfinite(calc_arr)
    if not np.any(valid):
        raise ValidationError(
            f"Variable {name!r}: no valid (finite) data points in either "
            "reference or calculated array."
        )

    ref_v = ref_arr[valid]
    calc_v = calc_arr[valid]

    denominator = atol + rtol * np.maximum(np.abs(ref_v), sf)
    E_i = np.abs(calc_v - ref_v) / denominator

    primary_score = float(np.percentile(E_i, 99))
    birge_ratio = float(np.sqrt(np.mean(E_i**2)))

    mean_signed_error = float(np.mean(calc_v - ref_v))
    bias_scale = atol + rtol * float(np.maximum(np.mean(np.abs(ref_v)), sf))
    normalized_signed_bias = mean_signed_error / bias_scale

    status, color = _classify_status(primary_score)

    valid_flat_indices = np.flatnonzero(valid)
    worst_valid_idx = int(np.argmax(E_i))
    worst_flat_idx = int(valid_flat_indices[worst_valid_idx])

    ref_at_worst = float(ref_arr.ravel()[worst_flat_idx])
    calc_at_worst = float(calc_arr.ravel()[worst_flat_idx])

    plot_html: str | None = None
    if status in ("MARGINAL", "DISCREPANT"):
        plot_html = _make_plot_html(
            ref_arr,
            calc_arr,
            case_name=case_name,
            section=section,
            variable=name,
            status=status,
            primary_score=primary_score,
            birge_ratio=birge_ratio,
            normalized_signed_bias=normalized_signed_bias,
        )

    return VarResult(
        name=name,
        section=section,
        status=status,
        color=color,
        reference_at_worst=ref_at_worst,
        calculated_at_worst=calc_at_worst,
        primary_score=primary_score,
        birge_ratio=birge_ratio,
        normalized_signed_bias=normalized_signed_bias,
        plot_html=plot_html,
    )


def compare_nc(
    py_path: Path,
    ref_path: Path,
    case_name: str,
) -> list[VarResult]:
    """Compare *py_path* against *ref_path* using the three-indicator system.

    Variables with structural failures (shape mismatch, missing) receive BROKEN status.
    """
    py_ds = open_validation_dataset(py_path)
    try:
        ref_ds = open_validation_dataset(ref_path)
        try:
            ref_vars = numeric_variable_names(ref_ds)
            py_vars = {
                str(name)
                for name, da in py_ds.data_vars.items()
                if np.issubdtype(da.dtype, np.number)
            }

            results: list[VarResult] = []
            for name in ref_vars:
                if name not in py_vars:
                    results.append(_broken_result(name))
                    continue

                if (
                    py_ds[name].dims != ref_ds[name].dims
                    or py_ds[name].shape != ref_ds[name].shape
                ):
                    results.append(_broken_result(name))
                    continue

                ref_arr = np.asarray(ref_ds[name].values, dtype=np.float64).ravel()
                calc_arr = np.asarray(py_ds[name].values, dtype=np.float64).ravel()

                try:
                    results.append(  # noqa: E501
                        _compute_var_result(name, ref_arr, calc_arr, case_name)
                    )
                except ValidationError:
                    results.append(_broken_result(name))

            for name in sorted(py_vars.difference(set(ref_vars))):
                results.append(_broken_result(name))

        finally:
            ref_ds.close()
    finally:
        py_ds.close()

    return results
