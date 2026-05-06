"""Single-column driver facade for the translated pyGOTM stack."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from pygotm.config import ConfigLike, coerce_config
from pygotm.gotm.gotm import (
    GotmRun,
    finalize_gotm,
    initialize_gotm_from_settings,
    integrate_gotm_compiled,
)
from pygotm.gotm.register_all_variables import FieldRecord
from pygotm.gotm.runtime_builder import (
    UnsupportedConfigurationError,
    runtime_output_to_dataset,
)

__all__ = ["GotmDriver"]

_SCALAR_COORDS = frozenset({"lat", "lon"})


def _field_attrs(record: FieldRecord) -> dict[str, str]:
    attrs = {
        "units": record.units,
        "long_name": record.long_name,
    }
    if record.standard_name:
        attrs["standard_name"] = record.standard_name
    if record.category:
        attrs["category"] = record.category
    return attrs


def _time_values(snapshot_times: list[str]) -> np.ndarray:
    if not snapshot_times:
        return np.array([], dtype="datetime64[s]")
    return np.array(
        [np.datetime64(value.replace(" ", "T")) for value in snapshot_times],
        dtype="datetime64[s]",
    )


def _profile_axis(
    record: FieldRecord,
    name: str,
    values: np.ndarray,
    nlev: int,
) -> tuple[str, np.ndarray]:
    if values.ndim != 1:
        msg = f"field {name!r} is not one-dimensional: shape={values.shape!r}"
        raise ValueError(msg)

    dimensions = record.dimensions
    if dimensions == ("zi",):
        if values.shape[0] != nlev + 1:
            msg = (
                f"field {name!r} must have nlev+1={nlev + 1} entries, "
                f"got {values.shape[0]}"
            )
            raise ValueError(msg)
        return "zi", values

    if dimensions == ("z",) and values.shape[0] == nlev + 1:
        # Most Fortran profiles are stored as DIMENSION(0:nlev); drop the
        # sentinel bottom slot when exposing them on the physical cell-centre axis.
        return "z", values[1:]

    if dimensions == ("z",) and values.shape[0] == nlev:
        return "z", values

    if values.shape[0] == nlev + 1:
        return "zi", values

    if values.shape[0] == nlev:
        return "z", values

    msg = (
        f"field {name!r} has unsupported profile length {values.shape[0]}; "
        f"expected {nlev} or {nlev + 1}"
    )
    raise ValueError(msg)


def _all_rows_equal(values: np.ndarray) -> bool:
    if values.shape[0] <= 1:
        return True
    return bool(np.allclose(values, values[0], equal_nan=True))


def _dataset_from_run(run: GotmRun) -> xr.Dataset:
    coords: dict[str, Any] = {
        "time": _time_values(run.snapshot_times),
        "lat": float(run.latitude),
        "lon": float(run.longitude),
    }
    data_vars: dict[str, Any] = {}

    if not run.snapshots:
        assert run.meanflow.z is not None
        assert run.meanflow.zi is not None
        coords["z"] = np.asarray(run.meanflow.z[1:], dtype=np.float64)
        coords["zi"] = np.asarray(run.meanflow.zi, dtype=np.float64)
        return xr.Dataset(coords=coords, attrs=_dataset_attrs(run))

    first_snapshot = run.snapshots[0]
    for name, record in run.registry.fields.items():
        if name in _SCALAR_COORDS:
            continue

        sample = first_snapshot[name]
        if isinstance(sample, np.ndarray):
            profiles: list[np.ndarray] = []
            dim_name: str | None = None
            for snapshot in run.snapshots:
                profile = np.asarray(snapshot[name], dtype=np.float64)
                axis_name, normalized = _profile_axis(record, name, profile, run.nlev)
                if dim_name is None:
                    dim_name = axis_name
                profiles.append(np.asarray(normalized, dtype=np.float64))

            assert dim_name is not None
            stacked = np.stack(profiles, axis=0)
            if name == "z":
                coords["z"] = (
                    stacked[0] if _all_rows_equal(stacked) else (("time", "z"), stacked)
                )
                continue
            if name == "zi":
                coords["zi"] = (
                    stacked[0]
                    if _all_rows_equal(stacked)
                    else (("time", "zi"), stacked)
                )
                continue

            data_vars[name] = (("time", dim_name), stacked, _field_attrs(record))
            continue

        series = np.array(
            [float(snapshot[name]) for snapshot in run.snapshots],
            dtype=np.float64,
        )
        data_vars[name] = (("time",), series, _field_attrs(record))

    if "z" not in coords:
        assert run.meanflow.z is not None
        coords["z"] = np.asarray(run.meanflow.z[1:], dtype=np.float64)
    if "zi" not in coords:
        assert run.meanflow.zi is not None
        coords["zi"] = np.asarray(run.meanflow.zi, dtype=np.float64)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=_dataset_attrs(run))


def _empty_dataset_from_run(run: GotmRun) -> xr.Dataset:
    coords: dict[str, Any] = {
        "time": np.array([], dtype="datetime64[s]"),
        "lat": float(run.latitude),
        "lon": float(run.longitude),
    }
    assert run.meanflow.z is not None
    assert run.meanflow.zi is not None
    coords["z"] = np.asarray(run.meanflow.z[1:], dtype=np.float64)
    coords["zi"] = np.asarray(run.meanflow.zi, dtype=np.float64)
    return xr.Dataset(coords=coords, attrs=_dataset_attrs(run))


def _dataset_attrs(run: GotmRun) -> dict[str, str | int | float]:
    return {
        "title": run.settings.title,
        "source_yaml": str(run.yaml_path),
        "nlev": run.nlev,
        "dt": float(run.dt),
    }


class GotmDriver:
    """Run the current single-column pyGOTM stack and emit xarray output."""

    def __init__(self, config: ConfigLike) -> None:
        self.config = coerce_config(config)

    def run(
        self,
        *,
        max_steps: int | None = None,
        output_path: str | Path | None = None,
        output: bool = True,
        on_step: Callable[[int, int], None] | None = None,
    ) -> xr.Dataset:
        """Execute a single-column run and return the resulting dataset."""

        if on_step is not None:
            msg = "compiled GOTM runtime does not yet support on_step callbacks"
            raise UnsupportedConfigurationError(msg)

        run = initialize_gotm_from_settings(
            self.config.resolved_settings(),
            yaml_path=self.config.source_path or Path("gotm.yaml"),
            document=self.config.resolved_document(),
        )
        try:
            bundle = integrate_gotm_compiled(run, max_steps=max_steps, output=output)
            dataset = (
                runtime_output_to_dataset(run, bundle)
                if output
                else _empty_dataset_from_run(run)
            )
        finally:
            finalize_gotm(run)

        if output_path is not None:
            self.write_dataset(dataset, output_path)
        return dataset

    @staticmethod
    def write_dataset(dataset: xr.Dataset, path: str | Path) -> None:
        """Write *dataset* to a NetCDF file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_netcdf(output_path, engine="scipy")
