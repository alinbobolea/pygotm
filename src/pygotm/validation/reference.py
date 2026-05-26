"""Reference-case discovery and NetCDF opening helpers for validation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr

__all__ = [
    "REFERENCE_CASE_NAMES",
    "ValidationCase",
    "discover_reference_cases",
    "numeric_variable_names",
    "open_reference_dataset",
    "open_validation_dataset",
    "resolve_reference_case",
]

REFERENCE_CASE_NAMES: tuple[str, ...] = (
    "couette",
    "blacksea",
    "channel",
    "entrainment",
    "estuary",
    "flex",
    "gotland",
    "lago_maggiore",
    "langmuir",
    "liverpool_bay",
    "medsea_east",
    "medsea_west",
    "nns_annual",
    "nns_seasonal",
    "ows_papa",
    "plume",
    "resolute",
    "reynolds",
    "rouse",
    "seagrass",
    "wave_breaking",
    "asics_med",
)


@dataclass(frozen=True)
class ValidationCase:
    """Resolved metadata for one GOTM reference case and YAML input file."""

    name: str
    directory: Path
    yaml_path: Path
    reference_path: Path

    @property
    def yaml_base(self) -> str:
        """Base name of the selected GOTM YAML input file."""

        return self.yaml_path.stem

    @property
    def task_name(self) -> str:
        """Dask task name for this case/input pair."""

        return f"{self.name}-{self.yaml_base}"

    @property
    def run_name(self) -> str:
        """Validation output name for this case/input pair."""

        if self.yaml_base == "gotm":
            return self.name
        return self.task_name


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cases_root(cases_root: Path | None = None) -> Path:
    return (cases_root or _project_root() / "validation" / "reference").resolve()


def _select_reference_output(case_dir: Path) -> Path:
    candidates = sorted(
        path for path in case_dir.glob("*.nc") if path.name != "restart.nc"
    )
    if not candidates:
        msg = f"no reference NetCDF output found in {case_dir}"
        raise FileNotFoundError(msg)
    return max(candidates, key=lambda path: (path.stat().st_size, path.name))


def _split_case_spec(
    case_name: str,
    yaml_file: str | Path | None,
) -> tuple[str, str | Path | None]:
    if yaml_file is not None:
        return case_name, yaml_file

    for separator in ("/", ":"):
        if separator in case_name:
            case_part, yaml_part = case_name.split(separator, 1)
            if case_part and yaml_part:
                return case_part, yaml_part

    return case_name, None


def _yaml_file_candidates(case_dir: Path, yaml_file: str | Path) -> tuple[Path, ...]:
    requested = Path(yaml_file)
    path = requested if requested.is_absolute() else case_dir / requested
    if path.suffix:
        return (path,)
    return (
        path.with_suffix(".yaml"),
        path.with_suffix(".yml"),
        path,
    )


def _select_case_yaml(
    case_dir: Path,
    case_name: str,
    yaml_file: str | Path | None,
) -> Path:
    if yaml_file is not None:
        for candidate in _yaml_file_candidates(case_dir, yaml_file):
            if candidate.is_file():
                return candidate.resolve()
        tried = ", ".join(
            str(path) for path in _yaml_file_candidates(case_dir, yaml_file)
        )
        msg = (
            f"missing YAML input file {str(yaml_file)!r} for case "
            f"{case_name!r}; tried {tried}"
        )
        raise FileNotFoundError(msg)

    preferred = case_dir / "gotm.yaml"
    if preferred.is_file():
        return preferred.resolve()

    preferred_yml = case_dir / "gotm.yml"
    if preferred_yml.is_file():
        return preferred_yml.resolve()

    all_yaml = sorted(
        path
        for suffix in ("*.yaml", "*.yml")
        for path in case_dir.glob(suffix)
        if path.name not in {"fabm.yaml", "fabm.yml", "output.yaml", "output.yml"}
    )
    gotm_yaml = [path for path in all_yaml if path.stem.startswith("gotm")]
    if len(gotm_yaml) == 1:
        return gotm_yaml[0].resolve()
    if len(all_yaml) == 1:
        return all_yaml[0].resolve()

    if not all_yaml:
        msg = f"missing GOTM YAML input file for case {case_name!r} under {case_dir}"
    else:
        names = ", ".join(path.name for path in all_yaml)
        msg = (
            f"multiple GOTM YAML input files found for case {case_name!r} "
            f"under {case_dir}: {names}; specify one explicitly"
        )
    raise FileNotFoundError(msg)


def _open_dataset(path: Path) -> xr.Dataset:
    scipy_error: Exception | None = None
    try:
        return xr.open_dataset(path, engine="scipy")
    except Exception as exc:
        scipy_error = exc

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"numpy\.ndarray size changed, may indicate binary "
                r"incompatibility\..*"
            ),
            category=RuntimeWarning,
        )
        try:
            return xr.open_dataset(path, engine="netcdf4")
        except Exception as exc:
            msg = (
                f"failed to open dataset {path} with scipy backend "
                f"({scipy_error!r}) and netcdf4 backend ({exc!r})"
            )
            raise OSError(msg) from exc


def open_validation_dataset(path: Path) -> xr.Dataset:
    """Open a validation NetCDF with the same backend fallback as references."""

    return _open_dataset(path)


def resolve_reference_case(
    case_name: str,
    *,
    yaml_file: str | Path | None = None,
    cases_root: Path | None = None,
) -> ValidationCase:
    """Resolve a named reference case to its YAML and Fortran NetCDF paths."""

    case_name, yaml_file = _split_case_spec(case_name, yaml_file)
    root = _cases_root(cases_root)
    case_dir = root / case_name
    if not case_dir.is_dir():
        msg = f"unknown GOTM reference case {case_name!r} under {root}"
        raise FileNotFoundError(msg)

    yaml_path = _select_case_yaml(case_dir, case_name, yaml_file)

    return ValidationCase(
        name=case_name,
        directory=case_dir,
        yaml_path=yaml_path,
        reference_path=_select_reference_output(case_dir),
    )


def discover_reference_cases(
    *,
    cases_root: Path | None = None,
) -> tuple[ValidationCase, ...]:
    """Resolve all declared reference cases in canonical order."""

    return tuple(
        resolve_reference_case(case_name, cases_root=cases_root)
        for case_name in REFERENCE_CASE_NAMES
    )


def open_reference_dataset(case: ValidationCase) -> xr.Dataset:
    """Open the Fortran reference NetCDF for *case*."""

    return _open_dataset(case.reference_path)


def numeric_variable_names(dataset: xr.Dataset) -> tuple[str, ...]:
    """Return numeric data variables in dataset order."""

    names: list[str] = []
    for name, data_array in dataset.data_vars.items():
        if np.issubdtype(data_array.dtype, np.number):
            names.append(cast(str, name))
    return tuple(names)
