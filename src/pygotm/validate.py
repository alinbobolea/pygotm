from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import click
import numpy as np
import xarray as xr

__all__ = [
    "REFERENCE_CASE_NAMES",
    "DatasetComparison",
    "ValidationCase",
    "ValidationFailure",
    "cli",
    "compare_datasets",
    "discover_reference_cases",
    "numeric_variable_names",
    "open_reference_dataset",
    "open_validation_dataset",
    "resolve_reference_case",
    "run_case_validation",
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
        """Validation output name, preserving legacy paths for gotm.yaml cases."""

        if self.yaml_base == "gotm":
            return self.name
        return self.task_name


@dataclass(frozen=True)
class ValidationFailure:
    variable: str
    index: tuple[int, ...]
    max_abs_error: float
    max_rel_error: float
    actual_value: float
    expected_value: float


@dataclass(frozen=True)
class DatasetComparison:
    checked_variables: tuple[str, ...]
    failures: tuple[ValidationFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


class CaseRunner(Protocol):
    def __call__(self, case: ValidationCase) -> xr.Dataset: ...


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cases_root(cases_root: Path | None = None) -> Path:
    return (cases_root or _project_root() / "gotm-model" / "cases-runs").resolve()


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
    return tuple(
        resolve_reference_case(case_name, cases_root=cases_root)
        for case_name in REFERENCE_CASE_NAMES
    )


def open_reference_dataset(case: ValidationCase) -> xr.Dataset:
    return _open_dataset(case.reference_path)


def _numeric_variable_names(dataset: xr.Dataset) -> tuple[str, ...]:
    names: list[str] = []
    for name, data_array in dataset.data_vars.items():
        if np.issubdtype(data_array.dtype, np.number):
            names.append(cast(str, name))
    return tuple(names)


def numeric_variable_names(dataset: xr.Dataset) -> tuple[str, ...]:
    """Return numeric data variables in dataset order."""

    return _numeric_variable_names(dataset)


def _as_float(value: np.generic | float | int) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _normalized_numeric_values(dataset: xr.Dataset, name: str) -> np.ndarray:
    values = np.asarray(dataset[name].values)
    if values.ndim == 0:
        return np.asarray(values, dtype=np.float64)
    return np.asarray(np.squeeze(values), dtype=np.float64)


def compare_datasets(
    actual: xr.Dataset,
    expected: xr.Dataset,
    *,
    rtol: float = 5.0e-6,
    atol: float = 1.0e-12,
    variables: tuple[str, ...] | None = None,
) -> DatasetComparison:
    """Compare datasets with GOTM's range-aware combined tolerance."""

    variable_names = variables or _numeric_variable_names(expected)
    failures: list[ValidationFailure] = []

    for name in variable_names:
        if name not in actual.data_vars:
            failures.append(
                ValidationFailure(
                    variable=name,
                    index=(),
                    max_abs_error=float("inf"),
                    max_rel_error=float("inf"),
                    actual_value=float("nan"),
                    expected_value=float("nan"),
                )
            )
            continue

        expected_values = _normalized_numeric_values(expected, name)
        actual_values = _normalized_numeric_values(actual, name)

        if actual_values.shape != expected_values.shape:
            failures.append(
                ValidationFailure(
                    variable=name,
                    index=(),
                    max_abs_error=float("inf"),
                    max_rel_error=float("inf"),
                    actual_value=float("nan"),
                    expected_value=float("nan"),
                )
            )
            continue

        abs_error = np.abs(actual_values - expected_values)
        finite_expected = expected_values[np.isfinite(expected_values)]
        if finite_expected.size:
            ref_range = float(np.max(finite_expected) - np.min(finite_expected))
        else:
            ref_range = 0.0
        variable_atol = max(1.0e-7 * ref_range, atol)
        finite = np.isfinite(actual_values) & np.isfinite(expected_values)
        matches = np.zeros_like(abs_error, dtype=np.bool_)
        matches[finite] = abs_error[finite] <= (
            variable_atol + rtol * np.abs(expected_values[finite])
        )
        matches |= np.isnan(actual_values) & np.isnan(expected_values)
        matches |= np.isinf(actual_values) & (actual_values == expected_values)
        if bool(np.all(matches)):
            continue

        rel_error = np.zeros_like(abs_error, dtype=np.float64)
        nonzero = np.abs(expected_values) > 0.0
        rel_error[nonzero] = abs_error[nonzero] / np.abs(expected_values[nonzero])

        bad_mask = ~matches
        score = np.where(
            bad_mask,
            np.nan_to_num(abs_error, nan=np.inf, posinf=np.inf, neginf=np.inf),
            -np.inf,
        )
        flat_index = int(np.argmax(score))
        index = tuple(
            int(i) for i in np.unravel_index(flat_index, expected_values.shape)
        )

        failures.append(
            ValidationFailure(
                variable=name,
                index=index,
                max_abs_error=_as_float(abs_error[index]),
                max_rel_error=_as_float(rel_error[index]),
                actual_value=_as_float(actual_values[index]),
                expected_value=_as_float(expected_values[index]),
            )
        )

    return DatasetComparison(
        checked_variables=variable_names,
        failures=tuple(failures),
    )


def run_case_validation(
    case_name: str,
    *,
    module_name: str | None = None,
    actual_path: Path | None = None,
    runner: CaseRunner | None = None,
    rtol: float = 5.0e-6,
    atol: float = 1.0e-12,
    cases_root: Path | None = None,
) -> DatasetComparison:
    case = resolve_reference_case(case_name, cases_root=cases_root)

    if actual_path is not None:
        actual_dataset = _open_dataset(actual_path)
    else:
        if runner is None:
            from pygotm.driver import GotmDriver

            def driver_runner(validation_case: ValidationCase) -> xr.Dataset:
                return GotmDriver(validation_case.yaml_path).run()

            runner_func = cast(CaseRunner, driver_runner)
        else:
            runner_func = runner
        actual_dataset = runner_func(case)
    reference_dataset = open_reference_dataset(case)
    try:
        return compare_datasets(
            actual_dataset,
            reference_dataset,
            rtol=rtol,
            atol=atol,
        )
    finally:
        actual_dataset.close()
        reference_dataset.close()


@click.command(name="validate")
@click.option("--module", "module_name", help="Validation module or subsystem label.")
@click.option("--case", "case_name", help="GOTM reference case name.")
@click.option(
    "--actual",
    "actual_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a pyGOTM-generated NetCDF output file to compare.",
)
@click.option("--rtol", default=5.0e-6, show_default=True, type=float)
@click.option("--atol", default=1.0e-12, show_default=True, type=float)
@click.option("--list-cases", is_flag=True, help="List the bundled reference cases.")
def cli(
    module_name: str | None,
    case_name: str | None,
    actual_path: Path | None,
    rtol: float,
    atol: float,
    list_cases: bool,
) -> None:
    """Validate a dataset against bundled GOTM reference outputs."""
    if list_cases:
        for case in discover_reference_cases():
            click.echo(f"{case.name}\t{case.reference_path}")
        return

    if case_name is None:
        raise click.UsageError("--case is required unless --list-cases is used")

    try:
        comparison = run_case_validation(
            case_name,
            module_name=module_name,
            actual_path=actual_path,
            rtol=rtol,
            atol=atol,
        )
    except (FileNotFoundError, KeyError, NotImplementedError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if comparison.failures:
        click.echo(
            f"Validation failed for case {case_name!r}: "
            f"{len(comparison.failures)} variable(s) exceeded tolerance."
        )
        for failure in comparison.failures[:10]:
            click.echo(
                "  "
                f"{failure.variable} index={failure.index} "
                f"abs={failure.max_abs_error:.6e} "
                f"rel={failure.max_rel_error:.6e} "
                f"actual={failure.actual_value:.6e} "
                f"expected={failure.expected_value:.6e}"
            )
        raise SystemExit(1)

    click.echo(
        f"Validation passed for case {case_name!r}; "
        f"checked {len(comparison.checked_variables)} numeric variables."
    )


if __name__ == "__main__":
    cli()
