"""Benchmark helpers for the compiled single-column validation runtime."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import click
import xarray as xr

from pygotm.gotm.gotm import finalize_gotm, initialize_gotm
from pygotm.gotm.runtime_builder import (
    RuntimeBundle,
    UnsupportedConfigurationError,
    build_runtime_from_run,
    runtime_output_to_dataset,
)
from pygotm.gotm.time_loop import time_loop_compiled
from pygotm.validate import (
    DatasetComparison,
    compare_datasets,
    open_reference_dataset,
    resolve_reference_case,
)
from pygotm.validation.warmup import trigger_numba_jit

__all__ = [
    "BenchmarkResult",
    "BenchmarkTimings",
    "benchmark_cases",
    "benchmark_compiled_case",
    "cli",
    "save_benchmark_results",
]

BenchmarkStatus = Literal["PASS", "FAIL", "ERROR", "UNSUPPORTED"]
ValidationStatus = Literal["PASS", "FAIL", "SKIP", "ERROR"]


@dataclass(slots=True, frozen=True)
class BenchmarkTimings:
    """Measured wall-clock timings for one benchmarked case."""

    warmup_s: float
    initialization_s: float
    runtime_build_s: float
    integration_s: float
    output_conversion_s: float
    validation_s: float
    total_s: float


@dataclass(slots=True, frozen=True)
class BenchmarkResult:
    """Benchmark result for one compiled single-column case."""

    case_name: str
    status: BenchmarkStatus
    validation_status: ValidationStatus
    error: str | None
    compiled_function: str
    nopython_signature_count: int
    n_steps: int
    n_output: int
    checked_variables: tuple[str, ...]
    failed_variables: tuple[str, ...]
    timings: BenchmarkTimings


def _zero_timings(total_s: float = 0.0) -> BenchmarkTimings:
    return BenchmarkTimings(
        warmup_s=0.0,
        initialization_s=0.0,
        runtime_build_s=0.0,
        integration_s=0.0,
        output_conversion_s=0.0,
        validation_s=0.0,
        total_s=total_s,
    )


def _compiled_function_for_bundle(bundle: RuntimeBundle) -> Any:
    return time_loop_compiled


def _validation_status(
    comparison: DatasetComparison | None,
) -> tuple[ValidationStatus, tuple[str, ...], tuple[str, ...]]:
    if comparison is None:
        return "SKIP", (), ()
    failed = tuple(failure.variable for failure in comparison.failures)
    return ("PASS" if not failed else "FAIL"), comparison.checked_variables, failed


def _benchmark_status(validation_status: ValidationStatus) -> BenchmarkStatus:
    return "FAIL" if validation_status == "FAIL" else "PASS"


def benchmark_compiled_case(
    case_name: str,
    *,
    max_steps: int | None = None,
    output: bool = True,
    warmup: bool = True,
    validate: bool = True,
) -> BenchmarkResult:
    """Run one case through the compiled runtime and record benchmark timings."""

    total_t0 = time.perf_counter()
    warmup_s = 0.0
    initialization_s = 0.0
    runtime_build_s = 0.0
    integration_s = 0.0
    output_conversion_s = 0.0
    validation_s = 0.0
    n_steps = 0
    n_output = 0
    compiled_function = ""
    signature_count = 0
    actual: xr.Dataset | None = None
    reference: xr.Dataset | None = None
    run: Any | None = None
    result_case_name = case_name

    try:
        if warmup:
            warmup_s = trigger_numba_jit()

        case = resolve_reference_case(case_name)
        result_case_name = case.run_name

        t0 = time.perf_counter()
        run = initialize_gotm(case.yaml_path)
        initialization_s = time.perf_counter() - t0

        try:
            t0 = time.perf_counter()
            bundle = build_runtime_from_run(run, max_steps=max_steps, output=output)
            runtime_build_s = time.perf_counter() - t0
            n_steps = bundle.params.nt

            dispatcher = _compiled_function_for_bundle(bundle)
            compiled_function = str(dispatcher.py_func.__name__)

            t0 = time.perf_counter()
            written = bundle.run()
            integration_s = time.perf_counter() - t0
            n_output = written
            signature_count = len(dispatcher.nopython_signatures)
            if written != bundle.output.nout:
                msg = (
                    f"compiled loop wrote {written} output slots; "
                    f"expected {bundle.output.nout}"
                )
                raise RuntimeError(msg)

            comparison: DatasetComparison | None = None
            if output:
                t0 = time.perf_counter()
                actual = runtime_output_to_dataset(run, bundle)
                output_conversion_s = time.perf_counter() - t0

                if validate:
                    t0 = time.perf_counter()
                    reference = open_reference_dataset(case)
                    reference_for_compare = reference
                    if "time" in actual.sizes and "time" in reference.sizes:
                        reference_for_compare = reference.isel(
                            time=slice(0, actual.sizes["time"])
                        ).squeeze(drop=True)
                    comparison = compare_datasets(actual, reference_for_compare)
                    validation_s = time.perf_counter() - t0

            validation_status, checked, failed = _validation_status(comparison)
            timings = BenchmarkTimings(
                warmup_s=warmup_s,
                initialization_s=initialization_s,
                runtime_build_s=runtime_build_s,
                integration_s=integration_s,
                output_conversion_s=output_conversion_s,
                validation_s=validation_s,
                total_s=time.perf_counter() - total_t0,
            )
            return BenchmarkResult(
                case_name=result_case_name,
                status=_benchmark_status(validation_status),
                validation_status=validation_status,
                error=None,
                compiled_function=compiled_function,
                nopython_signature_count=signature_count,
                n_steps=n_steps,
                n_output=n_output,
                checked_variables=checked,
                failed_variables=failed,
                timings=timings,
            )
        finally:
            if actual is not None:
                actual.close()
            if reference is not None:
                reference.close()
            finalize_gotm(run)
    except UnsupportedConfigurationError as exc:
        timings = BenchmarkTimings(
            warmup_s=warmup_s,
            initialization_s=initialization_s,
            runtime_build_s=runtime_build_s,
            integration_s=integration_s,
            output_conversion_s=output_conversion_s,
            validation_s=validation_s,
            total_s=time.perf_counter() - total_t0,
        )
        return BenchmarkResult(
            case_name=result_case_name,
            status="UNSUPPORTED",
            validation_status="SKIP",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            compiled_function=compiled_function,
            nopython_signature_count=signature_count,
            n_steps=n_steps,
            n_output=n_output,
            checked_variables=(),
            failed_variables=(),
            timings=timings,
        )
    except Exception as exc:
        timings = BenchmarkTimings(
            warmup_s=warmup_s,
            initialization_s=initialization_s,
            runtime_build_s=runtime_build_s,
            integration_s=integration_s,
            output_conversion_s=output_conversion_s,
            validation_s=validation_s,
            total_s=time.perf_counter() - total_t0,
        )
        return BenchmarkResult(
            case_name=result_case_name,
            status="ERROR",
            validation_status="ERROR",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            compiled_function=compiled_function,
            nopython_signature_count=signature_count,
            n_steps=n_steps,
            n_output=n_output,
            checked_variables=(),
            failed_variables=(),
            timings=timings,
        )


def benchmark_cases(
    case_names: Sequence[str],
    *,
    max_steps: int | None = None,
    output: bool = True,
    warmup: bool = True,
    validate: bool = True,
) -> tuple[BenchmarkResult, ...]:
    """Benchmark cases sequentially, warming up only before the first case."""

    results: list[BenchmarkResult] = []
    should_warmup = warmup
    for case_name in case_names:
        results.append(
            benchmark_compiled_case(
                case_name,
                max_steps=max_steps,
                output=output,
                warmup=should_warmup,
                validate=validate,
            )
        )
        should_warmup = False
    return tuple(results)


def save_benchmark_results(results: Sequence[BenchmarkResult], path: Path) -> None:
    """Write benchmark results to one aggregate JSON artifact."""

    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = {
        "generated_at": generated_at,
        "cases": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _format_result(result: BenchmarkResult) -> str:
    timings = result.timings
    return (
        f"{result.status:<11} {result.case_name:<18} "
        f"validation={result.validation_status:<5} "
        f"loop={result.compiled_function or '-'} "
        f"steps={result.n_steps} "
        f"integrate={timings.integration_s:.3f}s "
        f"convert={timings.output_conversion_s:.3f}s "
        f"total={timings.total_s:.3f}s"
    )


@click.command(name="benchmark")
@click.option(
    "--cases",
    default="couette,channel",
    show_default=True,
    help="Comma-separated reference case names to benchmark.",
)
@click.option("--max-steps", default=None, type=int, help="Limit integration steps.")
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Optional directory for one aggregate benchmark JSON output.",
)
@click.option("--no-output", "output", is_flag=True, flag_value=False, default=True)
@click.option("--no-warmup", "warmup", is_flag=True, flag_value=False, default=True)
@click.option("--no-validate", "validate", is_flag=True, flag_value=False, default=True)
def cli(
    cases: str,
    max_steps: int | None,
    output_dir: Path | None,
    output: bool,
    warmup: bool,
    validate: bool,
) -> None:
    """Benchmark the compiled single-column validation runtime."""

    case_names = tuple(case.strip() for case in cases.split(",") if case.strip())
    results = benchmark_cases(
        case_names,
        max_steps=max_steps,
        output=output,
        warmup=warmup,
        validate=validate,
    )

    for result in results:
        click.echo(_format_result(result))
        if result.error:
            click.echo((result.error.splitlines() or ["unknown error"])[0])

    if output_dir is not None:
        json_path = output_dir / "results.json"
        save_benchmark_results(results, json_path)
        click.echo(f"Wrote {json_path}")
