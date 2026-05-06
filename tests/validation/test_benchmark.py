"""Tests for compiled-runtime validation benchmark helpers."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import xarray as xr
from click.testing import CliRunner

from pygotm.gotm.runtime_builder import UnsupportedConfigurationError
from pygotm.gotm.time_loop import run_compiled_time_loop
from pygotm.validate import DatasetComparison, ValidationCase
from pygotm.validation.benchmark import (
    BenchmarkResult,
    BenchmarkTimings,
    benchmark_cases,
    benchmark_compiled_case,
    cli,
    save_benchmark_results,
)


class _FakeBundle:
    def __init__(self) -> None:
        self.params = SimpleNamespace(nt=4)
        self.output = SimpleNamespace(nout=2)
        self.runner = run_compiled_time_loop

    def run(self) -> int:
        return 2


def _fake_dataset() -> xr.Dataset:
    return xr.Dataset({"u": (("time", "z"), np.zeros((2, 3), dtype=np.float64))})


def _fake_case(name: str) -> ValidationCase:
    return ValidationCase(
        name=name,
        directory=Path(name),
        yaml_path=Path("gotm.yaml"),
        reference_path=Path(f"{name}.nc"),
    )


def _sample_result(
    case_name: str = "couette",
    warmup_s: float = 0.0,
) -> BenchmarkResult:
    return BenchmarkResult(
        case_name=case_name,
        status="PASS",
        validation_status="SKIP",
        error=None,
        compiled_function="time_loop_compiled",
        nopython_signature_count=0,
        n_steps=1,
        n_output=0,
        checked_variables=(),
        failed_variables=(),
        timings=BenchmarkTimings(
            warmup_s=warmup_s,
            initialization_s=0.0,
            runtime_build_s=0.0,
            integration_s=0.0,
            output_conversion_s=0.0,
            validation_s=0.0,
            total_s=0.0,
        ),
    )


def test_benchmark_compiled_case_records_timings_and_validation() -> None:
    fake_case = _fake_case("couette")
    fake_run = object()
    fake_bundle = _FakeBundle()
    comparison = DatasetComparison(checked_variables=("u",), failures=())

    with ExitStack() as stack:
        stack.enter_context(
            patch("pygotm.validation.benchmark.trigger_numba_jit", return_value=0.25)
        )
        stack.enter_context(
            patch(
                "pygotm.validation.benchmark.resolve_reference_case",
                return_value=fake_case,
            )
        )
        stack.enter_context(
            patch("pygotm.validation.benchmark.initialize_gotm", return_value=fake_run)
        )
        stack.enter_context(
            patch(
                "pygotm.validation.benchmark.build_runtime_from_run",
                return_value=fake_bundle,
            )
        )
        stack.enter_context(
            patch(
                "pygotm.validation.benchmark.runtime_output_to_dataset",
                return_value=_fake_dataset(),
            )
        )
        stack.enter_context(
            patch(
                "pygotm.validation.benchmark.open_reference_dataset",
                return_value=_fake_dataset(),
            )
        )
        stack.enter_context(
            patch(
                "pygotm.validation.benchmark.compare_datasets",
                return_value=comparison,
            )
        )
        finalize = stack.enter_context(
            patch("pygotm.validation.benchmark.finalize_gotm")
        )
        result = benchmark_compiled_case("couette")

    assert result.status == "PASS"
    assert result.validation_status == "PASS"
    assert result.compiled_function == "time_loop_compiled"
    assert result.n_steps == 4
    assert result.n_output == 2
    assert result.checked_variables == ("u",)
    assert result.timings.warmup_s == 0.25
    assert result.timings.integration_s >= 0.0
    finalize.assert_called_once_with(fake_run)


def test_benchmark_compiled_case_reports_unsupported_errors() -> None:
    fake_case = _fake_case("blacksea")
    fake_run = object()

    with ExitStack() as stack:
        stack.enter_context(
            patch("pygotm.validation.benchmark.trigger_numba_jit", return_value=0.0)
        )
        stack.enter_context(
            patch(
                "pygotm.validation.benchmark.resolve_reference_case",
                return_value=fake_case,
            )
        )
        stack.enter_context(
            patch("pygotm.validation.benchmark.initialize_gotm", return_value=fake_run)
        )
        stack.enter_context(
            patch(
                "pygotm.validation.benchmark.build_runtime_from_run",
                side_effect=UnsupportedConfigurationError("surface fluxes"),
            )
        )
        finalize = stack.enter_context(
            patch("pygotm.validation.benchmark.finalize_gotm")
        )
        result = benchmark_compiled_case("blacksea")

    assert result.status == "UNSUPPORTED"
    assert result.validation_status == "SKIP"
    assert result.error is not None
    assert "UnsupportedConfigurationError" in result.error
    assert "surface fluxes" in result.error
    finalize.assert_called_once_with(fake_run)


def test_benchmark_cases_warms_up_once() -> None:
    def fake_benchmark(case_name: str, **kwargs: Any) -> BenchmarkResult:
        return _sample_result(
            case_name=case_name,
            warmup_s=1.0 if bool(kwargs["warmup"]) else 0.0,
        )

    with patch(
        "pygotm.validation.benchmark.benchmark_compiled_case",
        side_effect=fake_benchmark,
    ):
        results = benchmark_cases(("couette", "channel"), warmup=True)

    assert results[0].timings.warmup_s == 1.0
    assert results[1].timings.warmup_s == 0.0


def test_benchmark_cli_prints_without_json_by_default(tmp_path: Path) -> None:
    runner = CliRunner()

    with patch(
        "pygotm.validation.benchmark.benchmark_cases",
        return_value=(_sample_result(),),
    ):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["--cases", "couette"])

            assert result.exit_code == 0
            assert not Path("validation").exists()

    assert "PASS" in result.output
    assert "Wrote" not in result.output


def test_benchmark_cli_output_dir_writes_only_aggregate_json(tmp_path: Path) -> None:
    runner = CliRunner()

    with patch(
        "pygotm.validation.benchmark.benchmark_cases",
        return_value=(_sample_result(),),
    ):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                ["--cases", "couette", "--output-dir", "validation"],
            )

            assert result.exit_code == 0
            assert Path("validation/results.json").is_file()
            assert not list(Path("validation").glob("*_benchmark.json"))

    assert "Wrote validation/results.json" in result.output


def test_save_benchmark_results_writes_only_aggregate_json(tmp_path: Path) -> None:
    result = BenchmarkResult(
        case_name="couette",
        status="PASS",
        validation_status="PASS",
        error=None,
        compiled_function="time_loop_compiled",
        nopython_signature_count=1,
        n_steps=2,
        n_output=3,
        checked_variables=("u",),
        failed_variables=(),
        timings=BenchmarkTimings(
            warmup_s=0.1,
            initialization_s=0.2,
            runtime_build_s=0.3,
            integration_s=0.4,
            output_conversion_s=0.5,
            validation_s=0.6,
            total_s=2.1,
        ),
    )
    path = tmp_path / "validation" / "results.json"

    save_benchmark_results((result,), path)

    raw = json.loads(path.read_text())
    assert raw["cases"][0]["case_name"] == "couette"
    assert raw["cases"][0]["timings"]["integration_s"] == 0.4
    assert sorted(p.name for p in path.parent.iterdir()) == ["results.json"]
    assert not list(path.parent.glob("*_benchmark.json"))
