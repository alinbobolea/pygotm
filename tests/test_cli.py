"""Tests for the public pyGOTM command-line surface."""

from __future__ import annotations

from pathlib import Path

import xarray as xr
from click.testing import CliRunner

from pygotm.__main__ import cli


def test_public_cli_exposes_studio_integration_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "version" in result.output
    assert "schema" in result.output
    assert "cite" in result.output
    assert "serve" in result.output
    assert "validate" in result.output
    assert "benchmark" not in result.output


def test_run_cli_writes_output_with_driver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "gotm.yaml"
    output_path = tmp_path / "result.nc"
    config_path.write_text("version: 7\n", encoding="utf-8")
    calls: list[tuple[Path, int | None, Path]] = []

    class FakeDriver:
        def __init__(self, config: Path) -> None:
            self.config = config

        def run(
            self,
            *,
            max_steps: int | None = None,
            output_path: Path | None = None,
            progress: object | None = None,
        ) -> xr.Dataset:
            assert output_path is not None
            assert progress is None
            output_path.write_text("netcdf", encoding="utf-8")
            calls.append((self.config, max_steps, output_path))
            return xr.Dataset()

    monkeypatch.setattr("pygotm.__main__.GotmDriver", FakeDriver)

    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(config_path),
            "--output",
            str(output_path),
            "--max-steps",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert calls == [(config_path, 3, output_path)]
    assert output_path.read_text(encoding="utf-8") == "netcdf"
    assert f"Wrote {output_path}" in result.output


def test_run_cli_progress_json_writes_events_to_stderr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "gotm.yaml"
    output_path = tmp_path / "result.nc"
    config_path.write_text("version: 7\n", encoding="utf-8")

    class FakeDriver:
        def __init__(self, config: Path) -> None:
            self.config = config

        def run(
            self,
            *,
            max_steps: int | None = None,
            output_path: Path | None = None,
            progress: object | None = None,
        ) -> xr.Dataset:
            del max_steps
            assert output_path is not None
            output_path.write_text("netcdf", encoding="utf-8")
            assert progress is not None
            progress.started("initializing")
            progress.phase("integrating", progress_mode="indeterminate")
            progress.finished(exit_code=0, output_path=output_path)
            return xr.Dataset()

    monkeypatch.setattr("pygotm.__main__.GotmDriver", FakeDriver)

    result = CliRunner().invoke(
        cli,
        ["run", str(config_path), "--output", str(output_path), "--progress", "json"],
    )

    assert result.exit_code == 0
    assert '"event":"started"' in result.output
    assert '"progress_mode":"indeterminate"' in result.output
    assert '"event":"finished"' in result.output


def test_run_cli_maps_missing_output_parent_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "gotm.yaml"
    output_path = tmp_path / "result.nc"
    config_path.write_text("version: 7\n", encoding="utf-8")

    class FakeDriver:
        def __init__(self, config: Path) -> None:
            self.config = config

        def run(
            self,
            *,
            max_steps: int | None = None,
            output_path: Path | None = None,
            progress: object | None = None,
        ) -> xr.Dataset:
            del max_steps, output_path, progress
            raise FileNotFoundError("missing input")

    monkeypatch.setattr("pygotm.__main__.GotmDriver", FakeDriver)

    result = CliRunner().invoke(
        cli,
        ["run", str(config_path), "--output", str(output_path)],
    )

    assert result.exit_code == 13
    assert "ERROR[13]" in result.output


def test_version_cli_json_uses_manifest_keys() -> None:
    result = CliRunner().invoke(cli, ["version", "--json"])

    assert result.exit_code == 0
    for key in (
        "pygotm_version",
        "pygotm_git_commit",
        "python_version",
        "numpy_version",
        "numba_version",
        "xarray_version",
        "netcdf4_version",
        "gsw_version",
        "pyfabm_version",
        "platform",
    ):
        assert f'"{key}"' in result.output


def test_validate_cli_uses_frechet_suite_options() -> None:
    result = CliRunner().invoke(cli, ["validate", "--help"])

    assert result.exit_code == 0
    assert "--cases" in result.output
    assert "--all" in result.output
    assert "--case " not in result.output
    assert "--rtol" not in result.output
    assert "--atol" not in result.output
