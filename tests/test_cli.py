"""Tests for the public pyGOTM command-line surface."""

from __future__ import annotations

from pathlib import Path

import xarray as xr
from click.testing import CliRunner

from pygotm.__main__ import cli


def test_public_cli_exposes_run_and_validate_only() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.output
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
        ) -> xr.Dataset:
            assert output_path is not None
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


def test_validate_cli_uses_frechet_suite_options() -> None:
    result = CliRunner().invoke(cli, ["validate", "--help"])

    assert result.exit_code == 0
    assert "--cases" in result.output
    assert "--all" in result.output
    assert "--case " not in result.output
    assert "--rtol" not in result.output
    assert "--atol" not in result.output
