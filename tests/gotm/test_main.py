"""Tests for pygotm.gotm.main."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from pygotm.gotm.main import main


def test_main_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    help_output = capsys.readouterr().out
    assert "Usage: gotm [OPTIONS]" in help_output

    assert main(["--version"]) == 0
    version_output = capsys.readouterr().out
    assert "pyGOTM:" in version_output


def test_main_write_yaml_without_existing_input(tmp_path: Path) -> None:
    out_path = tmp_path / "generated.yaml"
    assert main(["--write_yaml", str(out_path)]) == 0
    assert out_path.is_file()


def test_main_runs_with_minimal_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 7,
                "location": {"latitude": 0.0, "longitude": 0.0, "depth": 5.0},
                "time": {
                    "start": "2000-01-01 00:00:00",
                    "stop": "2000-01-01 00:10:00",
                    "dt": 600.0,
                },
                "grid": {"nlev": 2},
                "temperature": {"method": "off"},
                "salinity": {"method": "off"},
            }
        ),
        encoding="utf-8",
    )
    assert main([str(config_path), "--list_variables"]) == 0
