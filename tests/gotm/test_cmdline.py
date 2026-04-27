"""Tests for pygotm.gotm.cmdline."""

from __future__ import annotations

from pathlib import Path

import pytest

from pygotm.gotm.cmdline import format_help, parse_cmdline


def test_format_help_mentions_core_options() -> None:
    help_text = format_help("gotm")
    assert "Usage: gotm [OPTIONS]" in help_text
    assert "--write_yaml" in help_text


def test_parse_cmdline_supports_yaml_and_detail(tmp_path: Path) -> None:
    yaml_path = tmp_path / "gotm.yaml"
    yaml_path.write_text("version: 7\n", encoding="utf-8")
    options = parse_cmdline(
        [str(yaml_path), "--write_yaml", "out.yaml", "--detail", "full"]
    )
    assert options.yaml_file == str(yaml_path)
    assert options.write_yaml_path == "out.yaml"
    assert options.write_yaml_detail == 2


def test_parse_cmdline_handles_help_and_version() -> None:
    assert parse_cmdline(["--help"]).show_help
    assert parse_cmdline(["--version"]).show_version


def test_parse_cmdline_rejects_unknown_option() -> None:
    with pytest.raises(ValueError, match="not recognized"):
        parse_cmdline(["--unknown"])


def test_parse_cmdline_rejects_missing_yaml_file() -> None:
    with pytest.raises(FileNotFoundError):
        parse_cmdline(["missing.yaml"])
