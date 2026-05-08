"""Tests for FABM configuration resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from pygotm.fabm.config import fabm_enabled, load_fabm_config, resolve_fabm_config_path


def test_disabled_fabm_does_not_require_model_yaml(tmp_path: Path) -> None:
    gotm_yaml = tmp_path / "gotm.yaml"
    gotm_yaml.write_text("fabm:\n  use: false\n", encoding="utf-8")

    config = load_fabm_config({"fabm": {"use": False}}, gotm_yaml)

    assert not config.use
    assert config.config_path is None
    assert not fabm_enabled({"fabm": {"use": False}})


def test_resolves_default_sibling_fabm_yaml(tmp_path: Path) -> None:
    gotm_yaml = tmp_path / "gotm.yaml"
    gotm_yaml.write_text("fabm:\n  use: true\n", encoding="utf-8")
    fabm_yaml = tmp_path / "fabm.yaml"
    fabm_yaml.write_text("instances: {}\n", encoding="utf-8")

    config = load_fabm_config({"fabm": {"use": True}}, gotm_yaml)

    assert config.use
    assert config.config_path == fabm_yaml.resolve()
    assert (
        resolve_fabm_config_path(gotm_yaml, {"fabm": {"use": True}})
        == fabm_yaml.resolve()
    )


def test_invalid_fabm_yaml_path_fails_loudly(tmp_path: Path) -> None:
    gotm_yaml = tmp_path / "gotm.yaml"
    gotm_yaml.write_text("fabm:\n  use: true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="FABM model YAML not found"):
        load_fabm_config({"fabm": {"use": True, "file": "missing.yaml"}}, gotm_yaml)
