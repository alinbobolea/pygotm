"""Reproducible bundle config staging in the validation runner.

Regression tests for the Aquirae reference-bundle contract: ``run_case`` must
stage the exact configs a run consumed into ``validation/runs/<case>`` so the
bundle's ``fabm.yaml`` is byte-identical to the NetCDF ``fabm_yaml_sha256``
attribute (the legacy GOTM-lake ``selmaprotbas`` ``alpha``/``beta`` parameters
are stripped for conda ``pyfabm`` compatibility) and re-running the bundle
reproduces the recorded config hashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pygotm.fabm.config import resolve_fabm_config_path
from pygotm.gotm.run_metadata import file_sha256
from pygotm.validation.runner import _stage_bundle_configs

_FABM_WITH_LEGACY = """\
instances:
  diatoms:
    model: selmaprotbas/phytoplankton
    parameters:
      c0: 0.01
      alpha: 1.65625
      beta: 0.0
  flagellates:
    model: selmaprotbas/phytoplankton
    parameters:
      c0: 0.02
      alpha: 1.65625
      beta: 0.0
"""


class _FakeConfig:
    """Minimal stand-in exposing the two attributes the stager reads."""

    def __init__(self, source_path: Path, document: dict[str, Any]) -> None:
        self.source_path = source_path
        self._document = document

    def resolved_document(self) -> dict[str, Any]:
        return self._document


def _write_case(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    src = tmp_path / "src"
    src.mkdir()
    (src / "gotm.yaml").write_text(
        "fabm:\n  use: true\n  config_file: fabm.yaml\n", encoding="utf-8"
    )
    (src / "fabm.yaml").write_text(_FABM_WITH_LEGACY, encoding="utf-8")
    document = {"fabm": {"use": True, "config_file": "fabm.yaml"}}
    return src / "gotm.yaml", document


def test_stage_bundle_configs_writes_materialized_fabm(tmp_path: Path) -> None:
    gotm_path, document = _write_case(tmp_path)
    bundle = tmp_path / "runs" / "case"
    bundle.mkdir(parents=True)

    # The materialized config the driver actually hashes into the NetCDF.
    materialized = resolve_fabm_config_path(gotm_path, document)

    _stage_bundle_configs(_FakeConfig(gotm_path, document), bundle)

    staged_gotm = bundle / "gotm.yaml"
    staged_fabm = bundle / "fabm.yaml"
    assert staged_gotm.is_file()
    assert staged_fabm.is_file()

    # Legacy GOTM-lake selmaprotbas params are stripped in the staged sidecar.
    raw = yaml.safe_load(staged_fabm.read_text(encoding="utf-8"))
    for instance in raw["instances"].values():
        params = instance["parameters"]
        assert "alpha" not in params
        assert "beta" not in params
        assert "c0" in params

    # Staged sidecar == the materialized config the run consumes
    # (i.e. == NetCDF fabm_yaml_sha256).
    assert file_sha256(staged_fabm) == file_sha256(materialized)


def test_staged_bundle_fabm_is_reproducible(tmp_path: Path) -> None:
    gotm_path, document = _write_case(tmp_path)
    bundle = tmp_path / "runs" / "case"
    bundle.mkdir(parents=True)
    original = file_sha256(resolve_fabm_config_path(gotm_path, document))

    _stage_bundle_configs(_FakeConfig(gotm_path, document), bundle)

    # Re-resolving from the staged bundle yields the same config hash: a
    # downstream consumer re-running the bundle reproduces fabm_yaml_sha256.
    restaged = resolve_fabm_config_path(bundle / "gotm.yaml", document)
    assert file_sha256(restaged) == original
    assert file_sha256(bundle / "fabm.yaml") == original


def test_stage_bundle_configs_noop_without_fabm(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "gotm.yaml").write_text("title: plain\n", encoding="utf-8")
    bundle = tmp_path / "runs" / "case"
    bundle.mkdir(parents=True)

    _stage_bundle_configs(_FakeConfig(src / "gotm.yaml", {"title": "plain"}), bundle)

    assert (bundle / "gotm.yaml").is_file()
    assert not (bundle / "fabm.yaml").exists()
