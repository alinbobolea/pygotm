"""Tests for reference-case resolution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from pygotm.validation.reference import resolve_reference_case


def _write_case_assets(
    cases_root: Path,
    case_name: str,
    yaml_name: str,
) -> None:
    case_dir = cases_root / case_name
    case_dir.mkdir(parents=True)
    (case_dir / yaml_name).write_text("version: 7\n", encoding="utf-8")
    (case_dir / f"{case_name}.nc").write_bytes(b"netcdf")


def test_resolve_reference_case_accepts_non_gotm_yaml_name(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    _write_case_assets(cases_root, "blacksea", "custom_input.yaml")

    case = resolve_reference_case("blacksea", cases_root=cases_root)

    assert case.name == "blacksea"
    assert case.yaml_path == (cases_root / "blacksea" / "custom_input.yaml").resolve()
    assert case.task_name == "blacksea-custom_input"
    assert case.run_name == "blacksea-custom_input"


def test_resolve_reference_case_accepts_yaml_base_in_case_spec(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    _write_case_assets(cases_root, "blacksea", "experiment.yaml")
    (cases_root / "blacksea" / "gotm.yaml").write_text(
        "version: 7\n",
        encoding="utf-8",
    )

    case = resolve_reference_case("blacksea/experiment", cases_root=cases_root)

    assert case.name == "blacksea"
    assert case.yaml_path == (cases_root / "blacksea" / "experiment.yaml").resolve()
    assert case.task_name == "blacksea-experiment"


def test_resolve_reference_case_rejects_ambiguous_yaml_names(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    _write_case_assets(cases_root, "blacksea", "first.yaml")
    (cases_root / "blacksea" / "second.yaml").write_text(
        "version: 7\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="multiple GOTM YAML input files"):
        resolve_reference_case("blacksea", cases_root=cases_root)
