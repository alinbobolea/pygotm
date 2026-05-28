"""Regression tests for package metadata that affects release artifacts."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_does_not_advertise_pip_fabm_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert "fabm" not in optional
    for requirements in optional.values():
        assert all("pyfabm" not in requirement for requirement in requirements)


def test_pyproject_force_includes_gotm_saar_data_in_wheel() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]

    assert (
        force_include["src/pygotm/util/gsw/data/saar_2011_gotm.npz"]
        == "pygotm/util/gsw/data/saar_2011_gotm.npz"
    )
    assert (
        force_include["src/pygotm/util/gsw/data/README.md"]
        == "pygotm/util/gsw/data/README.md"
    )
