"""Version-source consistency tests."""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

import pygotm


def test_pyproject_is_version_source() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    expected = pyproject["project"]["version"]

    try:
        installed = metadata.version("pygotm")
    except metadata.PackageNotFoundError:
        installed = "unavailable"

    assert pygotm.__version__ == installed
    if installed != "unavailable":
        assert installed == expected
