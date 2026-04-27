"""Tests for pygotm.turbulence.gotm_lib_version."""

from __future__ import annotations

from io import StringIO

from pygotm.turbulence.gotm_lib_version import gotm_lib_version
from pygotm.util.gotm_version import git_commit_id


def test_import() -> None:
    assert callable(gotm_lib_version)


def test_writes_expected_version_line() -> None:
    buffer = StringIO()

    gotm_lib_version(buffer)

    assert buffer.getvalue() == f"GOTM library version: {git_commit_id}\n"
