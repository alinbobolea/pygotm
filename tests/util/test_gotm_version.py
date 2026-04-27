"""Tests for util/gotm_version.py — GOTM version string stub."""

import pygotm.util.gotm_version as gotm_version


def test_import() -> None:
    assert gotm_version is not None


def test_git_commit_id_is_string() -> None:
    assert isinstance(gotm_version.git_commit_id, str)


def test_git_branch_name_is_string() -> None:
    assert isinstance(gotm_version.git_branch_name, str)


def test_all_exports() -> None:
    assert set(gotm_version.__all__) == {"git_commit_id", "git_branch_name"}


def test_default_commit_id() -> None:
    assert gotm_version.git_commit_id == "4.1.0"


def test_default_branch_name() -> None:
    assert gotm_version.git_branch_name == "master"
