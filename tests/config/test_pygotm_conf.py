"""Unit tests for load_pygotm_conf."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pygotm.config.pygotm_conf import FABMConf, PyGotmConf, load_pygotm_conf


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    yaml_path = tmp_path / "gotm.yaml"
    conf = load_pygotm_conf(yaml_path)
    assert conf == PyGotmConf(fabm=FABMConf(chunk_size=None))


def test_chunk_size_present(tmp_path: Path) -> None:
    yaml_path = tmp_path / "gotm.yaml"
    (tmp_path / "pygotm-conf.yaml").write_text(textwrap.dedent("""\
        fabm:
          chunk_size: 24
        """))
    conf = load_pygotm_conf(yaml_path)
    assert conf.fabm.chunk_size == 24


def test_no_fabm_key_returns_defaults(tmp_path: Path) -> None:
    yaml_path = tmp_path / "gotm.yaml"
    (tmp_path / "pygotm-conf.yaml").write_text("other_key: value\n")
    conf = load_pygotm_conf(yaml_path)
    assert conf.fabm.chunk_size is None


def test_chunk_size_zero_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "gotm.yaml"
    (tmp_path / "pygotm-conf.yaml").write_text("fabm:\n  chunk_size: 0\n")
    with pytest.raises(ValueError, match="chunk_size"):
        load_pygotm_conf(yaml_path)


def test_chunk_size_negative_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "gotm.yaml"
    (tmp_path / "pygotm-conf.yaml").write_text("fabm:\n  chunk_size: -1\n")
    with pytest.raises(ValueError, match="chunk_size"):
        load_pygotm_conf(yaml_path)


def test_empty_file_returns_defaults(tmp_path: Path) -> None:
    yaml_path = tmp_path / "gotm.yaml"
    (tmp_path / "pygotm-conf.yaml").write_text("")
    conf = load_pygotm_conf(yaml_path)
    assert conf == PyGotmConf(fabm=FABMConf(chunk_size=None))
