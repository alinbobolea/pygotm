"""Unit tests for the Sphinx validation-HTML staging helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from _validation_staging import stage_validation_html  # noqa: E402


def _write(path: Path, body: str = "<html></html>") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_stages_top_level_report_and_case_files(tmp_path: Path) -> None:
    src = tmp_path / "validation"
    staged_root = tmp_path / "_validation_html"

    _write(src / "report.html", "<html>index</html>")
    _write(src / "couette-gotm.html", "<html>couette</html>")
    _write(src / "gotland-gotm.html", "<html>gotland</html>")
    _write(src / "runs" / "couette" / "couette.nc")
    _write(src / "reference" / "couette" / "couette.nc")

    copied = stage_validation_html(src=src, staged_root=staged_root)

    staged = staged_root / "validation"
    assert (staged / "report.html").read_text(encoding="utf-8") == "<html>index</html>"
    assert (staged / "couette-gotm.html").exists()
    assert (staged / "gotland-gotm.html").exists()
    assert not (staged / "runs").exists()
    assert not (staged / "reference").exists()
    assert {p.name for p in copied} == {
        "report.html",
        "couette-gotm.html",
        "gotland-gotm.html",
    }


def test_refreshes_previous_staging(tmp_path: Path) -> None:
    src = tmp_path / "validation"
    staged_root = tmp_path / "_validation_html"

    _write(src / "report.html", "<html>v1</html>")
    _write(src / "couette-gotm.html", "<html>old case</html>")
    stage_validation_html(src=src, staged_root=staged_root)

    (src / "couette-gotm.html").unlink()
    _write(src / "report.html", "<html>v2</html>")
    _write(src / "channel-gotm.html", "<html>new case</html>")

    stage_validation_html(src=src, staged_root=staged_root)

    staged = staged_root / "validation"
    assert (staged / "report.html").read_text(encoding="utf-8") == "<html>v2</html>"
    assert (staged / "channel-gotm.html").exists()
    assert not (staged / "couette-gotm.html").exists()


def test_missing_validation_directory_is_not_an_error(tmp_path: Path) -> None:
    src = tmp_path / "validation"  # never created
    staged_root = tmp_path / "_validation_html"

    copied = stage_validation_html(src=src, staged_root=staged_root)

    assert copied == []
    assert (staged_root / "validation").is_dir()


def test_empty_validation_directory_produces_empty_staging(tmp_path: Path) -> None:
    src = tmp_path / "validation"
    src.mkdir()
    staged_root = tmp_path / "_validation_html"

    copied = stage_validation_html(src=src, staged_root=staged_root)

    assert copied == []
    assert (staged_root / "validation").is_dir()
    assert list((staged_root / "validation").iterdir()) == []
