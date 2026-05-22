"""Unit tests for the Sphinx validation-HTML staging helpers in build_docs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from build_docs import (  # noqa: E402
    GOTM_CASES,
    stage_validation_html,
    stage_validation_rst_wrappers,
)


def _write(path: Path, body: str = "<html></html>") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# stage_validation_html
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# stage_validation_rst_wrappers
# ---------------------------------------------------------------------------


def test_rst_wrappers_created_for_all_cases(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    result = stage_validation_rst_wrappers(cases_dir=cases_dir)

    assert cases_dir.is_dir()
    assert len(result) == len(GOTM_CASES)
    for case in GOTM_CASES:
        rst = cases_dir / f"{case}.rst"
        assert rst in result
        assert rst.exists()


def test_rst_wrappers_contain_iframe_with_correct_src(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    stage_validation_rst_wrappers(cases_dir=cases_dir)

    for case in GOTM_CASES:
        content = (cases_dir / f"{case}.rst").read_text(encoding="utf-8")
        assert f"../{case}-gotm.html" in content
        assert "iframe" in content
        assert ".. raw:: html" in content


def test_rst_wrappers_refresh_on_second_call(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    stage_validation_rst_wrappers(cases_dir=cases_dir)

    # Inject a stale file that should be removed on the next call.
    stale = cases_dir / "stale_case.rst"
    stale.write_text("stale content", encoding="utf-8")

    stage_validation_rst_wrappers(cases_dir=cases_dir)

    assert not stale.exists()
    assert len(list(cases_dir.glob("*.rst"))) == len(GOTM_CASES)
