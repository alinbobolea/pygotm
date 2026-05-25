"""Tests for the curated citation database."""

from __future__ import annotations

from pygotm.citations.extractor import BIB_PATH, citation_records, load_bibliography


def test_curated_bibtex_parses_with_pybtex() -> None:
    bibliography = load_bibliography()

    assert "gotm" in bibliography.entries
    assert "pygotm" in bibliography.entries
    assert "kondo1975" in bibliography.entries


def test_citation_database_has_license_file() -> None:
    assert (
        BIB_PATH.with_name("LICENSE").read_text(encoding="utf-8").startswith("CC0-1.0")
    )


def test_all_citation_records_are_json_ready() -> None:
    records = citation_records()

    assert "gotm" in records["citation_keys"]
    assert records["citations"]
