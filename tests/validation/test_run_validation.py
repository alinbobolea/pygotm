"""Tests for validation run selection helpers."""

from __future__ import annotations

from pygotm.validation.run_validation import (
    ALL_CASES,
    DEFAULT_CASES,
    NON_STIM_CASES,
    _select_case_list,
)


def test_non_stim_group_excludes_stim_cases() -> None:
    selected = _select_case_list(
        cases=None,
        run_all=False,
        group="non-stim",
        exclude=None,
    )

    assert tuple(selected) == NON_STIM_CASES
    assert "plume" not in selected
    assert "resolute" not in selected


def test_exclude_filters_selected_cases() -> None:
    selected = _select_case_list(
        cases=None,
        run_all=True,
        group=None,
        exclude="plume,resolute",
    )

    assert selected == [case for case in ALL_CASES if case not in {"plume", "resolute"}]


def test_explicit_cases_take_precedence_over_default_group() -> None:
    selected = _select_case_list(
        cases="couette,channel",
        run_all=False,
        group=None,
        exclude=None,
    )

    assert selected == ["couette", "channel"]
    assert tuple(selected) != DEFAULT_CASES
