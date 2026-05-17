"""Tests for validation run selection helpers."""

from __future__ import annotations

import pytest

from pygotm.validation.report import CaseResult
from pygotm.validation.run_validation import (
    ALL_CASES,
    DEFAULT_CASES,
    NON_STIM_CASES,
    _make_on_result,
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


def test_on_result_reports_completed_case_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    on_result = _make_on_result(2)

    on_result(
        CaseResult(
            case_name="couette",
            status="FAIL",
            error=None,
            py_nc_path="couette.nc",
            ref_nc_path="ref.nc",
            wall_time_s=10.0,
            task_name="couette-gotm",
            n_pass=4,
            n_broken=1,
        )
    )
    on_result(
        CaseResult(
            case_name="channel",
            status="PASS",
            error=None,
            py_nc_path="channel.nc",
            ref_nc_path="ref.nc",
            wall_time_s=11.0,
            task_name="channel-gotm",
            n_pass=5,
        )
    )

    assert capsys.readouterr().out.splitlines() == [
        "Complete case: couette-gotm [10.0s] | 1/2 cases",
        "Complete case: channel-gotm [11.0s] | 2/2 cases",
    ]
