"""Tests for validation/parallel.py — Dask parallel runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pygotm.validation.parallel import _run_case_worker, run_cases_parallel
from pygotm.validation.report import CaseResult


def _fake_result(name: str) -> CaseResult:
    return CaseResult(
        case_name=name, status="PASS", error=None,
        py_nc_path=f"/runs/{name}.nc", ref_nc_path=f"/ref/{name}.nc",
        wall_time_s=0.1,
    )


def test_run_case_worker_inits_taichi_with_correct_arch(tmp_path: Path) -> None:
    import taichi as ti
    from taichi.lang import impl as ti_impl

    init_calls: list[object] = []

    def fake_init(**kwargs: object) -> None:
        init_calls.append(kwargs.get("arch"))

    mock_runtime = MagicMock()
    mock_runtime.prog = None

    with patch.object(ti_impl, "get_runtime", return_value=mock_runtime), \
         patch("taichi.init", side_effect=fake_init), \
         patch("pygotm.validation.parallel.validate_case", return_value=_fake_result("couette")):
        _run_case_worker("couette", tmp_path, "cpu", skip_run=False)

    assert len(init_calls) == 1
    assert init_calls[0] is ti.cpu


def test_run_case_worker_skips_init_when_already_initialised(tmp_path: Path) -> None:
    from taichi.lang import impl as ti_impl

    mock_runtime = MagicMock()
    mock_runtime.prog = object()

    with patch.object(ti_impl, "get_runtime", return_value=mock_runtime), \
         patch("taichi.init") as mock_init, \
         patch("pygotm.validation.parallel.validate_case", return_value=_fake_result("couette")):
        _run_case_worker("couette", tmp_path, "cpu", skip_run=False)

    mock_init.assert_not_called()


def test_run_cases_parallel_calls_on_result_for_each_case(tmp_path: Path) -> None:
    cases = ["couette", "channel"]
    received: list[str] = []

    mock_cluster = MagicMock()
    mock_cluster.__enter__ = MagicMock(return_value=mock_cluster)
    mock_cluster.__exit__ = MagicMock(return_value=False)
    mock_cluster.dashboard_link = "http://localhost:8787"

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    submitted: list[tuple[MagicMock, str]] = []

    def tracking_submit(fn: object, name: str, *args: object,
                        key: str = "", **kw: object) -> MagicMock:
        f = MagicMock()
        f.result.return_value = _fake_result(name)
        submitted.append((f, name))
        return f

    mock_client.submit.side_effect = tracking_submit

    def as_completed_mock(fmap: dict[MagicMock, str]) -> object:
        for f, _name in submitted:
            yield f

    with patch("pygotm.validation.parallel.LocalCluster", return_value=mock_cluster), \
         patch("pygotm.validation.parallel.Client", return_value=mock_client), \
         patch("pygotm.validation.parallel.as_completed", side_effect=as_completed_mock):
        run_cases_parallel(
            case_names=cases,
            runs_dir=tmp_path,
            arch_name="cpu",
            n_workers=2,
            skip_run=True,
            on_result=lambda r: received.append(r.case_name),
        )

    assert set(received) == {"couette", "channel"}


def test_run_cases_parallel_clamps_workers_to_case_count(tmp_path: Path) -> None:
    cluster_calls: list[dict[str, object]] = []

    class TrackingCluster(MagicMock):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            cluster_calls.append(kwargs)
            self.dashboard_link = "http://localhost:8787"

        def __enter__(self) -> "TrackingCluster":
            return self

        def __exit__(self, *a: object) -> bool:
            return False

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    submitted_futures: list[tuple[MagicMock, str]] = []

    def tracking_submit(fn: object, name: str, *a: object,
                        key: str = "", **kw: object) -> MagicMock:
        f = MagicMock()
        f.result.return_value = _fake_result(name)
        submitted_futures.append((f, name))
        return f

    mock_client.submit.side_effect = tracking_submit

    def as_completed_mock(fmap: object) -> object:
        for f, _n in submitted_futures:
            yield f

    with patch("pygotm.validation.parallel.LocalCluster", TrackingCluster), \
         patch("pygotm.validation.parallel.Client", return_value=mock_client), \
         patch("pygotm.validation.parallel.as_completed", side_effect=as_completed_mock):
        run_cases_parallel(
            case_names=["couette"],
            runs_dir=tmp_path,
            arch_name="cpu",
            n_workers=32,
            skip_run=True,
        )

    assert cluster_calls[0]["n_workers"] == 1
