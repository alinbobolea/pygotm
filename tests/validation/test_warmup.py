"""Tests for validation/warmup.py — Taichi kernel warm-up."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pygotm.validation.warmup import warm_taichi_kernels


def test_warm_taichi_kernels_returns_positive_float(tmp_path: Path) -> None:
    mock_driver = MagicMock()
    mock_case = MagicMock()
    mock_case.yaml_path = tmp_path / "gotm.yaml"
    with patch("pygotm.validate.resolve_reference_case", return_value=mock_case), \
         patch("pygotm.driver.GotmDriver", return_value=mock_driver):
        elapsed = warm_taichi_kernels("cpu", tmp_path)
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_warm_taichi_kernels_creates_warmup_directory(tmp_path: Path) -> None:
    mock_driver = MagicMock()
    mock_case = MagicMock()
    mock_case.yaml_path = tmp_path / "gotm.yaml"
    with patch("pygotm.validate.resolve_reference_case", return_value=mock_case), \
         patch("pygotm.driver.GotmDriver", return_value=mock_driver):
        warm_taichi_kernels("cpu", tmp_path)
    assert (tmp_path / "_warmup").is_dir()


def test_warm_taichi_kernels_calls_driver_run(tmp_path: Path) -> None:
    mock_instance = MagicMock()
    mock_case = MagicMock()
    mock_case.yaml_path = tmp_path / "gotm.yaml"
    with patch("pygotm.validate.resolve_reference_case", return_value=mock_case), \
         patch("pygotm.driver.GotmDriver", return_value=mock_instance):
        warm_taichi_kernels("cpu", tmp_path)
    mock_instance.run.assert_called_once()
    call_kwargs = mock_instance.run.call_args.kwargs
    assert "output_path" in call_kwargs
    assert "_warmup" in str(call_kwargs["output_path"])


def test_warm_taichi_kernels_uses_couette_case(tmp_path: Path) -> None:
    mock_driver = MagicMock()
    mock_case = MagicMock()
    mock_case.yaml_path = tmp_path / "gotm.yaml"
    with patch("pygotm.validate.resolve_reference_case", return_value=mock_case) as mock_resolve, \
         patch("pygotm.driver.GotmDriver", return_value=mock_driver):
        warm_taichi_kernels("cpu", tmp_path)
    mock_resolve.assert_called_once_with("couette")
