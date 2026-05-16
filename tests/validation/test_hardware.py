"""Tests for validation/hardware.py — platform detection with mocked hardware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pygotm.validation.hardware import (
    PlatformInfo,
    _probe_amd,
    _probe_nvidia,
    _probe_vulkan,
    detect_platform,
)


def test_detect_platform_returns_platform_info() -> None:
    info = detect_platform()
    assert isinstance(info, PlatformInfo)


def test_cpu_always_in_available_archs() -> None:
    info = detect_platform()
    assert "cpu" in info.available_archs


def test_cpu_count_is_positive() -> None:
    info = detect_platform()
    assert info.cpu_count >= 1


def test_gpu_count_is_non_negative() -> None:
    info = detect_platform()
    assert info.gpu_count >= 0


def test_hardware_dict_has_required_keys() -> None:
    info = detect_platform()
    for key in (
        "cpu_model",
        "cpu_count",
        "numba_version",
        "python_version",
        "platform",
    ):
        assert key in info.hardware, f"missing key: {key}"


def test_available_archs_are_valid_numba_backends() -> None:
    valid = {"cpu"}
    info = detect_platform()
    assert set(info.available_archs).issubset(valid)


def test_available_archs_has_no_duplicates() -> None:
    info = detect_platform()
    assert len(info.available_archs) == len(set(info.available_archs))


def test_probe_nvidia_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Tesla V100\nTesla A100\n"
    with patch("pygotm.validation.hardware.subprocess.run", return_value=mock_result):
        found, count, info = _probe_nvidia()
    assert found is True
    assert count == 2
    assert "Tesla V100" in info
    assert "Tesla A100" in info


def test_probe_nvidia_not_found() -> None:
    with patch(
        "pygotm.validation.hardware.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        found, count, info = _probe_nvidia()
    assert found is False
    assert count == 0
    assert info == ""


def test_probe_nvidia_nonzero_returncode() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    with patch("pygotm.validation.hardware.subprocess.run", return_value=mock_result):
        found, count, info = _probe_nvidia()
    assert found is False


def test_detect_platform_with_nvidia_gpu() -> None:
    with (
        patch(
            "pygotm.validation.hardware._probe_nvidia",
            return_value=(True, 1, "RTX 4090"),
        ),
        patch("pygotm.validation.hardware._probe_amd", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_vulkan", return_value=False),
    ):
        info = detect_platform()
    assert info.available_archs == ["cpu"]
    assert info.gpu_count == 1
    assert "gpu_info" in info.hardware
    assert "RTX 4090" in info.hardware["gpu_info"]


def test_probe_amd_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "GPU[0]: Instinct MI250X\n"
    with patch("pygotm.validation.hardware.subprocess.run", return_value=mock_result):
        found, count, info = _probe_amd()
    assert found is True
    assert count >= 1


def test_probe_amd_not_found() -> None:
    with patch(
        "pygotm.validation.hardware.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        found, count, info = _probe_amd()
    assert found is False


def test_detect_platform_with_amd_gpu() -> None:
    with (
        patch("pygotm.validation.hardware._probe_nvidia", return_value=(False, 0, "")),
        patch(
            "pygotm.validation.hardware._probe_amd", return_value=(True, 1, "MI250X")
        ),
        patch("pygotm.validation.hardware._probe_vulkan", return_value=False),
    ):
        info = detect_platform()
    assert info.available_archs == ["cpu"]
    assert info.gpu_count == 1


def test_probe_vulkan_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("pygotm.validation.hardware.subprocess.run", return_value=mock_result):
        assert _probe_vulkan() is True


def test_probe_vulkan_not_found() -> None:
    with patch(
        "pygotm.validation.hardware.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        assert _probe_vulkan() is False


def test_detect_platform_vulkan_only_when_no_cuda_amd() -> None:
    with (
        patch("pygotm.validation.hardware._probe_nvidia", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_amd", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_vulkan", return_value=True),
    ):
        info = detect_platform()
    assert info.available_archs == ["cpu"]
    assert info.gpu_count == 1


def test_detect_platform_no_vulkan_when_cuda_present() -> None:
    with (
        patch(
            "pygotm.validation.hardware._probe_nvidia",
            return_value=(True, 1, "RTX 4090"),
        ),
        patch("pygotm.validation.hardware._probe_amd", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_vulkan", return_value=True),
    ):
        info = detect_platform()
    assert "vulkan" not in info.available_archs
    assert info.available_archs == ["cpu"]


def test_detect_platform_metal_on_darwin() -> None:
    with (
        patch("pygotm.validation.hardware.sys") as mock_sys,
        patch("pygotm.validation.hardware._probe_nvidia", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_amd", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_vulkan", return_value=False),
    ):
        mock_sys.platform = "darwin"
        info = detect_platform()
    assert info.available_archs == ["cpu"]
    assert info.gpu_count == 1


def test_detect_platform_no_metal_on_linux() -> None:
    with (
        patch("pygotm.validation.hardware.sys") as mock_sys,
        patch("pygotm.validation.hardware._probe_nvidia", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_amd", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_vulkan", return_value=False),
    ):
        mock_sys.platform = "linux"
        info = detect_platform()
    assert "metal" not in info.available_archs


def test_detect_platform_no_gpu() -> None:
    with (
        patch("pygotm.validation.hardware._probe_nvidia", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_amd", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_vulkan", return_value=False),
        patch("pygotm.validation.hardware.sys") as mock_sys,
    ):
        mock_sys.platform = "linux"
        info = detect_platform()
    assert info.available_archs == ["cpu"]
    assert info.gpu_count == 0
    assert "gpu_info" not in info.hardware


def test_detect_platform_multiple_nvidia_gpus() -> None:
    with (
        patch(
            "pygotm.validation.hardware._probe_nvidia",
            return_value=(True, 4, "A100; A100; A100; A100"),
        ),
        patch("pygotm.validation.hardware._probe_amd", return_value=(False, 0, "")),
        patch("pygotm.validation.hardware._probe_vulkan", return_value=False),
    ):
        info = detect_platform()
    assert info.gpu_count == 4
