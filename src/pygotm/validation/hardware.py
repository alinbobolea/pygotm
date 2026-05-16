"""Platform capability detection for pyGOTM validation."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass

__all__ = ["PlatformInfo", "detect_platform"]


@dataclass
class PlatformInfo:
    cpu_count: int
    gpu_count: int
    available_archs: list[str]
    hardware: dict[str, str]


def _lscpu_info() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key == "Model name":
                info["cpu_model"] = val
            elif key == "CPU(s)":
                info["cpu_count"] = val
            elif key == "Core(s) per socket":
                info["cores_per_socket"] = val
            elif key == "Socket(s)":
                info["sockets"] = val
            elif key == "CPU max MHz":
                try:
                    info["cpu_max_mhz"] = f"{float(val):.0f}"
                except ValueError:
                    info["cpu_max_mhz"] = val
    except Exception:
        info.setdefault("cpu_model", platform.processor() or "unknown")
    return info


def _ram_total() -> str:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(re.search(r"(\d+)", line).group(1))  # type: ignore[union-attr]
                    return f"{kb / 1024 / 1024:.1f} GiB"
    except Exception:
        pass
    return "unknown"


def _numba_version() -> str:
    try:
        return importlib.metadata.version("numba")
    except Exception:
        return "unavailable"


def _probe_nvidia() -> tuple[bool, int, str]:
    """Return (available, gpu_count, info_string) for NVIDIA CUDA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            names = [
                line.strip()
                for line in result.stdout.strip().splitlines()
                if line.strip()
            ]
            if names:
                return True, len(names), "; ".join(names)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return False, 0, ""


def _probe_amd() -> tuple[bool, int, str]:
    """Return (available, gpu_count, info_string) for AMD ROCm GPUs."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [
                line.strip()
                for line in result.stdout.strip().splitlines()
                if line.strip()
            ]
            gpu_lines = [
                line
                for line in lines
                if any(k in line for k in ("GPU", "Radeon", "Instinct"))
            ]
            count = max(1, len(gpu_lines))
            return True, count, lines[0]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return False, 0, ""


def _probe_vulkan() -> bool:
    """Return True if Vulkan is available."""
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def detect_platform() -> PlatformInfo:
    """Probe hardware and return a :class:`PlatformInfo` describing availability."""
    cpu_count = os.cpu_count() or 1
    lscpu = _lscpu_info()
    if "cpu_count" in lscpu:
        try:
            cpu_count = int(lscpu["cpu_count"])
        except ValueError:
            pass

    hardware: dict[str, str] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_arch": platform.machine(),
        "cpu_model": lscpu.get("cpu_model", platform.processor() or "unknown"),
        "cpu_count": str(cpu_count),
        "numba_version": _numba_version(),
    }
    for k in ("cores_per_socket", "sockets", "cpu_max_mhz"):
        if k in lscpu:
            hardware[k] = lscpu[k]
    hardware["ram_total"] = _ram_total()

    available_archs: list[str] = ["cpu"]
    gpu_count = 0
    gpu_info_parts: list[str] = []

    has_cuda, n_cuda, cuda_info = _probe_nvidia()
    if has_cuda:
        gpu_count += n_cuda
        gpu_info_parts.append(f"CUDA: {cuda_info}")

    has_amd, n_amd, amd_info = _probe_amd()
    if has_amd:
        gpu_count += n_amd
        if amd_info:
            gpu_info_parts.append(f"ROCm: {amd_info}")

    if not has_cuda and not has_amd and _probe_vulkan():
        gpu_count = max(gpu_count, 1)
        gpu_info_parts.append("Vulkan")

    if sys.platform == "darwin":
        gpu_count = max(gpu_count, 1)
        gpu_info_parts.append("Metal")

    if gpu_info_parts:
        hardware["gpu_info"] = "; ".join(gpu_info_parts)

    return PlatformInfo(
        cpu_count=cpu_count,
        gpu_count=gpu_count,
        available_archs=available_archs,
        hardware=hardware,
    )
