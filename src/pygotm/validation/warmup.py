"""Taichi kernel warm-up for pyGOTM validation.

Runs the couette case (fastest, ~1-2 s) to trigger Taichi JIT compilation
and populate the offline kernel cache (~/.cache/taichi/). All subsequent
runs — including Dask worker subprocesses — load from cache, skipping
recompilation.  Must be called after ti.init() with the target arch.
"""

from __future__ import annotations

import time
from pathlib import Path

__all__ = ["warm_taichi_kernels"]

_WARMUP_CASE = "couette"


def warm_taichi_kernels(arch_name: str, runs_dir: Path) -> float:
    """Compile Taichi kernels by running the couette case; return elapsed seconds."""
    from pygotm.driver import GotmDriver
    from pygotm.validate import resolve_reference_case

    case = resolve_reference_case(_WARMUP_CASE)
    warmup_dir = runs_dir / "_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    warmup_nc = warmup_dir / "warmup.nc"

    t0 = time.monotonic()
    GotmDriver(case.yaml_path).run(output_path=warmup_nc)
    return time.monotonic() - t0
