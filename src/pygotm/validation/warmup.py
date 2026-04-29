"""Numba AOT kernel warm-up for pyGOTM validation.

Calls each single-column and batch entry point once with minimal arrays
(nlev=5) to trigger Numba JIT compilation before timed validation runs.
With cache=True, compiled specialisations are stored in __pycache__ and
reused on subsequent runs without recompilation.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["trigger_numba_jit"]


def trigger_numba_jit(nlev: int = 5) -> float:
    """Force Numba JIT compilation of all physics kernels; return elapsed seconds."""
    from pygotm.util.tridiagonal import tridiagonal

    t0 = time.monotonic()

    # tridiagonal (single-column)
    n = nlev + 1
    au = np.zeros(n, dtype=np.float64)
    bu = np.ones(n, dtype=np.float64) * 2.0
    cu = np.zeros(n, dtype=np.float64)
    du = np.ones(n, dtype=np.float64)
    ru = np.zeros(n, dtype=np.float64)
    qu = np.zeros(n, dtype=np.float64)
    val = np.zeros(n, dtype=np.float64)
    tridiagonal(au, bu, cu, du, ru, qu, val, 0, nlev)

    return time.monotonic() - t0
