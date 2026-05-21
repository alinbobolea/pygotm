from __future__ import annotations

import numpy as np

from pygotm.icethm.models.simple import step_simple


def test_simple_zeroes_warming_at_freezing() -> None:
    tf = np.zeros(1, dtype=np.float64)
    hice = np.zeros(1, dtype=np.float64)
    cover = np.zeros(1, dtype=np.int32)

    flux = step_simple(-2.1, 35.0, 1.0e-5, tf, hice, cover)

    assert flux == 0.0
    # step_simple reads the cached Tf value but never updates it;
    # Tf is initialised once at post_init_ice, not recomputed each step.
    assert tf[0] == 0.0
    assert hice[0] == 0.0
    assert cover[0] == 0


def test_simple_allows_cooling_and_warm_water() -> None:
    tf = np.zeros(1, dtype=np.float64)
    hice = np.zeros(1, dtype=np.float64)
    cover = np.zeros(1, dtype=np.int32)

    assert step_simple(-2.1, 35.0, -1.0e-5, tf, hice, cover) == -1.0e-5
    assert step_simple(5.0, 35.0, 1.0e-5, tf, hice, cover) == 1.0e-5
