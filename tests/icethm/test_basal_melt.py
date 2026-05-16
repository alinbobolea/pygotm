from __future__ import annotations

import numpy as np
import pytest

from pygotm.icethm.models.basal_melt import (
    basal_freezing_temperature,
    step_basal_melt,
)


def _state() -> tuple[np.ndarray, ...]:
    return tuple(np.zeros(1, dtype=np.float64) for _ in range(6))


def test_basal_melt_zero_near_interface_freezing_point() -> None:
    melt, tm, sm, qh, qs, tf = _state()
    S = 34.5
    H = 338.0
    T = basal_freezing_temperature(S, H)

    step_basal_melt(T, S, H, 0.01, melt, tm, sm, qh, qs, tf)

    assert abs(melt[0]) < 1.0e-8
    assert sm[0] == pytest.approx(S, rel=1.0e-5)
    assert tf[0] == pytest.approx(T)


def test_basal_melt_positive_when_water_is_warm() -> None:
    melt, tm, sm, qh, qs, tf = _state()
    S = 34.5
    H = 338.0
    T = basal_freezing_temperature(S, H) + 0.25

    step_basal_melt(T, S, H, 0.01, melt, tm, sm, qh, qs, tf)

    assert melt[0] > 0.0
    assert qh[0] > 0.0
    assert 0.0 < sm[0] < S
    assert qs[0] < 0.0
