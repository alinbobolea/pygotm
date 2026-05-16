from __future__ import annotations

import numpy as np

from pygotm.icethm.models.lebedev import step_lebedev


def test_lebedev_grows_from_freezing_degree_days() -> None:
    fdd = np.array([100.0], dtype=np.float64)
    hice = np.zeros(1, dtype=np.float64)
    cover = np.zeros(1, dtype=np.int32)
    albedo = np.zeros(1, dtype=np.float64)
    trans = np.zeros(1, dtype=np.float64)
    tf = np.zeros(1, dtype=np.float64)

    step_lebedev(-10.0, -2.0, 0.0, 0.0, fdd, hice, cover, albedo, trans, tf)

    assert hice[0] > 0.0
    assert cover[0] == 2
    assert 0.0 < trans[0] < 1.0
    assert albedo[0] == 0.545


def test_lebedev_resets_when_degree_days_melt_out() -> None:
    fdd = np.array([0.5], dtype=np.float64)
    hice = np.array([0.2], dtype=np.float64)
    cover = np.array([2], dtype=np.int32)
    albedo = np.zeros(1, dtype=np.float64)
    trans = np.zeros(1, dtype=np.float64)
    tf = np.zeros(1, dtype=np.float64)

    step_lebedev(5.0, 1.0, 0.0, 3600.0, fdd, hice, cover, albedo, trans, tf)

    assert hice[0] == 0.0
    assert cover[0] == 0
    assert trans[0] == 1.0
