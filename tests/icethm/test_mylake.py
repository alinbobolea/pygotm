from __future__ import annotations

import numpy as np

from pygotm.icethm.models.mylake import step_mylake


def test_mylake_frazil_consolidates_to_ice() -> None:
    hice = np.zeros(1, dtype=np.float64)
    hfrazil = np.array([0.029], dtype=np.float64)
    ts = np.zeros(1, dtype=np.float64)
    cover = np.zeros(1, dtype=np.int32)
    albedo = np.zeros(1, dtype=np.float64)
    trans = np.zeros(1, dtype=np.float64)
    tf = np.zeros(1, dtype=np.float64)
    qh = np.zeros(1, dtype=np.float64)
    qs = np.zeros(1, dtype=np.float64)

    step_mylake(
        -1.0,
        0.0,
        -5.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        60.0,
        hice,
        hfrazil,
        ts,
        cover,
        albedo,
        trans,
        tf,
        qh,
        qs,
    )

    assert hice[0] > 0.0
    assert hfrazil[0] == 0.0
    assert cover[0] == 2
    assert 0.0 <= trans[0] <= 1.0


def test_mylake_melts_nonnegative() -> None:
    hice = np.array([0.01], dtype=np.float64)
    hfrazil = np.zeros(1, dtype=np.float64)
    ts = np.zeros(1, dtype=np.float64)
    cover = np.array([2], dtype=np.int32)
    albedo = np.zeros(1, dtype=np.float64)
    trans = np.zeros(1, dtype=np.float64)
    tf = np.zeros(1, dtype=np.float64)
    qh = np.zeros(1, dtype=np.float64)
    qs = np.zeros(1, dtype=np.float64)

    step_mylake(
        5.0,
        0.0,
        5.0,
        1.0,
        1000.0,
        1000.0,
        0.0,
        0.0,
        3600.0,
        hice,
        hfrazil,
        ts,
        cover,
        albedo,
        trans,
        tf,
        qh,
        qs,
    )

    assert hice[0] >= 0.0
    assert trans[0] == 1.0 if hice[0] == 0.0 else 0.0 <= trans[0] <= 1.0
