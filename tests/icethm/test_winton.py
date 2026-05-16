from __future__ import annotations

import numpy as np

from pygotm.icethm.models.winton import ice_optics, step_winton


def test_winton_optics_open_water_and_ice() -> None:
    albedo = np.zeros(1, dtype=np.float64)
    trans = np.zeros(1, dtype=np.float64)

    ice_optics(0.0, 0.0, -2.0, albedo, trans)
    assert albedo[0] == 0.06
    assert trans[0] == 1.0

    ice_optics(1.0, 0.0, -2.0, albedo, trans)
    assert 0.0 < albedo[0] < 1.0
    assert 0.0 < trans[0] < 1.0


def test_winton_step_keeps_state_bounded() -> None:
    hice = np.array([1.0], dtype=np.float64)
    hsnow = np.zeros(1, dtype=np.float64)
    t1 = np.array([-2.0], dtype=np.float64)
    t2 = np.array([-2.0], dtype=np.float64)
    ts = np.array([-5.0], dtype=np.float64)
    cover = np.array([2], dtype=np.int32)
    albedo = np.zeros(1, dtype=np.float64)
    trans = np.zeros(1, dtype=np.float64)
    tf = np.zeros(1, dtype=np.float64)
    qh = np.zeros(1, dtype=np.float64)
    ocean_flux = np.array([10.0], dtype=np.float64)
    qs = np.zeros(1, dtype=np.float64)
    top = np.zeros(1, dtype=np.float64)
    bottom = np.zeros(1, dtype=np.float64)

    step_winton(
        -1.8,
        33.0,
        1.0,
        3600.0,
        20.0,
        -40.0,
        -10.0,
        -5.0,
        0.0,
        hice,
        hsnow,
        t1,
        t2,
        ts,
        cover,
        albedo,
        trans,
        tf,
        qh,
        ocean_flux,
        qs,
        top,
        bottom,
    )

    assert hice[0] >= 0.0
    assert hsnow[0] >= 0.0
    assert cover[0] in (0, 2)
    assert 0.0 <= albedo[0] <= 1.0
    assert 0.0 <= trans[0] <= 1.0
    if hice[0] > 0.0:
        assert t1[0] <= 0.0
        assert t2[0] <= 0.0
