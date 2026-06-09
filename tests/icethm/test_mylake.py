from __future__ import annotations

import numpy as np

from pygotm.icethm.models.mylake import step_mylake


def test_mylake_frazil_consolidates_to_ice() -> None:
    hice = np.zeros(1, dtype=np.float64)
    hfrazil = np.array([0.029], dtype=np.float64)
    dhis = np.zeros(1, dtype=np.float64)
    dhib = np.zeros(1, dtype=np.float64)
    ts = np.zeros(1, dtype=np.float64)
    cover = np.zeros(1, dtype=np.int32)
    albedo = np.zeros(1, dtype=np.float64)
    attenuation = np.array([5.0], dtype=np.float64)
    trans = np.zeros(1, dtype=np.float64)
    tf = np.zeros(1, dtype=np.float64)
    ocean_flux = np.zeros(1, dtype=np.float64)
    qh = np.zeros(1, dtype=np.float64)
    qs = np.zeros(1, dtype=np.float64)
    bottom_energy = np.zeros(1, dtype=np.float64)

    water_t = step_mylake(
        -1.0,
        0.0,
        -5.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        60.0,
        0.0,
        hice,
        hfrazil,
        dhis,
        dhib,
        ts,
        cover,
        albedo,
        attenuation,
        trans,
        tf,
        ocean_flux,
        qh,
        qs,
        bottom_energy,
    )

    assert hice[0] > 0.0
    assert hfrazil[0] == 0.0
    assert cover[0] == 2
    assert trans[0] == 0.0
    assert water_t == -1.0
    assert dhib[0] > 0.0


def test_mylake_melts_nonnegative() -> None:
    hice = np.array([0.01], dtype=np.float64)
    hfrazil = np.zeros(1, dtype=np.float64)
    dhis = np.zeros(1, dtype=np.float64)
    dhib = np.zeros(1, dtype=np.float64)
    ts = np.zeros(1, dtype=np.float64)
    cover = np.array([2], dtype=np.int32)
    albedo = np.zeros(1, dtype=np.float64)
    attenuation = np.array([5.0], dtype=np.float64)
    trans = np.zeros(1, dtype=np.float64)
    tf = np.zeros(1, dtype=np.float64)
    ocean_flux = np.zeros(1, dtype=np.float64)
    qh = np.zeros(1, dtype=np.float64)
    qs = np.zeros(1, dtype=np.float64)
    bottom_energy = np.zeros(1, dtype=np.float64)

    water_t = step_mylake(
        5.0,
        0.0,
        5.0,
        1.0,
        1000.0,
        1000.0,
        0.0,
        0.0,
        3600.0,
        0.0,
        hice,
        hfrazil,
        dhis,
        dhib,
        ts,
        cover,
        albedo,
        attenuation,
        trans,
        tf,
        ocean_flux,
        qh,
        qs,
        bottom_energy,
    )

    assert hice[0] >= 0.0
    assert trans[0] == 1.0 if hice[0] == 0.0 else 0.0 <= trans[0] <= 1.0
    assert cover[0] == 0
    assert attenuation[0] == 0.0
    assert water_t > 0.0
    assert dhib[0] < 0.0


def test_mylake_keeps_zero_attenuation_after_refreeze() -> None:
    hice = np.array([0.1], dtype=np.float64)
    hfrazil = np.zeros(1, dtype=np.float64)
    dhis = np.zeros(1, dtype=np.float64)
    dhib = np.zeros(1, dtype=np.float64)
    ts = np.zeros(1, dtype=np.float64)
    cover = np.array([2], dtype=np.int32)
    albedo = np.zeros(1, dtype=np.float64)
    attenuation = np.zeros(1, dtype=np.float64)
    trans = np.zeros(1, dtype=np.float64)
    tf = np.zeros(1, dtype=np.float64)
    ocean_flux = np.zeros(1, dtype=np.float64)
    qh = np.zeros(1, dtype=np.float64)
    qs = np.zeros(1, dtype=np.float64)
    bottom_energy = np.zeros(1, dtype=np.float64)

    step_mylake(
        -0.1,
        0.0,
        -5.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3600.0,
        0.0,
        hice,
        hfrazil,
        dhis,
        dhib,
        ts,
        cover,
        albedo,
        attenuation,
        trans,
        tf,
        ocean_flux,
        qh,
        qs,
        bottom_energy,
    )

    assert cover[0] == 2
    assert attenuation[0] == 0.0
    assert trans[0] == 1.0
