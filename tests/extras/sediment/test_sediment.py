"""Tests for sediment transport helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.extras.sediment.sediment import (
    NoFlux,
    SedimentState,
    SmithMcLean,
    do_sediment,
    end_sediment,
    init_sediment,
    save_sediment,
    sediment_eulerian,
    sediment_lagrangian,
    settling_velocity_zanke,
)


def test_settling_velocity_uses_zanke_formula_and_is_downward() -> None:
    wc, gs = settling_velocity_zanke(62.5e-6, 9.81, 2650.0, 1027.0)
    expected_gs = 9.81 * (2650.0 - 1027.0) / 1027.0
    expected_wc = (
        -10.0
        * 1.3e-6
        / 62.5e-6
        * (np.sqrt(1.0 + (0.01 * expected_gs * 62.5e-6**3) / 1.3e-6 / 1.3e-6) - 1.0)
    )
    assert gs == pytest.approx(expected_gs)
    assert wc == pytest.approx(expected_wc)
    assert wc < 0.0


def test_init_sediment_allocates_eulerian_arrays() -> None:
    state = SedimentState(sedi_calc=True, init_conc=2.0e-4)
    init_sediment(state, 4, 9.81, 1027.0)

    assert state.C is not None
    assert state.wc is not None
    assert state.C.shape == (5,)
    assert np.all(state.C[1:] == pytest.approx(2.0e-4))
    assert np.all(state.wc[1:] < 0.0)


def test_smith_mclean_boundary_sets_dirichlet_bottom_concentration() -> None:
    state = SedimentState(sedi_calc=True, sedi_method=SmithMcLean)
    nlev = 4
    h = np.ones(nlev + 1)
    init_sediment(state, nlev, 9.81, 1027.0)

    sediment_eulerian(
        state,
        nlev,
        1.0,
        h,
        np.full(nlev + 1, 1.0e-4),
        u_taub=max(state.ustarc * 1.5, 0.01),
        z0b=0.01,
    )

    assert state.DiffBcdw == 0
    assert state.DiffCdw >= 0.0
    assert state.C is not None
    assert np.isfinite(state.C).all()


def test_do_sediment_noflux_preserves_finite_nonnegative_concentration() -> None:
    state = SedimentState(sedi_calc=True, sedi_method=NoFlux, adv_method=1)
    nlev = 5
    h = np.ones(nlev + 1)
    init_sediment(state, nlev, 9.81, 1027.0)

    do_sediment(state, nlev, 0.1, h, np.full(nlev + 1, 1.0e-4))

    assert state.C is not None
    assert np.all(state.C[1:] >= 0.0)
    assert np.isfinite(state.C).all()


def test_lagrangian_sediment_updates_particle_positions_reproducibly() -> None:
    nlev = 4
    h = np.ones(nlev + 1)
    state = SedimentState(sedi_calc=True, sedi_eulerian=False, sedi_npar=5)
    init_sediment(state, nlev, 9.81, 1027.0, depth=4.0, h=h)
    assert state.zp is not None
    before = state.zp.copy()
    zlev = np.linspace(-4.0, 0.0, nlev + 1)

    sediment_lagrangian(
        state,
        nlev,
        0.1,
        zlev,
        np.full(nlev + 1, 1.0e-4),
        h,
        rng=np.random.default_rng(123),
    )

    assert state.zp is not None
    assert not np.array_equal(state.zp, before)
    assert np.all(state.zp >= -4.0)
    assert np.all(state.zp <= 0.0)


def test_save_and_end_sediment() -> None:
    state = SedimentState(sedi_calc=True)
    init_sediment(state, 3, 9.81, 1027.0)

    values = save_sediment(state)
    end_sediment(state)

    assert values.shape == (4,)
    assert state.C is None
