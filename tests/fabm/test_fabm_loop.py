"""Tests for FABM loop compiled array helpers."""

from __future__ import annotations

import numpy as np

from pygotm.fabm.fabm_loop import _apply_sinking
from pygotm.fabm.gotm_fabm import (
    step_fabm_post_rates_single,
    step_fabm_transport_single,
)
from pygotm.util.diff_center import NEUMANN, diff_center


def _python_transport_reference(
    cc: np.ndarray,
    vert_move: np.ndarray,
    h_step: np.ndarray,
    nuh_step: np.ndarray,
    *,
    nlev: int,
    dt: float,
    cnpar: float,
    precip: float,
    n_interior: int,
) -> None:
    y = np.zeros(nlev + 1, dtype=np.float64)
    ws = np.zeros(nlev + 1, dtype=np.float64)
    adv_cu = np.zeros(nlev + 1, dtype=np.float64)
    au = np.zeros(nlev + 1, dtype=np.float64)
    bu = np.zeros(nlev + 1, dtype=np.float64)
    cu = np.zeros(nlev + 1, dtype=np.float64)
    du = np.zeros(nlev + 1, dtype=np.float64)
    ru = np.zeros(nlev + 1, dtype=np.float64)
    qu = np.zeros(nlev + 1, dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e15, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)

    _apply_sinking(
        vert_move,
        h_step,
        cc,
        nlev,
        dt,
        n_interior,
        y,
        ws,
        adv_cu,
    )
    for var in range(n_interior):
        y[1 : nlev + 1] = cc[var, :]
        diff_center(
            nlev,
            dt,
            cnpar,
            0,
            h_step,
            NEUMANN,
            NEUMANN,
            -float(cc[var, -1]) * precip,
            0.0,
            nuh_step,
            l_sour,
            q_sour,
            tau_r,
            y_obs,
            y,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )
        cc[var, :] = y[1 : nlev + 1]


def test_compiled_fabm_transport_matches_python_path_for_100_steps() -> None:
    nlev = 4
    n_interior = 2
    dt = 30.0
    cnpar = 0.6
    precip = 1.0e-7
    h_step = np.array([0.0, 1.0, 1.2, 1.4, 1.6], dtype=np.float64)
    nuh_step = np.array([0.0, 2.0e-4, 2.5e-4, 3.0e-4, 3.5e-4], dtype=np.float64)
    vert_move = np.array(
        [[0.0, -1.0e-6, -2.0e-6, -3.0e-6], [2.0e-6, 1.0e-6, 0.0, -1.0e-6]],
        dtype=np.float64,
    )
    initial = np.array(
        [[0.4, 0.5, 0.6, 0.7], [0.8, 0.7, 0.6, 0.5]],
        dtype=np.float64,
    )
    expected = initial.copy()
    actual = initial.copy()

    y = np.zeros(nlev + 1, dtype=np.float64)
    ws = np.zeros(nlev + 1, dtype=np.float64)
    adv_cu = np.zeros(nlev + 1, dtype=np.float64)
    au = np.zeros(nlev + 1, dtype=np.float64)
    bu = np.zeros(nlev + 1, dtype=np.float64)
    cu = np.zeros(nlev + 1, dtype=np.float64)
    du = np.zeros(nlev + 1, dtype=np.float64)
    ru = np.zeros(nlev + 1, dtype=np.float64)
    qu = np.zeros(nlev + 1, dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e15, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)

    for _ in range(100):
        _python_transport_reference(
            expected,
            vert_move,
            h_step,
            nuh_step,
            nlev=nlev,
            dt=dt,
            cnpar=cnpar,
            precip=precip,
            n_interior=n_interior,
        )
        step_fabm_transport_single(
            nlev,
            dt,
            cnpar,
            precip,
            1,
            n_interior,
            vert_move,
            h_step,
            nuh_step,
            actual,
            y,
            ws,
            adv_cu,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
            l_sour,
            q_sour,
            tau_r,
            y_obs,
        )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert step_fabm_transport_single.nopython_signatures


def test_compiled_fabm_post_rates_matches_python_update_for_boundaries() -> None:
    nlev = 4
    n_interior = 2
    n_surface = 1
    n_bottom = 1
    dt = 20.0
    initial = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
            [4.0, 4.1, 4.2, 4.3],
        ],
        dtype=np.float64,
    )
    bulk = np.full_like(initial, 1.0e-3)
    surf = np.full_like(initial, 2.0e-3)
    bottom = np.full_like(initial, -5.0e-4)
    expected = initial.copy()
    actual = initial.copy()

    for var in range(n_interior):
        expected[var] += dt * bulk[var]
        expected[var, -1] += dt * (surf[var, -1] - bulk[var, -1])
        expected[var, 0] += dt * (bottom[var, 0] - bulk[var, 0])
    for var in range(n_interior, n_interior + n_surface):
        expected[var, :] = expected[var, -1] + dt * surf[var, -1]
    for var in range(n_interior + n_surface, initial.shape[0]):
        expected[var, :] = expected[var, 0] + dt * bottom[var, 0]

    step_fabm_post_rates_single(
        nlev,
        dt,
        n_interior,
        n_surface,
        n_bottom,
        bulk,
        surf,
        bottom,
        actual,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert step_fabm_post_rates_single.nopython_signatures
