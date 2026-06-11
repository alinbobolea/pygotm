"""Tests for FABM loop compiled array helpers."""

from __future__ import annotations

import numpy as np

from pygotm.fabm.fabm_loop import (
    _accumulate_diagnostics,
    _apply_sinking,
    _reduce_accumulated_diagnostics,
    _repair_state,
)
from pygotm.fabm.gotm_fabm import (
    step_fabm_lake_advection_single,
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


def test_compiled_fabm_lake_advection_applies_stream_concentrations() -> None:
    nlev = 2
    dt = 0.1
    cc = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float64)
    stream_flow = np.array([2.0], dtype=np.float64)
    stream_Q = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
    stream_concentrations = np.array([[3.0], [0.0]], dtype=np.float64)
    stream_has_concentration = np.array([[1], [0]], dtype=np.int64)
    no_river_dilution = np.array([1, 1], dtype=np.int64)
    h_step = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    vco_step = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    vc_step = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    afo_step = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    wq_step = np.zeros(nlev + 1, dtype=np.float64)
    qres_step = np.zeros(nlev + 1, dtype=np.float64)
    y = np.zeros(nlev + 1, dtype=np.float64)
    adv_cu = np.zeros(nlev + 1, dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)

    step_fabm_lake_advection_single(
        nlev,
        dt,
        6,
        2,
        stream_flow,
        stream_Q,
        stream_concentrations,
        stream_has_concentration,
        no_river_dilution,
        h_step,
        vco_step,
        vc_step,
        afo_step,
        wq_step,
        qres_step,
        cc,
        y,
        adv_cu,
        l_sour,
        q_sour,
    )

    np.testing.assert_allclose(cc, [[1.0, 2.6], [4.0, 6.0]])
    assert step_fabm_lake_advection_single.nopython_signatures


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


def test_window_mean_reduction_recovers_daily_mean_not_midnight_snapshot() -> None:
    """``time_method: mean`` must average FABM output over the output window.

    GOTM-lake writes daily *means* (lake_erken ``gotm.yaml`` sets
    ``time_method: mean``), but the point-mode FABM loop records the
    instantaneous value at the output step. The run starts at 00:00 with hourly
    steps and daily output, so the snapshot lands at **local midnight**. For
    diurnally varying biogeochemistry — gross primary production is zero at
    night, dissolved oxygen sits at its daily minimum — the midnight snapshot is
    systematically biased relative to the reference daily mean. This exercises
    the accumulate/reduce arithmetic the fix uses and confirms it recovers the
    window mean (matching ``np.mean`` over the window), which differs sharply
    from the midnight endpoint.
    """
    output_every = 24
    hours = np.arange(1, output_every + 1, dtype=np.float64)
    # GPP-like diurnal signal: zero at night, positive during daylight hours.
    diurnal = np.maximum(0.0, np.sin(2.0 * np.pi * hours / 24.0))

    acc: dict[str, np.ndarray | float] = {}
    profiles = []
    for amp in diurnal:
        profile = amp * np.array([1.0, 2.0, 3.0], dtype=np.float64)
        profiles.append(profile.copy())
        # Engine returns a fresh array each sub-step (copy semantics).
        _accumulate_diagnostics(
            acc, {"gpp": profile.copy(), "surface_flux": float(amp)}
        )

    reduced = _reduce_accumulated_diagnostics(acc, 1.0 / output_every)
    expected_mean = np.mean(np.stack(profiles), axis=0)

    np.testing.assert_allclose(reduced["gpp"], expected_mean, rtol=0.0, atol=1e-15)
    assert abs(float(reduced["surface_flux"]) - float(np.mean(diurnal))) < 1e-15

    # The midnight snapshot (final sub-step at 24:00) is ~0 and far from the
    # daily mean: this is exactly the systematic error the fix removes.
    midnight_snapshot = profiles[-1]
    assert np.all(midnight_snapshot < 1e-9)
    assert np.all(reduced["gpp"] > 0.3)


def test_window_sum_reduction_matches_integrated_time_method() -> None:
    """``output_reduce_mode == 2`` (``integrated``) sums sub-steps (inv = 1)."""
    acc: dict[str, np.ndarray | float] = {}
    samples = [
        np.array([0.5, 1.0], dtype=np.float64),
        np.array([1.5, 2.0], dtype=np.float64),
        np.array([2.0, 3.0], dtype=np.float64),
    ]
    for sample in samples:
        _accumulate_diagnostics(acc, {"flux": sample.copy()})

    summed = _reduce_accumulated_diagnostics(acc, 1.0)
    np.testing.assert_allclose(summed["flux"], np.sum(np.stack(samples), axis=0))


def test_accumulate_diagnostics_does_not_alias_caller_arrays() -> None:
    """First sub-step is copied, so the caller's buffer is never mutated."""
    acc: dict[str, np.ndarray | float] = {}
    first = np.array([1.0, 2.0], dtype=np.float64)
    _accumulate_diagnostics(acc, {"x": first})
    _accumulate_diagnostics(acc, {"x": np.array([3.0, 4.0], dtype=np.float64)})

    np.testing.assert_allclose(acc["x"], [4.0, 6.0])
    np.testing.assert_allclose(first, [1.0, 2.0])  # caller array untouched

    _accumulate_diagnostics(acc, {"s": 1.5})
    _accumulate_diagnostics(acc, {"s": 2.5})
    assert acc["s"] == 4.0


class _FakeRepairModel:
    """Minimal pyfabm-like model whose ``check_state`` clips to a minimum.

    Mirrors pyfabm's ``check_state(repair=True)``: it clamps the model's
    internal ``state`` array in place and reports validity.
    """

    def __init__(self, shape: tuple[int, ...], minimum: float = 0.0) -> None:
        self.state = np.zeros(shape, dtype=np.float64)
        self._minimum = minimum

    def check_state(self, repair: bool = False) -> bool:
        valid = bool(np.all(self.state >= self._minimum))
        if repair:
            np.maximum(self.state, self._minimum, out=self.state)
        return valid


def test_repair_state_clips_negative_state_via_check_state() -> None:
    """``repair_state`` must clamp negative tracers back to their minimum.

    Without this, the explicit FABM update drives selmaprotbas phosphate
    negative within ~3 months of the lake_erken run and the rate laws emit
    NaNs that spread across the whole column. ``check_state(repair=True)`` is
    exactly the routine GOTM-FABM drives via ``fabm_check_state``.
    """
    raw = np.array([[-1.0, 2.0, -0.5], [0.1, -3.0, 4.0]], dtype=np.float64)
    cc = raw.copy()
    model = _FakeRepairModel(raw.shape, minimum=0.0)
    model.state[:] = cc  # emulates _set_model_state(model, cc)

    _repair_state(model, cc)

    np.testing.assert_allclose(cc, np.maximum(raw, 0.0))
    assert np.all(cc >= 0.0)


def test_repair_state_falls_back_to_clipping_negatives_to_zero() -> None:
    """Without a ``check_state`` method, negatives are clipped to zero."""

    class _NoCheckModel:
        pass

    cc = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    _repair_state(_NoCheckModel(), cc)
    np.testing.assert_allclose(cc, [[0.0, 2.0], [3.0, 0.0]])
