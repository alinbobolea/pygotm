"""Tests for GOTM lake stream forcing."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.input.input import close_input, init_input
from pygotm.observations.streams import (
    BOTTOM_FLOW,
    DEPTH_RANGE,
    INTERLEAVING,
    SURFACE_FLOW,
    configure_streams_from_document,
    post_init_streams,
    unesco_density,
    update_streams,
)


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    zi = np.asarray([-3.0, -2.0, -1.0, 0.0], dtype=np.float64)
    z = np.asarray([0.0, -2.5, -1.5, -0.5], dtype=np.float64)
    return h, zi, z


def _empty_sources() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    return tuple(np.zeros(4, dtype=np.float64) for _ in range(5))


def test_unesco_density_matches_fortran_check_value() -> None:
    assert unesco_density(35.0, 25.0, 1000.0, True) == pytest.approx(
        1062.53817,
        rel=1.0e-8,
    )


def test_surface_inflow_populates_top_layer_sources() -> None:
    h, zi, z = _grid()
    state = configure_streams_from_document(
        {
            "streams": {
                "in": {
                    "method": SURFACE_FLOW,
                    "flow": {"constant_value": 2.0},
                    "temp": {"constant_value": 10.0},
                    "salt": {"constant_value": 0.5},
                }
            }
        },
        lake=True,
    )
    init_input(3)
    try:
        post_init_streams(state, 3)
        Qs, Qt, Ls, Lt, Qlayer = _empty_sources()

        update_streams(
            state,
            3,
            60.0,
            np.zeros(4),
            np.zeros(4),
            z,
            zi,
            h,
            Qs,
            Qt,
            Ls,
            Lt,
            Qlayer,
        )
    finally:
        close_input()

    np.testing.assert_allclose(Qlayer, [0.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(Qt, [0.0, 0.0, 0.0, 20.0])
    np.testing.assert_allclose(Qs, [0.0, 0.0, 0.0, 1.0])
    assert state.int_inflow == pytest.approx(120.0)
    assert state.int_outflow == pytest.approx(0.0)


def test_stream_fabm_concentrations_are_registered_and_buffered() -> None:
    state = configure_streams_from_document(
        {
            "streams": {
                "in": {
                    "method": SURFACE_FLOW,
                    "flow": {"constant_value": 1.0},
                    "temp": {"constant_value": 10.0},
                    "selmaprotbas_po": {"constant_value": 2.5},
                },
                "out": {
                    "method": BOTTOM_FLOW,
                    "flow": {"constant_value": -0.5},
                    "selmaprotbas_nn": {"constant_value": 7.0},
                },
            }
        },
        lake=True,
    )
    init_input(3)
    try:
        post_init_streams(state, 3)
    finally:
        close_input()

    assert state.concentration_names == ("selmaprotbas_po", "selmaprotbas_nn")
    np.testing.assert_array_equal(state.concentration_has["selmaprotbas_po"], [1, 0])
    np.testing.assert_array_equal(state.concentration_has["selmaprotbas_nn"], [0, 1])
    np.testing.assert_allclose(
        state.concentration_values["selmaprotbas_po"], [2.5, 0.0]
    )
    np.testing.assert_allclose(
        state.concentration_values["selmaprotbas_nn"], [0.0, 7.0]
    )


def test_depth_range_weights_match_overlap_lengths() -> None:
    h, zi, z = _grid()
    state = configure_streams_from_document(
        {
            "streams": {
                "range": {
                    "method": DEPTH_RANGE,
                    "zl": -2.5,
                    "zu": -0.5,
                    "flow": {"constant_value": 4.0},
                    "temp": {"constant_value": 8.0},
                    "salt": {"constant_value": 0.0},
                }
            }
        },
        lake=True,
    )
    init_input(3)
    try:
        post_init_streams(state, 3)
        Qs, Qt, Ls, Lt, Qlayer = _empty_sources()
        update_streams(
            state,
            3,
            1.0,
            np.zeros(4),
            np.zeros(4),
            z,
            zi,
            h,
            Qs,
            Qt,
            Ls,
            Lt,
            Qlayer,
        )
    finally:
        close_input()

    assert state.weights is not None
    np.testing.assert_allclose(state.weights[0], [0.0, 0.25, 0.5, 0.25])
    np.testing.assert_allclose(Qlayer, [0.0, 1.0, 2.0, 1.0])


def test_outflow_without_tracers_becomes_linear_sink() -> None:
    h, zi, z = _grid()
    state = configure_streams_from_document(
        {
            "streams": {
                "out": {
                    "method": BOTTOM_FLOW,
                    "flow": {"constant_value": -3.0},
                    "temp": {"constant_value": -1.0},
                    "salt": {"constant_value": -1.0},
                }
            }
        },
        lake=True,
    )
    init_input(3)
    try:
        post_init_streams(state, 3)
        Qs, Qt, Ls, Lt, Qlayer = _empty_sources()
        update_streams(
            state,
            3,
            10.0,
            np.zeros(4),
            np.zeros(4),
            z,
            zi,
            h,
            Qs,
            Qt,
            Ls,
            Lt,
            Qlayer,
        )
    finally:
        close_input()

    np.testing.assert_allclose(Qlayer, [0.0, -3.0, 0.0, 0.0])
    np.testing.assert_allclose(Lt, [0.0, -3.0, 0.0, 0.0])
    np.testing.assert_allclose(Ls, [0.0, -3.0, 0.0, 0.0])
    np.testing.assert_allclose(Qt, 0.0)
    np.testing.assert_allclose(Qs, 0.0)
    assert state.int_outflow == pytest.approx(-30.0)


def test_interleaving_with_dense_freshwater_selects_bottom_layer() -> None:
    h, zi, z = _grid()
    state = configure_streams_from_document(
        {
            "streams": {
                "dense": {
                    "method": INTERLEAVING,
                    "flow": {"constant_value": 1.0},
                    "temp": {"constant_value": 4.0},
                    "salt": {"constant_value": 0.0},
                }
            }
        },
        lake=True,
    )
    init_input(3)
    try:
        post_init_streams(state, 3)
        Qs, Qt, Ls, Lt, Qlayer = _empty_sources()
        ambient_T = np.asarray([0.0, 20.0, 20.0, 20.0], dtype=np.float64)
        update_streams(
            state,
            3,
            1.0,
            np.zeros(4),
            ambient_T,
            z,
            zi,
            h,
            Qs,
            Qt,
            Ls,
            Lt,
            Qlayer,
        )
    finally:
        close_input()

    assert state.weights is not None
    np.testing.assert_allclose(state.weights[0], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(Qlayer, [0.0, 1.0, 0.0, 0.0])
