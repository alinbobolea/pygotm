"""Tests for FABM interface wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.fabm.gotm_fabm import (
    FabmState,
    calculate_conserved_quantities,
    clean_gotm_fabm,
    configure_gotm_fabm,
    do_gotm_fabm,
    do_repair_state,
    gotm_driver_fatal_error,
    gotm_fabm_create_model,
    init_gotm_fabm,
    light,
    register_bulk_observation,
    register_field,
    register_horizontal_observation,
    register_scalar_observation,
    right_hand_side_ppdd,
    right_hand_side_rhs,
    set_env_gotm_fabm,
)


class MockModel:
    state_variable_count = 2

    def __init__(self, _path: str | None = None) -> None:
        self.started = False
        self.environment: dict[str, object] = {}

    def initialize_state(self, cc: np.ndarray) -> None:
        cc[:] = 1.0

    def start(self) -> None:
        self.started = True

    def set_environment(self, **kwargs: object) -> None:
        self.environment.update(kwargs)

    def get_rates(self, cc: np.ndarray) -> np.ndarray:
        return np.ones_like(cc) * 0.5

    def get_sources(self, cc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.ones_like(cc), np.ones_like(cc) * 0.25

    def repair_state(self, cc: np.ndarray) -> None:
        np.maximum(cc, 0.0, out=cc)


def test_configure_and_create_model_with_factory() -> None:
    state = FabmState()
    configure_gotm_fabm(state, {"use": True, "repair_state": True})
    model = gotm_fabm_create_model(state, "fabm.yaml", model_factory=MockModel)

    assert state.fabm_calc
    assert state.repair_state
    assert isinstance(model, MockModel)


def test_init_register_environment_rhs_and_step() -> None:
    state = FabmState(fabm_calc=True, model=MockModel())
    init_gotm_fabm(state, 3, 2.0)
    assert state.cc is not None
    np.testing.assert_allclose(state.cc, 1.0)

    register_scalar_observation(state, "scalar", 2.0)
    register_bulk_observation(state, "bulk", np.ones(4), np.ones(4) * 10.0)
    register_horizontal_observation(state, "horizontal", 3.0, 20.0)
    assert len(state.observations) == 3

    set_env_gotm_fabm(state, temperature=np.ones(4))
    assert "temperature" in state.environment

    rhs = np.zeros_like(state.cc)
    right_hand_side_rhs(state, 1, 2, 3, state.cc, rhs)
    np.testing.assert_allclose(rhs, 0.5)

    do_gotm_fabm(state, 3, 1)
    np.testing.assert_allclose(state.cc[:, 1:], 2.0)


def test_ppdd_repair_light_conserved_and_clean() -> None:
    state = FabmState(fabm_calc=True, model=MockModel())
    init_gotm_fabm(state, 2, 1.0)
    assert state.cc is not None
    pp = np.zeros_like(state.cc)
    dd = np.zeros_like(state.cc)

    right_hand_side_ppdd(state, 1, 2, 2, state.cc, pp, dd)
    np.testing.assert_allclose(pp, 1.0)
    np.testing.assert_allclose(dd, 0.25)

    state.cc[0, 1] = -5.0
    do_repair_state(state, 2, "test")
    assert state.cc[0, 1] == 0.0

    light(state, 2)
    assert state.bioshade is not None
    assert np.all(state.bioshade <= 1.0)

    total = np.zeros(2)
    calculate_conserved_quantities(state, 2, np.array([0.0, 2.0, 3.0]), total)
    assert np.all(total >= 0.0)

    clean_gotm_fabm(state)
    assert state.cc is None
    assert state.observations == []


def test_register_field_and_fatal_error() -> None:
    state = FabmState()
    register_field(state, "oxygen", prefix="fabm_", data0d=1.0)
    assert "fabm_oxygen" in state.registered_fields

    with pytest.raises(RuntimeError, match="loc: bad"):
        gotm_driver_fatal_error(None, "loc", "bad")
