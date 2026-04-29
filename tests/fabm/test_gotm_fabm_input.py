"""Tests for FABM input registration."""

from __future__ import annotations

import numpy as np

from pygotm.fabm.gotm_fabm import FabmState
from pygotm.fabm.gotm_fabm_input import (
    FabmInputState,
    append_input,
    configure_gotm_fabm_input,
    fabm_input_create,
    init_gotm_fabm_input,
)
from pygotm.input.input import close_input, init_input


def test_create_and_append_scalar_input() -> None:
    state = FabmInputState()
    variable = fabm_input_create(
        state,
        "nutrient",
        {"constant_value": 3.0},
        scalar_id="nutrient_id",
    )

    assert variable.scalar_input is not None
    assert len(state.variables) == 1

    other = fabm_input_create(FabmInputState(), "detritus", scalar_id="detritus")
    append_input(state, other)
    assert len(state.variables) == 2


def test_configure_from_mapping_creates_variables() -> None:
    state = FabmInputState()
    configure_gotm_fabm_input(
        state,
        {
            "oxygen": {"profile": True, "constant_value": 1.0},
            "alkalinity": {"constant_value": 2.0},
        },
    )

    assert len(state.variables) == 2
    assert state.variables[0].profile_input is not None
    assert state.variables[1].scalar_input is not None


def test_init_registers_profile_relaxation_layers() -> None:
    init_input(4)
    try:
        input_state = FabmInputState()
        variable = fabm_input_create(
            input_state,
            "oxygen",
            {
                "profile": True,
                "constant_value": 7.0,
                "relax_tau": 100.0,
                "relax_tau_bot": 10.0,
                "relax_tau_surf": 20.0,
                "thickness_bot": 1.1,
                "thickness_surf": 1.1,
            },
            interior_id="oxygen_id",
        )
        fabm = FabmState()
        h = np.ones(5)

        init_gotm_fabm_input(input_state, fabm, 4, h)

        assert variable.relax_tau_1d is not None
        assert variable.relax_tau_1d[1] == 10.0
        assert variable.relax_tau_1d[4] == 20.0
        assert len(fabm.observations) == 1
    finally:
        close_input()


def test_init_registers_scalar_and_horizontal_observations() -> None:
    init_input(2)
    try:
        input_state = FabmInputState()
        fabm_input_create(input_state, "scalar", scalar_id="scalar_id")
        fabm_input_create(
            input_state,
            "horizontal",
            horizontal_id="horizontal_id",
            spec={"relax_tau": 5.0},
        )
        fabm = FabmState()

        init_gotm_fabm_input(input_state, fabm, 2, np.ones(3))

        assert [obs.kind for obs in fabm.observations] == ["scalar", "horizontal"]
    finally:
        close_input()
