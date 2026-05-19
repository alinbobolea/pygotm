from __future__ import annotations

import numpy as np

from pygotm.icethm.driver import init_ice, outputs_to_buffers, step_ice
from pygotm.icethm.params import IceModelEnum, make_ice_params


def test_init_ice_sets_winton_layers_from_air_and_freezing_point() -> None:
    state = init_ice(
        make_ice_params(model=IceModelEnum.WINTON, Hice_init=1.0),
        T_air_init=-10.0,
        S_sfc_init=35.0,
    )

    assert state.Hice[0] == 1.0
    assert state.ice_cover[0] == 2
    assert state.T1[0] == -10.0
    assert state.T2[0] == state.Tf[0]
    assert state.Tice_surface[0] == -10.0


def test_step_ice_dispatches_simple() -> None:
    state = init_ice(
        make_ice_params(model=IceModelEnum.SIMPLE),
        T_air_init=0.0,
        S_sfc_init=35.0,
    )

    flux = step_ice(
        int(IceModelEnum.SIMPLE),
        -2.1,
        35.0,
        0.0,
        1.0,
        60.0,
        1.0e-5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        state.Hice,
        state.Hsnow,
        state.Hfrazil,
        state.T1,
        state.T2,
        state.Tice_surface,
        state.fdd,
        state.ice_cover,
        state.Tf,
        state.albedo_ice,
        state.transmissivity,
        state.ocean_ice_flux,
        state.ocean_ice_heat_flux,
        state.ocean_ice_salt_flux,
        state.surface_ice_energy,
        state.bottom_ice_energy,
        state.melt_rate,
        state.T_melt,
        state.S_melt,
    )

    assert flux == 0.0
    assert state.Tf[0] == -0.0575 * 35.0


def test_outputs_to_buffers_writes_known_names() -> None:
    state = init_ice(
        make_ice_params(model=IceModelEnum.WINTON, Hice_init=1.5),
        T_air_init=0.0,
        S_sfc_init=35.0,
    )
    buffers = {
        "Hice": np.zeros(2, dtype=np.float64),
        "Tf": np.zeros(2, dtype=np.float64),
    }

    outputs_to_buffers(state, buffers, 1)

    assert buffers["Hice"][1] == 1.5
    assert buffers["Tf"][1] == state.Tf[0]
