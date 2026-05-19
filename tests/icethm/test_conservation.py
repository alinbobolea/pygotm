from __future__ import annotations

import math

from pygotm.icethm.driver import init_ice, step_ice
from pygotm.icethm.params import IceModelEnum, make_ice_params


def test_winton_driver_conservation_bounds() -> None:
    state = init_ice(
        make_ice_params(
            model=IceModelEnum.WINTON,
            Hice_init=1.0,
            ocean_ice_flux_init=10.0,
        ),
        T_air_init=-5.0,
        S_sfc_init=33.0,
    )

    step_ice(
        int(IceModelEnum.WINTON),
        -1.8,
        33.0,
        -5.0,
        1.0,
        3600.0,
        0.0,
        20.0,
        -40.0,
        -10.0,
        -5.0,
        0.0,
        0.01,
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

    assert state.Hice[0] >= 0.0
    assert state.Hsnow[0] >= 0.0
    assert 0.0 <= state.transmissivity[0] <= 1.0
    assert 0.0 <= state.albedo_ice[0] <= 1.0
    assert math.isfinite(state.Tf[0])
    assert math.isfinite(state.ocean_ice_heat_flux[0])
