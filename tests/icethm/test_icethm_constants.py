from __future__ import annotations

from pygotm.icethm.constants import (
    ALB_ICE,
    FREEZE_SLOPE,
    L_ICE,
    LAMBDA1,
    MU_TS,
    RHO_ICE,
)
from pygotm.icethm.params import IceModelEnum, canonical_ice_model, make_ice_params
from pygotm.icethm.state import make_ice_state


def test_constants_are_float64_compatible_values() -> None:
    assert RHO_ICE > 900.0
    assert L_ICE > 3.0e5
    assert FREEZE_SLOPE == 0.0575
    assert MU_TS == 0.054
    assert 0.0 < ALB_ICE < 1.0
    assert LAMBDA1 < 0.0


def test_params_and_state_defaults() -> None:
    params = make_ice_params(model="winton", Hice_init=1.0)
    state = make_ice_state()

    assert params.model == IceModelEnum.WINTON
    assert canonical_ice_model("basal-melt") == IceModelEnum.BASAL_MELT
    assert state.Hice.shape == (1,)
    assert state.Hice.dtype.name == "float64"
    assert state.ice_cover.dtype.name == "int32"
    assert state.transmissivity[0] == 1.0
