from __future__ import annotations

import math

from pygotm import constants


def test_gotm_constant_values_match_authoritative_sources() -> None:
    assert constants.AIRSEA_SPECIFIC_HEAT_AIR_J_KG_K == 1008.0
    assert constants.AIRSEA_SPECIFIC_HEAT_WATER_J_KG_K == 3985.0
    assert constants.AIRSEA_EMISSIVITY == 0.97
    assert constants.STANDARD_GRAVITY_M_S2 == 9.81
    assert constants.AIRSEA_REFERENCE_DENSITY_KG_M3 == 1025.0
    assert constants.DENSITY_MODULE_REFERENCE_DENSITY_KG_M3 == 1027.0


def test_angle_conversion_constants_are_consistent() -> None:
    assert constants.PI == math.pi
    assert math.isclose(constants.DEG_TO_RAD * constants.RAD_TO_DEG, 1.0)
