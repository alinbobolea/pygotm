"""Tests for pygotm.airsea.airsea_variables."""

from __future__ import annotations

import pytest

from pygotm.airsea.airsea_variables import (
    BERLIAND_BERLIAND,
    BIGNAMI,
    CLARK,
    COGLEY,
    CONST,
    HASTENRATH_LAMB,
    JOSEY1,
    JOSEY2,
    PAYNE,
    AirSeaState,
    bolz,
    const06,
    cpa,
    cpw,
    emiss,
    g,
    kappa,
    kelvin,
    rgas,
    rho_0,
)


def test_import_and_instantiate() -> None:
    state = AirSeaState()
    assert state is not None


def test_public_constants_match_fortran_values() -> None:
    assert cpa == pytest.approx(1008.0)
    assert cpw == pytest.approx(3985.0)
    assert emiss == pytest.approx(0.97)
    assert bolz == pytest.approx(5.670374419e-8)
    assert kelvin == pytest.approx(273.15)
    assert const06 == pytest.approx(0.62198)
    assert rgas == pytest.approx(287.1)
    assert g == pytest.approx(9.81)
    assert rho_0 == pytest.approx(1025.0)
    assert kappa == pytest.approx(0.41)


def test_selector_constants_match_fortran_values() -> None:
    assert CONST == 0
    assert PAYNE == 1
    assert COGLEY == 2
    assert CLARK == 3
    assert HASTENRATH_LAMB == 4
    assert BIGNAMI == 5
    assert BERLIAND_BERLIAND == 6
    assert JOSEY1 == 7
    assert JOSEY2 == 8


def test_default_state_matches_airsea_post_init_baseline() -> None:
    state = AirSeaState()
    assert state.es == pytest.approx(0.0)
    assert state.ea == pytest.approx(0.0)
    assert state.qs == pytest.approx(0.0)
    assert state.qa == pytest.approx(0.0)
    assert state.L == pytest.approx(0.0)
    assert state.rhoa == pytest.approx(0.0)
    assert state.ta == pytest.approx(0.0)
    assert state.rain_impact is False
    assert state.calc_evaporation is False
