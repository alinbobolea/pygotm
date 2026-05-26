"""Tests for pygotm.observations.observations."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from pygotm.config.settings import GotmSettings, load_settings
from pygotm.input.input import close_input, init_input
from pygotm.meanflow.meanflow import MeanflowState, init_meanflow, post_init_meanflow
from pygotm.meanflow.updategrid import updategrid
from pygotm.observations.observations import (
    ANALYTICAL,
    ANALYTICAL_OFFSET,
    CONST_PROF,
    FROMFILE,
    TWO_LAYERS,
    ObservationsState,
    clean_observations,
    get_all_obs,
    init_observations,
    post_init_observations,
)
from pygotm.util.density import METHOD_LINEAR_USER, DensityState, init_density


@pytest.fixture(autouse=True)
def _cleanup_input_manager() -> Iterator[None]:
    close_input()
    yield
    close_input()


def _grid(nlev: int = 6, depth: float = 12.0) -> tuple[MeanflowState, DensityState]:
    meanflow = MeanflowState()
    init_meanflow(meanflow)
    meanflow.depth = depth
    meanflow.grid_method = 0
    post_init_meanflow(meanflow, nlev, latitude=45.0)
    updategrid(meanflow, nlev, 3600.0, 0.0)
    density = DensityState()
    density.density_method = METHOD_LINEAR_USER
    density.rho0 = 1027.0
    density.alpha0 = 2.0e-4
    density.beta0 = 7.5e-4
    init_density(density, nlev)
    return meanflow, density


def test_init_observations_maps_settings_to_runtime_state() -> None:
    settings = GotmSettings.model_validate(
        {
            "temperature": {"method": "two_layer", "type": "conservative"},
            "salinity": {"method": "constant", "type": "absolute"},
            "light_extinction": {"method": "jerlov-ii"},
            "w": {"adv_discr": "muscl"},
        }
    )
    state = ObservationsState()
    init_observations(state, settings)
    assert state.tprof_input.method == ANALYTICAL_OFFSET + TWO_LAYERS
    assert state.sprof_input.method == ANALYTICAL_OFFSET + CONST_PROF
    assert state.initial_temperature_type == 3
    assert state.initial_salinity_type == 2
    assert state.extinct_method == 5


def test_init_observations_prefers_nested_mimic3d_vertical_velocity() -> None:
    settings = GotmSettings.model_validate(
        {
            "w": {
                "max": {"method": "file", "file": "top.dat"},
                "height": {"method": "file", "file": "top_height.dat"},
            },
            "mimic_3d": {
                "w": {
                    "max": {"method": "file", "file": "nested.dat", "column": 2},
                    "height": {
                        "method": "file",
                        "file": "nested_height.dat",
                        "column": 3,
                    },
                    "adv_discr": "superbee",
                }
            },
        }
    )
    state = ObservationsState()

    init_observations(state, settings)

    assert state.w_adv_input.method == FROMFILE
    assert state.w_adv_input.path == "nested.dat"
    assert state.w_adv_input.index == 2
    assert state.w_height_input.path == "nested_height.dat"
    assert state.w_height_input.index == 3


def test_init_observations_keeps_top_level_vertical_velocity_compatibility() -> None:
    settings = GotmSettings.model_validate(
        {
            "w": {
                "max": {"method": "file", "file": "top.dat", "column": 4},
                "height": {"method": "file", "file": "top_height.dat"},
            }
        }
    )
    state = ObservationsState()

    init_observations(state, settings)

    assert state.w_adv_input.method == FROMFILE
    assert state.w_adv_input.path == "top.dat"
    assert state.w_adv_input.index == 4


def test_init_observations_disables_legacy_shared_vertical_velocity_file() -> None:
    settings = GotmSettings.model_validate(
        {
            "mimic_3d": {
                "w": {
                    "max": {"method": "file", "file": "w_adv.dat", "column": 1},
                    "height": {"method": "file", "file": "w_adv.dat", "column": 2},
                }
            }
        }
    )
    state = ObservationsState()

    init_observations(state, settings)

    assert state.w_adv_input.method == 0
    assert state.w_adv_input.path == ""
    assert state.w_adv_input.index == 1
    assert state.w_height_input.path == "w_adv.dat"
    assert state.w_height_input.index == 1


def test_init_observations_activates_constant_vertical_advection() -> None:
    settings = GotmSettings.model_validate(
        {
            "mimic_3d": {
                "w": {
                    "max": {"method": "constant", "constant_value": 1.0e-4},
                    "height": {"method": "constant", "constant_value": -5.0},
                }
            }
        }
    )
    state = ObservationsState()

    init_observations(state, settings)

    assert state.w_adv_input.method == 1
    assert state.w_adv_input.constant_value == pytest.approx(1.0e-4)
    assert state.w_height_input.method == 0
    assert state.w_height_input.constant_value == pytest.approx(-5.0)


def test_init_observations_uses_real_case_nested_zeta_period() -> None:
    from tests.fixtures import bundled_case_path

    settings = load_settings(bundled_case_path("seagrass"))
    state = ObservationsState()

    init_observations(state, settings)

    assert state.period_1 == pytest.approx(15.0)


def test_post_init_observations_creates_two_layer_profile_and_relaxation() -> None:
    settings = GotmSettings.model_validate(
        {
            "temperature": {
                "method": "two_layer",
                "two_layer": {"z_s": 2.0, "t_s": 10.0, "z_b": 6.0, "t_b": 4.0},
                "relax": {"tau": 50.0, "h_s": 2.0, "tau_s": 20.0},
            },
            "salinity": {
                "method": "constant",
                "constant_value": 35.0,
                "relax": {"tau": 100.0, "h_b": 2.0, "tau_b": 40.0},
            },
        }
    )
    state = ObservationsState()
    init_observations(state, settings)
    meanflow, density = _grid()
    assert meanflow.z is not None
    assert meanflow.zi is not None
    assert meanflow.h is not None
    init_input(6)
    post_init_observations(
        state,
        meanflow.depth,
        6,
        meanflow.z,
        meanflow.zi,
        meanflow.h,
        meanflow.gravity,
        density,
    )
    assert state.tprof_input.data is not None
    assert state.TRelaxTau is not None
    assert state.SRelaxTau is not None
    assert state.tprof_input.data[6] == pytest.approx(10.0)
    assert state.tprof_input.data[1] <= 4.0 + 1.0e-12
    assert np.any(np.isclose(state.TRelaxTau[1:], 20.0))
    assert np.any(np.isclose(state.SRelaxTau[1:], 40.0))


def test_const_nn_profile_requires_constant_counterpart() -> None:
    settings = GotmSettings.model_validate(
        {
            "temperature": {"method": "two_layer"},
            "salinity": {"method": "buoyancy", "NN": 1.0e-4},
        }
    )
    state = ObservationsState()
    init_observations(state, settings)
    meanflow, density = _grid()
    assert meanflow.z is not None
    assert meanflow.zi is not None
    assert meanflow.h is not None
    init_input(6)
    with pytest.raises(ValueError, match="requires constant temperature"):
        post_init_observations(
            state,
            meanflow.depth,
            6,
            meanflow.z,
            meanflow.zi,
            meanflow.h,
            meanflow.gravity,
            density,
        )


def test_get_all_obs_updates_analytical_tides() -> None:
    state = ObservationsState()
    state.dpdx_input.method = ANALYTICAL
    state.dpdy_input.method = ANALYTICAL
    state.zeta_input.method = ANALYTICAL
    state.AmpMu = 2.0
    state.AmpMv = 1.0
    state.amp_1 = 0.5
    state.PeriodM = 100.0
    state.period_1 = 100.0
    get_all_obs(state, 0, 25, 0, np.zeros(1), fsecs=25.0)
    assert state.dpdx_input.value == pytest.approx(2.0)
    assert state.dpdy_input.value == pytest.approx(1.0)
    assert state.zeta_input.value == pytest.approx(0.5)


def test_builtin_light_extinction_sets_jerlov_values() -> None:
    settings = GotmSettings.model_validate(
        {"light_extinction": {"method": "jerlov-ib"}}
    )
    state = ObservationsState()
    init_observations(state, settings)
    meanflow, density = _grid()
    assert meanflow.z is not None
    assert meanflow.zi is not None
    assert meanflow.h is not None
    init_input(6)
    post_init_observations(
        state,
        meanflow.depth,
        6,
        meanflow.z,
        meanflow.zi,
        meanflow.h,
        meanflow.gravity,
        density,
    )
    assert state.A_input.value == pytest.approx(0.67)
    assert state.g1_input.value == pytest.approx(1.0)
    assert state.g2_input.value == pytest.approx(17.0)


def test_clean_observations_releases_allocations() -> None:
    state = ObservationsState(
        idpdx=np.zeros(3),
        idpdy=np.zeros(3),
        SRelaxTau=np.ones(3),
        TRelaxTau=np.ones(3),
    )
    clean_observations(state)
    assert state.idpdx is None
    assert state.idpdy is None
    assert state.SRelaxTau is None
    assert state.TRelaxTau is None
