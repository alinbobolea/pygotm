"""Tests for the CVMix interface module."""

from __future__ import annotations

import numpy as np

from pygotm.cvmix.gotm_cvmix import (
    CVMIX_SHEAR_PP,
    CVMixState,
    bottom_layer,
    clean_cvmix,
    do_cvmix,
    init_cvmix_yaml,
    interior_conv,
    interior_nonconv,
    post_init_cvmix,
    surface_layer,
)


def test_init_yaml_applies_nested_options() -> None:
    state = CVMixState()
    init_cvmix_yaml(
        state,
        {
            "interior": {
                "use": True,
                "background": {"use": True, "diffusivity": 2.0e-5},
                "shear": {"use": True, "mix_scheme": "PP"},
                "convection": {"use": True, "diffusivity": 0.5},
            }
        },
    )

    assert state.use_interior
    assert state.use_background
    assert state.background_diffusivity == 2.0e-5
    assert state.shear_mix_scheme == CVMIX_SHEAR_PP
    assert state.use_convection


def test_post_init_allocates_grid_arrays() -> None:
    state = CVMixState()
    post_init_cvmix(state, 4, 20.0, 9.81, 1027.0)

    assert state.z_w is not None
    assert state.z_w[0] == -20.0
    assert state.z_w[-1] == 0.0
    assert state.cvmix_gorho0 == 9.81 / 1027.0


def test_interior_background_shear_and_convection_update_diffusivities() -> None:
    nlev = 5
    state = CVMixState(
        use_interior=True,
        use_background=True,
        use_shear=True,
        use_convection=True,
    )
    NN = np.full(nlev + 1, 1.0e-5)
    NN[2] = -1.0e-6
    SS = np.full(nlev + 1, 1.0e-4)
    num = np.zeros(nlev + 1)
    nuh = np.zeros(nlev + 1)
    nus = np.zeros(nlev + 1)
    Rig = np.zeros(nlev + 1)

    interior_nonconv(state, nlev, NN, NN, NN, SS, num, nuh, nus, Rig)
    interior_conv(state, nlev, NN, num, nuh, nus)

    assert np.all(num[1:nlev] >= state.background_viscosity)
    assert np.all(nuh[1:nlev] >= state.background_diffusivity)
    assert num[2] >= state.convection_viscosity
    assert Rig[1] > 0.0


def test_surface_and_bottom_layers_enhance_boundary_diffusivity() -> None:
    nlev = 6
    state = CVMixState(use_surface_layer=True, use_bottom_layer=True)
    h = np.ones(nlev + 1)
    num = np.zeros(nlev + 1)
    nuh = np.zeros(nlev + 1)

    surface_layer(state, nlev, h, num, nuh, u_taus=0.02, hbl=3.0)
    bottom_layer(state, nlev, h, num, nuh, u_taub=0.03, hbl=3.0)

    assert state.zsbl == -3.0
    assert state.zbbl == 3.0
    assert np.any(num > 0.0)
    assert np.any(nuh > 0.0)


def test_do_cvmix_dispatch_and_clean() -> None:
    nlev = 4
    state = CVMixState(use_interior=True, use_background=True)
    post_init_cvmix(state, nlev, 8.0, 9.81, 1027.0)
    h = np.ones(nlev + 1)
    arr = np.zeros(nlev + 1)
    num = np.zeros(nlev + 1)
    nuh = np.zeros(nlev + 1)
    nus = np.zeros(nlev + 1)
    Rig = np.zeros(nlev + 1)

    do_cvmix(
        state,
        nlev,
        8.0,
        h,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        0.0,
        0.0,
        num,
        nuh,
        nus,
        Rig,
    )
    clean_cvmix(state)

    assert np.all(num[1:nlev] >= state.background_viscosity)
    assert state.z_w is None
