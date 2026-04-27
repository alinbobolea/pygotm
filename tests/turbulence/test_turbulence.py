"""Tests for pygotm.turbulence.turbulence — state fields and dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.turbulence.turbulence import (
    CHCD01A,
    Constant,
    Dirichlet,
    Munk_Anderson,
    Neumann,
    Schumann_Gerz,
    TurbulenceState,
    algebraic,
    clean_turbulence,
    diss_eq,
    do_turbulence,
    first_order,
    generic_eq,
    init_turbulence,
    injection,
    k_bc,
    logarithmic,
    no_model,
    omega_bc,
    omega_eq,
    post_init_turbulence,
    psi_bc,
    q2l_bc,
    q2over2_bc,
    quasi_Eq_H15,
    second_order,
    tke_keps,
    weak_Eq_Kb_Eq,
)

_NLEV = 12
_ARRAY_FIELD_NAMES = (
    "tke",
    "eps",
    "omega",
    "L",
    "tkeo",
    "kb",
    "epsb",
    "P",
    "B",
    "Pb",
    "Px",
    "PSTK",
    "num",
    "nuh",
    "nus",
    "nucl",
    "gamu",
    "gamv",
    "gamb",
    "gamh",
    "gams",
    "cmue1",
    "cmue2",
    "cmue3",
    "sq_var",
    "sl_var",
    "gam",
    "as_",
    "an",
    "at",
    "av",
    "aw",
    "SPF",
    "r",
    "Rig",
    "xRf",
    "uu",
    "vv",
    "ww",
)


def _make_inputs(nlev: int = _NLEV) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = nlev + 1
    return np.ones(n), np.zeros(n), np.zeros(n)


def _make_state(
    nlev: int = _NLEV,
    *,
    turb_method: int = first_order,
    **overrides: int | float | bool,
) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, overrides=overrides, turb_method=turb_method)
    post_init_turbulence(state, nlev)
    return state


def test_import_and_instantiate() -> None:
    state = TurbulenceState()
    assert state is not None


def test_default_scalar_configuration_values() -> None:
    state = TurbulenceState()
    assert state.turb_method == first_order
    assert state.tke_method == tke_keps
    assert state.len_scale_method == diss_eq
    assert state.stab_method == Schumann_Gerz
    assert state.k_ubc == Neumann
    assert state.psi_ubc == Neumann
    assert state.ubc_type == logarithmic
    assert state.cm0_fix == pytest.approx(0.5477)
    assert state.Prandtl0_fix == pytest.approx(0.74)
    assert state.kappa == pytest.approx(0.4)
    assert state.k_min == pytest.approx(1.0e-8)
    assert state.eps_min == pytest.approx(1.0e-12)
    assert state.compute_kappa is True
    assert state.compute_c3 is True
    assert state.compute_param is False


def test_public_parameter_constants_match_fortran_values() -> None:
    assert no_model == 0
    assert algebraic == 1
    assert first_order == 2
    assert second_order == 3
    assert Constant == 1
    assert Munk_Anderson == 2
    assert Schumann_Gerz == 3
    assert Dirichlet == 0
    assert Neumann == 1
    assert generic_eq == 10
    assert omega_eq == 11


def test_default_arrays_none_before_post_init() -> None:
    state = TurbulenceState()
    for name in _ARRAY_FIELD_NAMES:
        assert getattr(state, name) is None, f"{name} should be None before post_init"


def test_init_turbulence_accepts_mapping_and_keyword_overrides() -> None:
    state = TurbulenceState()
    init_turbulence(
        state,
        overrides={"turb_method": no_model, "const_num": 1.5e-3},
        const_nuh=2.5e-3,
        len_scale_method=omega_eq,
        compute_param=True,
    )
    assert state.turb_method == no_model
    assert state.const_num == pytest.approx(1.5e-3)
    assert state.const_nuh == pytest.approx(2.5e-3)
    assert state.len_scale_method == omega_eq
    assert state.compute_param is True


def test_init_turbulence_rejects_unknown_override() -> None:
    state = TurbulenceState()
    with pytest.raises(AttributeError, match="unknown turbulence configuration field"):
        init_turbulence(state, not_a_field=1)


def test_init_turbulence_rejects_array_override() -> None:
    state = TurbulenceState()
    with pytest.raises(ValueError, match="allocated profile field"):
        init_turbulence(state, tke=1)


def test_array_shapes_after_post_init() -> None:
    state = _make_state()
    expected_shape = (_NLEV + 1,)
    for name in _ARRAY_FIELD_NAMES:
        array = getattr(state, name)
        assert array is not None
        assert array.shape == expected_shape


def test_post_init_default_array_values() -> None:
    state = _make_state()
    assert state.tke is not None and np.all(state.tke == state.k_min)
    assert state.tkeo is not None and np.all(state.tkeo == state.k_min)
    assert state.eps is not None and np.all(state.eps == state.eps_min)
    assert state.kb is not None and np.all(state.kb == state.kb_min)
    assert state.epsb is not None and np.all(state.epsb == state.epsb_min)
    assert state.num is not None and np.all(state.num == 1.0e-6)
    assert state.nuh is not None and np.all(state.nuh == 1.0e-6)
    assert state.nus is not None and np.all(state.nus == 1.0e-6)
    assert state.sq_var is not None and np.all(state.sq_var == state.sq)
    assert state.sl_var is not None and np.all(state.sl_var == state.sl)
    assert state.SPF is not None and np.all(state.SPF == 1.0)


def test_post_init_zero_seeded_arrays() -> None:
    state = _make_state()
    zero_seeded = (
        "omega",
        "P",
        "B",
        "Pb",
        "Px",
        "PSTK",
        "nucl",
        "gamu",
        "gamv",
        "gamb",
        "gamh",
        "gams",
        "cmue3",
        "gam",
        "at",
        "av",
        "aw",
        "r",
        "Rig",
        "xRf",
        "uu",
        "vv",
        "ww",
    )
    for name in zero_seeded:
        array = getattr(state, name)
        assert array is not None
        assert np.all(array == 0.0), f"{name} must start at zero"


def test_post_init_derives_model_constants_and_lengthscale_floor() -> None:
    state = _make_state()
    assert state.cm0 == pytest.approx(state.cm0_fix)
    assert state.cmsf == pytest.approx(state.cm0_fix)
    assert state.cde == pytest.approx(state.cm0**3)
    assert state.b1 == pytest.approx(2.0**1.5 / state.cde)
    assert state.L is not None
    expected_l_min = state.cde * state.k_min**1.5 / state.eps_min
    np.testing.assert_allclose(state.L, expected_l_min, rtol=1.0e-12)


def test_no_model_post_init_sets_constant_viscosity_and_diffusivity() -> None:
    state = TurbulenceState()
    init_turbulence(state, turb_method=no_model, const_num=2.0e-4, const_nuh=3.0e-4)
    post_init_turbulence(state, _NLEV)
    assert state.num is not None and np.all(state.num == pytest.approx(2.0e-4))
    assert state.nuh is not None and np.all(state.nuh == pytest.approx(3.0e-4))
    assert state.nus is not None and np.all(state.nus == pytest.approx(1.0e-6))


def test_post_init_accepts_zero_levels() -> None:
    state = TurbulenceState()
    init_turbulence(state)
    post_init_turbulence(state, 0)
    assert state.tke is not None
    assert state.tke.shape == (1,)


def test_k_bc_logarithmic_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state)
    state.cm0 = state.cm0_fix

    value = k_bc(state, Dirichlet, logarithmic, 0.25, 0.01, 0.012)

    assert value == pytest.approx(0.012**2 / state.cm0**2)
    assert k_bc(state, Neumann, logarithmic, 0.25, 0.01, 0.012) == pytest.approx(0.0)


def test_k_bc_injection_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state)
    state.cmsf = 0.55
    state.sig_k = 1.0
    state.cw = 100.0
    state.gen_alpha = -2.0
    state.gen_l = 0.2

    zi = 0.5
    z0 = 0.05
    u_tau = 0.01
    f_k = state.cw * u_tau**3
    capital_k = (-state.sig_k * f_k / (state.cmsf * state.gen_alpha * state.gen_l)) ** (
        2.0 / 3.0
    ) / z0**state.gen_alpha

    assert k_bc(state, Dirichlet, injection, zi, z0, u_tau) == pytest.approx(
        capital_k * (zi + z0) ** state.gen_alpha
    )
    assert k_bc(state, Neumann, injection, zi, z0, u_tau) == pytest.approx(
        -state.cmsf
        / state.sig_k
        * capital_k**1.5
        * state.gen_alpha
        * state.gen_l
        * (zi + z0) ** (1.5 * state.gen_alpha)
    )


def test_q2over2_bc_logarithmic_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state)
    state.cm0 = state.cm0_fix
    state.cde = state.cm0**3
    state.b1 = 2.0**1.5 / state.cde

    zi = 0.25
    z0 = 0.01
    u_tau = 0.012

    assert q2over2_bc(state, Dirichlet, logarithmic, zi, z0, u_tau) == pytest.approx(
        u_tau**2 * state.b1 ** (2.0 / 3.0) / 2.0
    )
    assert q2over2_bc(state, Neumann, logarithmic, zi, z0, u_tau) == pytest.approx(0.0)


def test_q2over2_bc_injection_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state)

    zi = 0.5
    z0 = 0.05
    u_tau = 0.01
    f_k = state.cw * u_tau**3
    capital_k = (-f_k / (np.sqrt(2.0) * state.sq * state.gen_alpha * state.gen_l)) ** (
        2.0 / 3.0
    ) / z0**state.gen_alpha

    assert q2over2_bc(state, Dirichlet, injection, zi, z0, u_tau) == pytest.approx(
        capital_k * (zi + z0) ** state.gen_alpha
    )
    assert q2over2_bc(state, Neumann, injection, zi, z0, u_tau) == pytest.approx(
        -np.sqrt(2.0)
        * state.sq
        * capital_k**1.5
        * state.gen_alpha
        * state.gen_l
        * (zi + z0) ** (1.5 * state.gen_alpha)
    )


def test_psi_bc_logarithmic_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state)
    state.cm0 = state.cm0_fix

    zi = 0.25
    ki = 2.0e-4
    z0 = 0.01
    u_tau = 0.012

    assert psi_bc(state, Dirichlet, logarithmic, zi, ki, z0, u_tau) == pytest.approx(
        state.cm0**state.gen_p
        * state.kappa**state.gen_n
        * ki**state.gen_m
        * (zi + z0) ** state.gen_n
    )
    assert psi_bc(state, Neumann, logarithmic, zi, ki, z0, u_tau) == pytest.approx(
        -state.gen_n
        * state.cm0 ** (state.gen_p + 1.0)
        * state.kappa ** (state.gen_n + 1.0)
        / state.sig_psi
        * ki ** (state.gen_m + 0.5)
        * (zi + z0) ** state.gen_n
    )


def test_psi_bc_injection_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state, ubc_type=injection, lbc_type=injection)
    state.cm0 = state.cm0_fix
    state.cmsf = 0.55

    zi = 0.05
    ki = 2.0e-4
    z0 = 1.0e-3
    u_tau = 0.012
    f_k = state.cw * u_tau**3
    capital_k = (-state.sig_k * f_k / (state.cmsf * state.gen_alpha * state.gen_l)) ** (
        2.0 / 3.0
    ) / z0**state.gen_alpha

    assert psi_bc(state, Dirichlet, injection, zi, ki, z0, u_tau) == pytest.approx(
        state.cm0**state.gen_p
        * capital_k**state.gen_m
        * state.gen_l**state.gen_n
        * (zi + z0) ** (state.gen_m * state.gen_alpha + state.gen_n)
    )
    assert psi_bc(state, Neumann, injection, zi, ki, z0, u_tau) == pytest.approx(
        -(state.gen_m * state.gen_alpha + state.gen_n)
        * state.cmsf
        * state.cm0**state.gen_p
        / state.sig_psi
        * capital_k ** (state.gen_m + 0.5)
        * state.gen_l ** (state.gen_n + 1.0)
        * (zi + z0) ** ((state.gen_m + 0.5) * state.gen_alpha + state.gen_n)
    )


def test_omega_bc_logarithmic_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state)
    state.cm0 = state.cm0_fix

    zi = 0.25
    ki = 2.0e-4
    z0 = 0.01
    u_tau = 0.012

    assert omega_bc(
        state,
        Dirichlet,
        logarithmic,
        zi,
        ki,
        z0,
        u_tau,
    ) == pytest.approx(ki**0.5 / (state.cm0 * state.kappa * (zi + z0)))
    assert omega_bc(
        state,
        Neumann,
        logarithmic,
        zi,
        ki,
        z0,
        u_tau,
    ) == pytest.approx(ki / (state.sig_w * (zi + z0)))


def test_omega_bc_injection_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state)
    state.cm0 = state.cm0_fix
    state.cmsf = 0.55

    zi = 0.05
    ki = 2.0e-4
    z0 = 1.0e-3
    u_tau = 0.012
    f_k = state.cw * u_tau**3
    capital_k = (-state.sig_k * f_k / (state.cmsf * state.gen_alpha * state.gen_l)) ** (
        2.0 / 3.0
    ) / z0**state.gen_alpha

    assert omega_bc(state, Dirichlet, injection, zi, ki, z0, u_tau) == pytest.approx(
        capital_k**0.5
        / (state.cm0 * state.gen_l)
        * (zi + z0) ** (0.5 * state.gen_alpha - 1.0)
    )
    assert omega_bc(state, Neumann, injection, zi, ki, z0, u_tau) == pytest.approx(
        -state.cmsf
        * capital_k
        * (0.5 * state.gen_alpha - 1.0)
        / (state.sig_w * state.cm0)
        * (zi + z0) ** (state.gen_alpha - 1.0)
    )


def test_q2l_bc_logarithmic_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state)

    zi = 0.25
    ki = 2.0e-4
    z0 = 0.01
    value = q2l_bc(state, Dirichlet, logarithmic, zi, ki, z0, 0.012)

    assert value == pytest.approx(2.0 * state.kappa * ki * (zi + z0))
    assert q2l_bc(state, Neumann, logarithmic, zi, ki, z0, 0.012) == pytest.approx(
        -2.0 * np.sqrt(2.0) * state.sl * state.kappa**2 * ki**1.5 * (zi + z0)
    )


def test_q2l_bc_injection_matches_fortran_formula() -> None:
    state = TurbulenceState()
    init_turbulence(state)
    state.cw = 100.0
    state.sq = 0.2
    state.sl = 0.2
    state.gen_alpha = -2.0
    state.gen_l = 0.2

    zi = 0.5
    ki = 2.0e-4
    z0 = 0.05
    u_tau = 0.01
    f_k = state.cw * u_tau**3
    capital_k = (-f_k / (np.sqrt(2.0) * state.sq * state.gen_alpha * state.gen_l)) ** (
        2.0 / 3.0
    ) / z0**state.gen_alpha

    assert q2l_bc(state, Dirichlet, injection, zi, ki, z0, u_tau) == pytest.approx(
        2.0 * capital_k * state.gen_l * (zi + z0) ** (state.gen_alpha + 1.0)
    )
    assert q2l_bc(state, Neumann, injection, zi, ki, z0, u_tau) == pytest.approx(
        -2.0
        * np.sqrt(2.0)
        * state.sl
        * (state.gen_alpha + 1.0)
        * capital_k**1.5
        * state.gen_l**2
        * (zi + z0) ** (1.5 * state.gen_alpha + 1.0)
    )


@pytest.mark.parametrize("nlev", [1, 5, 50])
def test_boundary_indices_accessible(nlev: int) -> None:
    state = _make_state(nlev=nlev)
    assert state.tke is not None
    _ = state.tke[0]
    _ = state.tke[nlev]


def test_post_init_rejects_negative_levels() -> None:
    state = TurbulenceState()
    with pytest.raises(ValueError, match="nlev must be non-negative"):
        post_init_turbulence(state, -1)


def test_no_nan_or_inf_in_allocated_arrays() -> None:
    state = _make_state(nlev=100)
    for name in _ARRAY_FIELD_NAMES:
        array = getattr(state, name)
        assert array is not None
        assert not np.any(np.isnan(array)), f"{name} contains NaN"
        assert not np.any(np.isinf(array)), f"{name} contains Inf"


def test_do_turbulence_no_model_is_noop() -> None:
    state = _make_state(turb_method=no_model)
    h, NN, SS = _make_inputs()
    assert state.num is not None
    assert state.nuh is not None
    num_before = state.num.copy()
    nuh_before = state.nuh.copy()
    do_turbulence(
        state,
        _NLEV,
        dt=3600.0,
        depth=25.0,
        u_taus=0.01,
        u_taub=0.01,
        z0s=0.02,
        z0b=0.001,
        h=h,
        NN=NN,
        SS=SS,
    )
    assert state.num is not None and np.array_equal(state.num, num_before)
    assert state.nuh is not None and np.array_equal(state.nuh, nuh_before)


def test_do_turbulence_cvmix_is_noop() -> None:
    state = TurbulenceState()
    init_turbulence(state, turb_method=100)
    post_init_turbulence(state, _NLEV)
    h, NN, SS = _make_inputs()
    do_turbulence(
        state,
        _NLEV,
        dt=3600.0,
        depth=25.0,
        u_taus=0.01,
        u_taub=0.01,
        z0s=0.02,
        z0b=0.001,
        h=h,
        NN=NN,
        SS=SS,
    )


def test_do_turbulence_still_rejects_untranslated_algebraic_model() -> None:
    state = _make_state(turb_method=algebraic)
    h, NN, SS = _make_inputs()
    with pytest.raises(NotImplementedError, match="algebraic diffusivity"):
        do_turbulence(
            state,
            _NLEV,
            dt=3600.0,
            depth=25.0,
            u_taus=0.01,
            u_taub=0.01,
            z0s=0.02,
            z0b=0.001,
            h=h,
            NN=NN,
            SS=SS,
        )


def test_do_turbulence_first_order_runs_translated_dispatch() -> None:
    state = _make_state()
    h = np.ones(_NLEV + 1, dtype=np.float64)
    NN = np.zeros(_NLEV + 1, dtype=np.float64)
    SS = np.full(_NLEV + 1, 1.0e-4, dtype=np.float64)

    do_turbulence(
        state,
        _NLEV,
        dt=60.0,
        depth=25.0,
        u_taus=0.01,
        u_taub=0.01,
        z0s=0.02,
        z0b=0.001,
        h=h,
        NN=NN,
        SS=SS,
    )

    assert state.num is not None
    assert state.nuh is not None
    assert state.tke is not None
    assert state.eps is not None
    assert np.all(np.isfinite(state.num))
    assert np.all(np.isfinite(state.nuh))
    assert np.all(np.isfinite(state.tke))
    assert np.all(np.isfinite(state.eps))
    assert np.any(np.abs(state.num[1:_NLEV] - 1.0e-6) > 0.0)


def test_do_turbulence_second_order_runs_translated_dispatch() -> None:
    state = _make_state(
        turb_method=second_order,
        scnd_method=weak_Eq_Kb_Eq,
        kb_method=1,
        epsb_method=1,
        scnd_coeff=CHCD01A,
    )
    h = np.ones(_NLEV + 1, dtype=np.float64)
    NN = np.zeros(_NLEV + 1, dtype=np.float64)
    SS = np.full(_NLEV + 1, 1.0e-4, dtype=np.float64)

    do_turbulence(
        state,
        _NLEV,
        dt=60.0,
        depth=25.0,
        u_taus=0.01,
        u_taub=0.01,
        z0s=0.02,
        z0b=0.001,
        h=h,
        NN=NN,
        SS=SS,
    )

    assert state.kb is not None
    assert state.epsb is not None
    assert state.num is not None
    assert np.all(np.isfinite(state.kb))
    assert np.all(np.isfinite(state.epsb))
    assert np.all(np.isfinite(state.num))


def test_do_turbulence_h15_requires_stokes_inputs() -> None:
    state = _make_state(
        turb_method=second_order,
        scnd_method=quasi_Eq_H15,
        kb_method=1,
        epsb_method=1,
        scnd_coeff=CHCD01A,
    )
    h, NN, SS = _make_inputs()

    with pytest.raises(ValueError, match="requires SSCSTK and SSSTK"):
        do_turbulence(
            state,
            _NLEV,
            dt=60.0,
            depth=25.0,
            u_taus=0.01,
            u_taub=0.01,
            z0s=0.02,
            z0b=0.001,
            h=h,
            NN=NN,
            SS=SS,
        )


def test_do_turbulence_rejects_invalid_model_code() -> None:
    state = TurbulenceState()
    init_turbulence(state, turb_method=99)
    post_init_turbulence(state, _NLEV)
    h, NN, SS = _make_inputs()
    with pytest.raises(ValueError, match="invalid turb_method=99"):
        do_turbulence(
            state,
            _NLEV,
            dt=3600.0,
            depth=25.0,
            u_taus=0.01,
            u_taub=0.01,
            z0s=0.02,
            z0b=0.001,
            h=h,
            NN=NN,
            SS=SS,
        )


def test_clean_turbulence_sets_arrays_none() -> None:
    state = _make_state()
    clean_turbulence(state)
    for name in _ARRAY_FIELD_NAMES:
        assert getattr(state, name) is None, f"{name} should be None after clean"


def test_clean_turbulence_preserves_scalars() -> None:
    state = _make_state()
    state.kappa = 0.41
    clean_turbulence(state)
    assert state.kappa == pytest.approx(0.41)


def test_reinitialise_with_different_nlev() -> None:
    state = TurbulenceState()
    init_turbulence(state)
    post_init_turbulence(state, 4)
    assert state.tke is not None and state.tke.shape == (5,)

    post_init_turbulence(state, 20)
    assert state.tke is not None and state.tke.shape == (21,)
