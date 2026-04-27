"""Tests for pygotm.turbulence.production."""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, read_field_array
from type_helpers import ReadyTurbulenceState, require_turbulence_state

from pygotm.turbulence.production import ProductionWorkspace, step_production
from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 12


def _zeros(nlev: int) -> np.ndarray:
    return np.zeros(nlev + 1, dtype=np.float64)


def _make_state(
    nlev: int = _NLEV,
    *,
    iw_model: int = 0,
    alpha: float = 0.0,
) -> ReadyTurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, iw_model=iw_model, alpha=alpha)
    post_init_turbulence(state, nlev)
    return require_turbulence_state(state)


def _run_step_production(
    state: ReadyTurbulenceState,
    NN: np.ndarray,
    SS: np.ndarray,
    *,
    xP: np.ndarray | None = None,
    SSCSTK: np.ndarray | None = None,
    SSSTK: np.ndarray | None = None,
    n_cols: int = 1,
    workspace: ProductionWorkspace | None = None,
) -> ProductionWorkspace:
    nlev = NN.size - 1
    ws = (
        workspace if workspace is not None else ProductionWorkspace(nlev, n_cols=n_cols)
    )

    for col in range(n_cols):
        fill_field_from_array(ws.NN, NN, col=col)
        fill_field_from_array(ws.SS, SS, col=col)
        fill_field_from_array(ws.xP, xP if xP is not None else _zeros(nlev), col=col)
        fill_field_from_array(
            ws.SSCSTK,
            SSCSTK if SSCSTK is not None else _zeros(nlev),
            col=col,
        )
        fill_field_from_array(
            ws.SSSTK,
            SSSTK if SSSTK is not None else _zeros(nlev),
            col=col,
        )
        fill_field_from_array(ws.num, state.num, col=col)
        fill_field_from_array(ws.nuh, state.nuh, col=col)
        fill_field_from_array(ws.nucl, state.nucl, col=col)
        fill_field_from_array(ws.P, state.P, col=col)
        fill_field_from_array(ws.B, state.B, col=col)
        fill_field_from_array(ws.Pb, state.Pb, col=col)
        fill_field_from_array(ws.Px, state.Px, col=col)
        fill_field_from_array(ws.PSTK, state.PSTK, col=col)

    step_production(
        n_cols,
        nlev,
        state.iw_model,
        state.alpha,
        int(xP is not None),
        int(SSCSTK is not None),
        int(SSSTK is not None),
        ws.NN,
        ws.SS,
        ws.xP,
        ws.SSCSTK,
        ws.SSSTK,
        ws.num,
        ws.nuh,
        ws.nucl,
        ws.P,
        ws.B,
        ws.Pb,
        ws.Px,
        ws.PSTK,
    )

    state.P[:] = read_field_array(ws.P)
    state.B[:] = read_field_array(ws.B)
    state.Pb[:] = read_field_array(ws.Pb)
    state.Px[:] = read_field_array(ws.Px)
    state.PSTK[:] = read_field_array(ws.PSTK)
    return ws


def test_import() -> None:
    from pygotm.turbulence.production import step_production as _step  # noqa: F401

    assert callable(_step)


def test_workspace_instantiates() -> None:
    workspace = ProductionWorkspace(_NLEV, n_cols=2)
    assert workspace.NN.shape == (2, _NLEV + 1)
    assert workspace.P.shape == (2, _NLEV + 1)


def test_smoke() -> None:
    state = _make_state()
    zeros = _zeros(_NLEV)
    _run_step_production(state, zeros, zeros)


def test_base_production_matches_analytic_formula_including_boundaries() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    assert state.num is not None
    assert state.nuh is not None

    state.num[:] = np.linspace(1.0e-4, 2.2e-4, nlev + 1)
    state.nuh[:] = np.linspace(3.0e-4, 4.2e-4, nlev + 1)

    NN = np.linspace(-2.0e-4, 3.0e-4, nlev + 1)
    SS = np.linspace(1.0e-5, 7.0e-5, nlev + 1)

    _run_step_production(state, NN, SS)

    np.testing.assert_allclose(state.P, state.num * SS, rtol=1e-12)
    np.testing.assert_allclose(state.B, -state.nuh * NN, rtol=1e-12)
    np.testing.assert_allclose(state.Pb, state.nuh * NN * NN, rtol=1e-12)


def test_internal_wave_alpha_term_is_applied_only_for_iw_model_one() -> None:
    nlev = _NLEV
    NN = np.linspace(1.0e-5, 2.2e-4, nlev + 1)
    SS = _zeros(nlev)

    inactive = _make_state(nlev, iw_model=0, alpha=0.75)
    active = _make_state(nlev, iw_model=1, alpha=0.75)
    assert inactive.num is not None
    assert active.num is not None
    inactive.num[:] = 2.5e-4
    active.num[:] = 2.5e-4

    _run_step_production(inactive, NN, SS)
    _run_step_production(active, NN, SS)

    expected = active.num * active.alpha * NN
    np.testing.assert_allclose(active.P - inactive.P, expected, rtol=1e-12)


def test_xP_updates_only_when_present() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    xP = np.linspace(0.0, 6.0e-6, nlev + 1)

    _run_step_production(state, _zeros(nlev), _zeros(nlev), xP=xP)
    np.testing.assert_allclose(state.Px, xP, rtol=1e-12)

    previous = state.Px.copy()
    _run_step_production(state, _zeros(nlev), _zeros(nlev))
    np.testing.assert_allclose(state.Px, previous, rtol=1e-12)


def test_stokes_cross_shear_updates_P_and_PSTK() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    assert state.num is not None
    assert state.nucl is not None

    state.num[:] = 2.0e-4
    state.nucl[:] = 8.0e-5

    NN = _zeros(nlev)
    SS = np.full(nlev + 1, 3.0e-5)
    SSCSTK = np.full(nlev + 1, 7.0e-5)

    _run_step_production(state, NN, SS, SSCSTK=SSCSTK)

    expected_P = state.num * SS + state.nucl * SSCSTK
    expected_PSTK = state.num * SSCSTK
    np.testing.assert_allclose(state.P, expected_P, rtol=1e-12)
    np.testing.assert_allclose(state.PSTK, expected_PSTK, rtol=1e-12)


def test_stokes_shear_only_resets_PSTK_before_adding_nucl_term() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    assert state.nucl is not None
    assert state.PSTK is not None

    state.nucl[:] = 5.0e-5
    state.PSTK[:] = 99.0

    SSSTK = np.full(nlev + 1, 4.0e-5)
    _run_step_production(state, _zeros(nlev), _zeros(nlev), SSSTK=SSSTK)

    np.testing.assert_allclose(state.P, 0.0, atol=1e-30)
    np.testing.assert_allclose(state.PSTK, state.nucl * SSSTK, rtol=1e-12)


def test_combined_stokes_terms_match_fortran_formula() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    assert state.num is not None
    assert state.nucl is not None

    state.num[:] = 1.5e-4
    state.nucl[:] = 4.0e-5

    SS = np.full(nlev + 1, 2.0e-5)
    SSCSTK = np.full(nlev + 1, 6.0e-5)
    SSSTK = np.full(nlev + 1, 9.0e-5)

    _run_step_production(state, _zeros(nlev), SS, SSCSTK=SSCSTK, SSSTK=SSSTK)

    expected_P = state.num * SS + state.nucl * SSCSTK
    expected_PSTK = state.num * SSCSTK + state.nucl * SSSTK
    np.testing.assert_allclose(state.P, expected_P, rtol=1e-12)
    np.testing.assert_allclose(state.PSTK, expected_PSTK, rtol=1e-12)


def test_absent_stokes_arguments_preserve_previous_PSTK() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    SSCSTK = np.full(nlev + 1, 5.0e-5)

    _run_step_production(state, _zeros(nlev), _zeros(nlev), SSCSTK=SSCSTK)
    previous = state.PSTK.copy()

    _run_step_production(state, _zeros(nlev), _zeros(nlev))
    np.testing.assert_allclose(state.PSTK, previous, rtol=1e-12)


def test_pb_nonnegative_with_nonnegative_diffusivity() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    NN = np.linspace(-4.0e-4, 4.0e-4, nlev + 1)
    SS = _zeros(nlev)

    _run_step_production(state, NN, SS)

    assert np.all(state.Pb >= 0.0)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    state = _make_state(nlev, iw_model=1, alpha=0.25)
    assert state.num is not None
    assert state.nuh is not None
    assert state.nucl is not None

    state.num[:] = np.linspace(1.0e-4, 2.0e-4, nlev + 1)
    state.nuh[:] = np.linspace(2.0e-4, 3.0e-4, nlev + 1)
    state.nucl[:] = np.linspace(3.0e-5, 5.0e-5, nlev + 1)

    NN = np.linspace(-1.0e-4, 2.0e-4, nlev + 1)
    SS = np.linspace(1.0e-5, 3.0e-5, nlev + 1)
    xP = np.linspace(0.0, 2.0e-6, nlev + 1)
    SSCSTK = np.linspace(0.0, 4.0e-5, nlev + 1)
    SSSTK = np.linspace(0.0, 6.0e-5, nlev + 1)

    single = _run_step_production(
        state,
        NN,
        SS,
        xP=xP,
        SSCSTK=SSCSTK,
        SSSTK=SSSTK,
        n_cols=1,
    )
    multi = _run_step_production(
        state,
        NN,
        SS,
        xP=xP,
        SSCSTK=SSCSTK,
        SSSTK=SSSTK,
        n_cols=2,
    )

    for name in ("P", "B", "Pb", "Px", "PSTK"):
        single_arr = read_field_array(getattr(single, name), col=0)
        multi_0 = read_field_array(getattr(multi, name), col=0)
        multi_1 = read_field_array(getattr(multi, name), col=1)
        np.testing.assert_allclose(multi_0, single_arr, rtol=1e-12)
        np.testing.assert_allclose(multi_1, single_arr, rtol=1e-12)


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev, iw_model=1, alpha=0.1)
    NN = np.linspace(-2.0e-4, 3.0e-4, nlev + 1)
    SS = np.linspace(1.0e-6, 5.0e-5, nlev + 1)
    xP = np.linspace(0.0, 1.0e-5, nlev + 1)
    SSCSTK = np.linspace(0.0, 3.0e-5, nlev + 1)
    SSSTK = np.linspace(0.0, 2.0e-5, nlev + 1)

    _run_step_production(state, NN, SS, xP=xP, SSCSTK=SSCSTK, SSSTK=SSSTK)

    for array in (state.P, state.B, state.Pb, state.Px, state.PSTK):
        assert array is not None
        assert np.isfinite(array).all()
