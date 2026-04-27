"""Tests for pygotm.turbulence.internal_wave."""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, read_field_array

from pygotm.turbulence.internal_wave import InternalWaveWorkspace, step_internal_wave
from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 6


def _make_state(nlev: int = _NLEV, *, iw_model: int = 2) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, iw_model=iw_model)
    post_init_turbulence(state, nlev)
    return state


def _reference_internal_wave(
    *,
    iw_model: int,
    klimiw: float,
    rich_cr: float,
    numiw: float,
    nuhiw: float,
    numshear: float,
    tke: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_num = num.copy()
    out_nuh = nuh.copy()

    if iw_model == 2:
        rich2 = rich_cr * rich_cr
        for i in range(1, tke.size - 1):
            if tke[i] <= klimiw:
                rich = NN[i] / (SS[i] + 1.0e-10)
                if rich < rich_cr:
                    if rich > 0.0:
                        pot = 1.0 - rich * rich / rich2
                        x = numshear * pot**3
                        out_num[i] = numiw + x
                        out_nuh[i] = nuhiw + x
                    else:
                        out_num[i] = numiw + numshear
                        out_nuh[i] = nuhiw + numshear
                else:
                    out_num[i] = numiw
                    out_nuh[i] = nuhiw

    return out_num, out_nuh


def _run_step(
    state: TurbulenceState,
    *,
    tke: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    n_cols: int = 1,
) -> InternalWaveWorkspace:
    workspace = InternalWaveWorkspace(_NLEV, n_cols=n_cols)
    for col in range(n_cols):
        fill_field_from_array(workspace.tke, tke, col=col)
        fill_field_from_array(workspace.num, num, col=col)
        fill_field_from_array(workspace.nuh, nuh, col=col)
        fill_field_from_array(workspace.NN, NN, col=col)
        fill_field_from_array(workspace.SS, SS, col=col)

    step_internal_wave(
        n_cols,
        _NLEV,
        state.iw_model,
        state.klimiw,
        state.rich_cr,
        state.numiw,
        state.nuhiw,
        state.numshear,
        workspace.tke,
        workspace.num,
        workspace.nuh,
        workspace.NN,
        workspace.SS,
    )
    return workspace


def test_import() -> None:
    assert callable(step_internal_wave)


def test_workspace_instantiates() -> None:
    workspace = InternalWaveWorkspace(_NLEV, n_cols=2)
    assert workspace.num.shape == (2, _NLEV + 1)


def test_matches_fortran_reference_across_all_branches() -> None:
    state = _make_state()
    tke = np.array([4.0e-6, 5.0e-7, 5.0e-7, 5.0e-7, 2.0e-6, 5.0e-7, 4.0e-6])
    num = np.linspace(2.0e-4, 8.0e-4, _NLEV + 1)
    nuh = np.linspace(3.0e-4, 9.0e-4, _NLEV + 1)
    NN = np.array([0.0, -1.0e-4, 1.0e-4, 1.0, 1.0e-4, 2.0e-4, 0.0])
    SS = np.array([0.0, 4.0e-4, 5.0e-4, 1.0, 5.0e-4, 1.0e-5, 0.0])

    workspace = _run_step(state, tke=tke, num=num, nuh=nuh, NN=NN, SS=SS)
    expected_num, expected_nuh = _reference_internal_wave(
        iw_model=state.iw_model,
        klimiw=state.klimiw,
        rich_cr=state.rich_cr,
        numiw=state.numiw,
        nuhiw=state.nuhiw,
        numshear=state.numshear,
        tke=tke,
        num=num,
        nuh=nuh,
        NN=NN,
        SS=SS,
    )

    np.testing.assert_allclose(
        read_field_array(workspace.num), expected_num, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        read_field_array(workspace.nuh), expected_nuh, rtol=1.0e-12
    )
    assert read_field_array(workspace.num)[0] == num[0]
    assert read_field_array(workspace.nuh)[_NLEV] == nuh[_NLEV]


def test_non_matching_iw_model_leaves_fields_unchanged() -> None:
    state = _make_state(iw_model=1)
    tke = np.full(_NLEV + 1, 5.0e-7)
    num = np.linspace(1.0e-4, 7.0e-4, _NLEV + 1)
    nuh = np.linspace(2.0e-4, 8.0e-4, _NLEV + 1)
    NN = np.linspace(-2.0e-4, 2.0e-4, _NLEV + 1)
    SS = np.linspace(1.0e-4, 7.0e-4, _NLEV + 1)

    workspace = _run_step(state, tke=tke, num=num, nuh=nuh, NN=NN, SS=SS)

    np.testing.assert_allclose(read_field_array(workspace.num), num)
    np.testing.assert_allclose(read_field_array(workspace.nuh), nuh)


def test_multicolumn_parity_for_identical_columns() -> None:
    state = _make_state()
    tke = np.array([4.0e-6, 5.0e-7, 5.0e-7, 5.0e-7, 2.0e-6, 5.0e-7, 4.0e-6])
    num = np.linspace(2.0e-4, 8.0e-4, _NLEV + 1)
    nuh = np.linspace(3.0e-4, 9.0e-4, _NLEV + 1)
    NN = np.array([0.0, -1.0e-4, 1.0e-4, 1.0, 1.0e-4, 2.0e-4, 0.0])
    SS = np.array([0.0, 4.0e-4, 5.0e-4, 1.0, 5.0e-4, 1.0e-5, 0.0])

    single = _run_step(state, tke=tke, num=num, nuh=nuh, NN=NN, SS=SS)
    multi = _run_step(state, tke=tke, num=num, nuh=nuh, NN=NN, SS=SS, n_cols=2)

    for name in ("num", "nuh"):
        expected = read_field_array(getattr(single, name))
        for col in range(2):
            np.testing.assert_allclose(
                read_field_array(getattr(multi, name), col=col),
                expected,
            )


def test_no_nan_or_inf_for_valid_inputs() -> None:
    state = _make_state()
    tke = np.full(_NLEV + 1, 5.0e-7)
    num = np.full(_NLEV + 1, 2.0e-4)
    nuh = np.full(_NLEV + 1, 3.0e-4)
    NN = np.linspace(-3.0e-4, 3.0e-4, _NLEV + 1)
    SS = np.zeros(_NLEV + 1)

    workspace = _run_step(state, tke=tke, num=num, nuh=nuh, NN=NN, SS=SS)

    assert np.isfinite(read_field_array(workspace.num)).all()
    assert np.isfinite(read_field_array(workspace.nuh)).all()
