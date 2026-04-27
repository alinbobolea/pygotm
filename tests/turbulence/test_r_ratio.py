"""Tests for pygotm.turbulence.r_ratio."""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, read_field_array

from pygotm.turbulence.r_ratio import RRatioWorkspace, step_r_ratio

_NLEV = 8


def test_import() -> None:
    assert callable(step_r_ratio)


def test_workspace_instantiates() -> None:
    workspace = RRatioWorkspace(_NLEV, n_cols=2)
    assert workspace.r.shape == (2, _NLEV + 1)


def test_matches_direct_fortran_formula() -> None:
    tke = np.linspace(2.0e-4, 6.0e-4, _NLEV + 1)
    eps = np.linspace(1.0e-6, 9.0e-6, _NLEV + 1)
    kb = np.linspace(1.0e-5, 9.0e-5, _NLEV + 1)
    epsb = np.linspace(2.0e-6, 1.0e-5, _NLEV + 1)

    workspace = RRatioWorkspace(_NLEV)
    fill_field_from_array(workspace.tke, tke)
    fill_field_from_array(workspace.eps, eps)
    fill_field_from_array(workspace.kb, kb)
    fill_field_from_array(workspace.epsb, epsb)

    step_r_ratio(
        1,
        _NLEV,
        workspace.tke,
        workspace.eps,
        workspace.kb,
        workspace.epsb,
        workspace.r,
    )

    expected = kb * eps / (epsb * tke)
    result = read_field_array(workspace.r)

    np.testing.assert_allclose(result, expected, rtol=1.0e-12)
    assert result[0] == expected[0]
    assert result[_NLEV] == expected[_NLEV]


def test_multicolumn_parity_for_identical_columns() -> None:
    tke = np.linspace(1.5e-4, 5.5e-4, _NLEV + 1)
    eps = np.linspace(2.0e-6, 1.0e-5, _NLEV + 1)
    kb = np.linspace(5.0e-6, 4.5e-5, _NLEV + 1)
    epsb = np.linspace(1.0e-6, 9.0e-6, _NLEV + 1)

    single = RRatioWorkspace(_NLEV)
    fill_field_from_array(single.tke, tke)
    fill_field_from_array(single.eps, eps)
    fill_field_from_array(single.kb, kb)
    fill_field_from_array(single.epsb, epsb)
    step_r_ratio(1, _NLEV, single.tke, single.eps, single.kb, single.epsb, single.r)

    multi = RRatioWorkspace(_NLEV, n_cols=2)
    for col in range(2):
        fill_field_from_array(multi.tke, tke, col=col)
        fill_field_from_array(multi.eps, eps, col=col)
        fill_field_from_array(multi.kb, kb, col=col)
        fill_field_from_array(multi.epsb, epsb, col=col)
    step_r_ratio(2, _NLEV, multi.tke, multi.eps, multi.kb, multi.epsb, multi.r)

    expected = read_field_array(single.r)
    for col in range(2):
        np.testing.assert_allclose(read_field_array(multi.r, col=col), expected)


def test_no_nan_or_inf_for_valid_inputs() -> None:
    workspace = RRatioWorkspace(_NLEV)
    fill_field_from_array(workspace.tke, np.full(_NLEV + 1, 3.0e-4))
    fill_field_from_array(workspace.eps, np.full(_NLEV + 1, 6.0e-6))
    fill_field_from_array(workspace.kb, np.full(_NLEV + 1, 2.0e-5))
    fill_field_from_array(workspace.epsb, np.full(_NLEV + 1, 4.0e-6))

    step_r_ratio(
        1,
        _NLEV,
        workspace.tke,
        workspace.eps,
        workspace.kb,
        workspace.epsb,
        workspace.r,
    )

    result = read_field_array(workspace.r)
    assert np.isfinite(result).all()
