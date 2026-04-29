"""Tests for pygotm.turbulence.r_ratio."""

from __future__ import annotations

import numpy as np

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

    workspace = RRatioWorkspace(_NLEV, n_cols=1)
    workspace.tke[0] = tke
    workspace.eps[0] = eps
    workspace.kb[0] = kb
    workspace.epsb[0] = epsb

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
    result = workspace.r[0]

    np.testing.assert_allclose(result, expected, rtol=1.0e-12)
    assert result[0] == expected[0]
    assert result[_NLEV] == expected[_NLEV]


def test_multicolumn_parity_for_identical_columns() -> None:
    tke = np.linspace(1.5e-4, 5.5e-4, _NLEV + 1)
    eps = np.linspace(2.0e-6, 1.0e-5, _NLEV + 1)
    kb = np.linspace(5.0e-6, 4.5e-5, _NLEV + 1)
    epsb = np.linspace(1.0e-6, 9.0e-6, _NLEV + 1)

    single = RRatioWorkspace(_NLEV, n_cols=1)
    single.tke[0] = tke
    single.eps[0] = eps
    single.kb[0] = kb
    single.epsb[0] = epsb
    step_r_ratio(1, _NLEV, single.tke, single.eps, single.kb, single.epsb, single.r)

    multi = RRatioWorkspace(_NLEV, n_cols=2)
    for col in range(2):
        multi.tke[col] = tke
        multi.eps[col] = eps
        multi.kb[col] = kb
        multi.epsb[col] = epsb
    step_r_ratio(2, _NLEV, multi.tke, multi.eps, multi.kb, multi.epsb, multi.r)

    expected = single.r[0]
    for col in range(2):
        np.testing.assert_allclose(multi.r[col], expected)


def test_no_nan_or_inf_for_valid_inputs() -> None:
    workspace = RRatioWorkspace(_NLEV, n_cols=1)
    workspace.tke[0] = np.full(_NLEV + 1, 3.0e-4)
    workspace.eps[0] = np.full(_NLEV + 1, 6.0e-6)
    workspace.kb[0] = np.full(_NLEV + 1, 2.0e-5)
    workspace.epsb[0] = np.full(_NLEV + 1, 4.0e-6)

    step_r_ratio(
        1,
        _NLEV,
        workspace.tke,
        workspace.eps,
        workspace.kb,
        workspace.epsb,
        workspace.r,
    )

    result = workspace.r[0]
    assert np.isfinite(result).all()
