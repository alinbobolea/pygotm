import numpy as np
import pytest
import taichi as ti

from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.util.adv_center import (
    CENTRAL,
    CONSERVATIVE,
    FLUX,
    MUSCL,
    NON_CONSERVATIVE,
    ONE_SIDED,
    P1,
    P2,
    P2_PDM,
    SPLMAX13,
    SUPERBEE,
    UPSTREAM,
    VALUE,
    ZERO_DIVERGENCE,
    adv_center,
    adv_center_column,
    init_adv_center,
)


@ti_kernel
def adv_center_kernel(  # type: ignore[no-untyped-def]
    nlev: ti.i32,
    dt: ti.f64,
    h: TemplateArg,
    ho: TemplateArg,
    ww: TemplateArg,
    bc_up: ti.i32,
    bc_down: ti.i32,
    y_up: ti.f64,
    y_down: ti.f64,
    method: ti.i32,
    mode: ti.i32,
    y: TemplateArg,
    cu: TemplateArg,
):
    adv_center(
        nlev,
        dt,
        h,
        ho,
        ww,
        bc_up,
        bc_down,
        y_up,
        y_down,
        method,
        mode,
        y,
        cu,
    )


@ti_kernel
def adv_center_multi_kernel(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    h: TemplateArg,
    ho: TemplateArg,
    ww: TemplateArg,
    bc_up: ti.i32,
    bc_down: ti.i32,
    y_up: ti.f64,
    y_down: ti.f64,
    method: ti.i32,
    mode: ti.i32,
    y: TemplateArg,
    cu: TemplateArg,
):
    for col in range(n_cols):
        adv_center_column(
            col,
            nlev,
            dt,
            h,
            ho,
            ww,
            bc_up,
            bc_down,
            y_up,
            y_down,
            method,
            mode,
            y,
            cu,
        )


def _field(values: np.ndarray) -> ti.Field:
    field = ti.field(dtype=ti.f64, shape=values.shape)
    for index in np.ndindex(values.shape):
        field[index] = values[index]
    return field


def _adv_reconstruct_reference(
    scheme: int,
    cfl: float,
    fuu: float,
    fu: float,
    fd: float,
) -> float:
    deltaf = fd - fu
    deltafu = fu - fuu

    if deltaf * deltafu > 0.0:
        ratio = deltafu / deltaf
        if scheme == SUPERBEE:
            limiter = max(min(2.0 * ratio, 1.0), min(ratio, 2.0))
        elif scheme == P2_PDM:
            x = (1.0 / 6.0) * (1.0 - 2.0 * cfl)
            limiter = (0.5 + x) + (0.5 - x) * ratio
            limiter = min(2.0 * ratio / (cfl + 1.0e-10), limiter, 2.0 / (1.0 - cfl))
        elif scheme == SPLMAX13:
            limiter = min(
                2.0 * ratio,
                (1.0 / 3.0) * max(1.0 + 2.0 * ratio, 2.0 + ratio),
                2.0,
            )
        elif scheme == MUSCL:
            limiter = min(2.0 * ratio, 0.5 * (1.0 + ratio), 2.0)
        elif scheme == P2:
            x = (1.0 / 6.0) * (1.0 - 2.0 * cfl)
            limiter = (0.5 + x) + (0.5 - x) * ratio
        elif scheme == CENTRAL:
            limiter = 1.0 / (1.0 - cfl)
        else:
            limiter = 0.0
        return fu + 0.5 * limiter * (1.0 - cfl) * deltaf

    if scheme == P2:
        x = (1.0 / 6.0) * (1.0 - 2.0 * cfl)
        return fu + 0.5 * (1.0 - cfl) * ((0.5 + x) * deltaf + (0.5 - x) * deltafu)
    if scheme == CENTRAL:
        return 0.5 * (fu + fd)
    return fu


def _adv_center_reference(
    nlev: int,
    dt: float,
    h: np.ndarray,
    ww: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    method: int,
    mode: int,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    updated = y.copy()
    cu = np.zeros(nlev + 1, dtype=np.float64)
    cmax = 0.0
    for k in range(1, nlev):
        courant = abs(ww[k]) * dt / (0.5 * (h[k] + h[k + 1]))
        if courant > cmax:
            cmax = courant
    iterations = min(100, int(cmax) + 1)

    for _ in range(iterations):
        for k in range(1, nlev):
            if ww[k] > 0.0:
                courant = ww[k] / iterations * dt / (0.5 * (h[k] + h[k + 1]))
                y_upstream = updated[k - 1] if k > 1 else updated[k]
                y_central = updated[k]
                y_downstream = updated[k + 1]
            else:
                courant = -ww[k] / iterations * dt / (0.5 * (h[k] + h[k + 1]))
                y_upstream = updated[k + 2] if k < nlev - 1 else updated[k + 1]
                y_central = updated[k + 1]
                y_downstream = updated[k]

            reconstructed = _adv_reconstruct_reference(
                method,
                courant,
                y_upstream,
                y_central,
                y_downstream,
            )
            cu[k] = ww[k] * reconstructed

        if bc_up == FLUX:
            cu[nlev] = -y_up
        elif bc_up == VALUE:
            cu[nlev] = ww[nlev] * y_up
        elif bc_up == ONE_SIDED:
            cu[nlev] = ww[nlev] * updated[nlev] if ww[nlev] >= 0.0 else 0.0
        else:
            cu[nlev] = cu[nlev - 1]

        if bc_down == FLUX:
            cu[0] = y_down
        elif bc_down == VALUE:
            cu[0] = ww[0] * y_down
        elif bc_down == ONE_SIDED:
            cu[0] = ww[0] * updated[1] if ww[0] <= 0.0 else 0.0
        else:
            cu[0] = cu[1]

        if mode == NON_CONSERVATIVE:
            for k in range(1, nlev + 1):
                updated[k] = updated[k] - dt / iterations * (
                    (cu[k] - cu[k - 1]) / h[k] - updated[k] * (ww[k] - ww[k - 1]) / h[k]
                )
        else:
            for k in range(1, nlev + 1):
                updated[k] = updated[k] - dt / iterations * ((cu[k] - cu[k - 1]) / h[k])

    return updated, cu


def test_workspace_allocates_flux_field_and_releases_it() -> None:
    workspace = init_adv_center(6, n_cols=2)

    assert workspace.cu.shape == (2, 7)
    assert workspace.names() == ("cu",)

    workspace.clear()

    assert workspace.names() == ()
    assert not hasattr(workspace, "cu")


@pytest.mark.parametrize(
    ("method", "fuu", "fu", "fd", "cfl"),
    [
        (UPSTREAM, 0.3, 0.6, 1.2, 0.4),
        (P1, 0.3, 0.6, 1.2, 0.4),
        (P2, 0.3, 0.6, 1.2, 0.4),
        (SUPERBEE, 0.3, 0.6, 1.2, 0.4),
        (MUSCL, 0.3, 0.6, 1.2, 0.4),
        (P2_PDM, 0.3, 0.6, 1.2, 0.4),
        (SPLMAX13, 0.3, 0.6, 1.2, 0.4),
        (CENTRAL, 0.3, 0.6, 1.2, 0.4),
        (P2, 1.0, 1.0, 1.0, 0.2),
        (CENTRAL, 1.0, 1.0, 1.0, 0.2),
    ],
)
def test_adv_center_matches_reference_across_limiters(
    method: int,
    fuu: float,
    fu: float,
    fd: float,
    cfl: float,
) -> None:
    nlev = 2
    dt = 0.2
    h = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    ho = h.copy()
    speed = np.array([0.0, cfl * 0.5 * (h[1] + h[2]) / dt, 0.0], dtype=np.float64)
    y = np.array([0.0, fu, fd], dtype=np.float64)
    y[0] = fuu

    workspace = init_adv_center(nlev)
    y_field = _field(y)
    adv_center_kernel(
        nlev,
        dt,
        _field(h),
        _field(ho),
        _field(speed),
        ZERO_DIVERGENCE,
        VALUE,
        0.0,
        fuu,
        method,
        CONSERVATIVE,
        y_field,
        workspace.cu,
    )

    expected, _ = _adv_center_reference(
        nlev,
        dt,
        h,
        speed,
        ZERO_DIVERGENCE,
        VALUE,
        0.0,
        fuu,
        method,
        CONSERVATIVE,
        y,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    assert np.allclose(result[1:], expected[1:], rtol=1e-12, atol=1e-12)


def test_adv_center_conservative_split_iterations_match_reference() -> None:
    nlev = 4
    dt = 2.5
    h = np.array([0.0, 1.0, 0.8, 1.1, 0.9], dtype=np.float64)
    ho = np.array([0.0, 1.2, 1.0, 1.0, 0.7], dtype=np.float64)
    ww = np.array([0.0, 1.2, -1.4, 0.8, 0.2], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.4, 1.6, 0.8], dtype=np.float64)
    y_up = 0.15
    y_down = -0.1

    workspace = init_adv_center(nlev)
    y_field = _field(y)
    adv_center_kernel(
        nlev,
        dt,
        _field(h),
        _field(ho),
        _field(ww),
        FLUX,
        ONE_SIDED,
        y_up,
        y_down,
        P2_PDM,
        CONSERVATIVE,
        y_field,
        workspace.cu,
    )

    expected, expected_flux = _adv_center_reference(
        nlev,
        dt,
        h,
        ww,
        FLUX,
        ONE_SIDED,
        y_up,
        y_down,
        P2_PDM,
        CONSERVATIVE,
        y,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    fluxes = np.array([workspace.cu[i] for i in range(nlev + 1)])
    assert np.allclose(result[1:], expected[1:], rtol=1e-12, atol=1e-12)
    assert np.allclose(fluxes, expected_flux, rtol=1e-12, atol=1e-12)


def test_adv_center_nonconservative_matches_reference() -> None:
    nlev = 4
    dt = 1.25
    h = np.array([0.0, 1.0, 1.1, 0.9, 1.2], dtype=np.float64)
    ho = np.array([0.0, 0.9, 1.0, 1.2, 1.1], dtype=np.float64)
    ww = np.array([-0.2, -0.4, 0.5, -0.3, 0.1], dtype=np.float64)
    y = np.array([0.0, 0.6, 0.9, 0.2, 0.7], dtype=np.float64)
    y_up = 1.1
    y_down = -0.3

    workspace = init_adv_center(nlev)
    y_field = _field(y)
    adv_center_kernel(
        nlev,
        dt,
        _field(h),
        _field(ho),
        _field(ww),
        VALUE,
        ZERO_DIVERGENCE,
        y_up,
        y_down,
        MUSCL,
        NON_CONSERVATIVE,
        y_field,
        workspace.cu,
    )

    expected, _ = _adv_center_reference(
        nlev,
        dt,
        h,
        ww,
        VALUE,
        ZERO_DIVERGENCE,
        y_up,
        y_down,
        MUSCL,
        NON_CONSERVATIVE,
        y,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    assert np.allclose(result[1:], expected[1:], rtol=1e-12, atol=1e-12)
    assert np.isfinite(result[1:]).all()


def test_adv_center_multicolumn_ncols_one_matches_single_column() -> None:
    nlev = 4
    dt = 1.0
    h = np.array([0.0, 1.0, 0.9, 1.1, 1.0], dtype=np.float64)
    ho = np.array([0.0, 1.0, 0.9, 1.1, 1.0], dtype=np.float64)
    ww = np.array([0.0, 0.4, -0.2, 0.3, -0.1], dtype=np.float64)
    y = np.array([0.0, 1.2, 0.8, 0.5, 0.1], dtype=np.float64)
    y_up = 0.2
    y_down = -0.05

    single_workspace = init_adv_center(nlev)
    single_y = _field(y)
    multi_workspace = init_adv_center(nlev, n_cols=1)
    multi_y = _field(np.expand_dims(y, axis=0))
    h_multi = _field(np.expand_dims(h, axis=0))
    ho_multi = _field(np.expand_dims(ho, axis=0))
    ww_multi = _field(np.expand_dims(ww, axis=0))

    adv_center_kernel(
        nlev,
        dt,
        _field(h),
        _field(ho),
        _field(ww),
        FLUX,
        VALUE,
        y_up,
        y_down,
        UPSTREAM,
        CONSERVATIVE,
        single_y,
        single_workspace.cu,
    )
    adv_center_multi_kernel(
        1,
        nlev,
        dt,
        h_multi,
        ho_multi,
        ww_multi,
        FLUX,
        VALUE,
        y_up,
        y_down,
        UPSTREAM,
        CONSERVATIVE,
        multi_y,
        multi_workspace.cu,
    )

    single_result = np.array([single_y[i] for i in range(nlev + 1)])
    multi_result = np.array([multi_y[0, i] for i in range(nlev + 1)])
    assert np.allclose(multi_result, single_result, rtol=1e-12, atol=1e-12)


def test_adv_center_no_nan_inf() -> None:
    nlev = 4
    dt = 1.0
    h = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    ho = h.copy()
    ww = np.array([0.0, 0.3, -0.2, 0.4, -0.1], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.5, 0.8, 0.2], dtype=np.float64)

    workspace = init_adv_center(nlev)
    y_field = _field(y)
    adv_center_kernel(
        nlev,
        dt,
        _field(h),
        _field(ho),
        _field(ww),
        FLUX,
        FLUX,
        0.0,
        0.0,
        UPSTREAM,
        CONSERVATIVE,
        y_field,
        workspace.cu,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    assert np.isfinite(result[1:]).all(), "adv_center produced NaN or Inf"
