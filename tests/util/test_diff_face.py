import numpy as np
import pytest
import taichi as ti

from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.util.diff_face import DIRICHLET, NEUMANN, diff_face, diff_face_column
from pygotm.util.tridiagonal import init_tridiagonal


@ti_kernel
def diff_face_kernel(  # type: ignore[no-untyped-def]
    nlev: ti.i32,
    dt: ti.f64,
    cnpar: ti.f64,
    bc_up: ti.i32,
    bc_down: ti.i32,
    y_up: ti.f64,
    y_down: ti.f64,
    h: TemplateArg,
    nu_y: TemplateArg,
    l_sour: TemplateArg,
    q_sour: TemplateArg,
    y: TemplateArg,
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
):
    diff_face(
        nlev,
        dt,
        cnpar,
        h,
        bc_up,
        bc_down,
        y_up,
        y_down,
        nu_y,
        l_sour,
        q_sour,
        y,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )


@ti_kernel
def diff_face_multi_kernel(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    cnpar: ti.f64,
    bc_up: ti.i32,
    bc_down: ti.i32,
    y_up: ti.f64,
    y_down: ti.f64,
    h: TemplateArg,
    nu_y: TemplateArg,
    l_sour: TemplateArg,
    q_sour: TemplateArg,
    y: TemplateArg,
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
):
    for col in range(n_cols):
        diff_face_column(
            col,
            nlev,
            dt,
            cnpar,
            h,
            bc_up,
            bc_down,
            y_up,
            y_down,
            nu_y,
            l_sour,
            q_sour,
            y,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )


def _field(values: np.ndarray) -> ti.Field:
    field = ti.field(dtype=ti.f64, shape=values.shape)
    for index in np.ndindex(values.shape):
        field[index] = values[index]
    return field


def _reference_diff_face(
    nlev: int,
    dt: float,
    cnpar: float,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    h: np.ndarray,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    local_nu_y = nu_y.copy()
    updated = y.copy()
    au = np.zeros(nlev + 1, dtype=np.float64)
    bu = np.zeros(nlev + 1, dtype=np.float64)
    cu = np.zeros(nlev + 1, dtype=np.float64)
    du = np.zeros(nlev + 1, dtype=np.float64)

    if nlev == 2:
        local_nu_y[0] = local_nu_y[1]
        local_nu_y[nlev] = local_nu_y[1]
        updated[0] = updated[1]
        updated[nlev] = updated[1]

    for i in range(2, nlev - 1):
        c = dt * (local_nu_y[i + 1] + local_nu_y[i]) / (h[i] + h[i + 1]) / h[i + 1]
        a = dt * (local_nu_y[i] + local_nu_y[i - 1]) / (h[i] + h[i + 1]) / h[i]
        linear_source = dt * l_sour[i]

        cu[i] = -cnpar * c
        au[i] = -cnpar * a
        bu[i] = 1.0 + cnpar * (a + c) - linear_source
        du[i] = (1.0 - (1.0 - cnpar) * (a + c)) * updated[i]
        du[i] += (1.0 - cnpar) * (a * updated[i - 1] + c * updated[i + 1])
        du[i] += dt * q_sour[i]

    if bc_up == NEUMANN:
        a = dt * (local_nu_y[nlev - 1] + local_nu_y[nlev - 2])
        a /= (h[nlev - 1] + h[nlev]) * h[nlev - 1]
        linear_source = dt * l_sour[nlev - 1]
        au[nlev - 1] = -cnpar * a
        bu[nlev - 1] = 1.0 + cnpar * a - linear_source
        du[nlev - 1] = (1.0 - (1.0 - cnpar) * a) * updated[nlev - 1]
        du[nlev - 1] += (1.0 - cnpar) * a * updated[nlev - 2]
        du[nlev - 1] += dt * q_sour[nlev - 1]
        du[nlev - 1] += 2.0 * dt * y_up / (h[nlev - 1] + h[nlev])
    else:
        au[nlev - 1] = 0.0
        bu[nlev - 1] = 1.0
        du[nlev - 1] = y_up

    if bc_down == NEUMANN:
        c = dt * (local_nu_y[2] + local_nu_y[1]) / (h[1] + h[2]) / h[2]
        linear_source = dt * l_sour[1]
        cu[1] = -cnpar * c
        bu[1] = 1.0 + cnpar * c - linear_source
        du[1] = (1.0 - (1.0 - cnpar) * c) * updated[1]
        du[1] += (1.0 - cnpar) * c * updated[2]
        du[1] += dt * q_sour[1]
        du[1] += 2.0 * dt * y_down / (h[1] + h[2])
    else:
        bu[1] = 1.0
        cu[1] = 0.0
        du[1] = y_down

    lower = np.array([au[i] for i in range(1, nlev)], dtype=np.float64)
    diagonal = np.array([bu[i] for i in range(1, nlev)], dtype=np.float64)
    upper = np.array([cu[i] for i in range(1, nlev)], dtype=np.float64)
    rhs = np.array([du[i] for i in range(1, nlev)], dtype=np.float64)

    matrix = np.diag(diagonal)
    if nlev > 2:
        matrix += np.diag(lower[1:], k=-1)
        matrix += np.diag(upper[:-1], k=1)

    updated[1:nlev] = np.linalg.solve(matrix, rhs)
    return updated


def test_diff_face_dirichlet_matches_numpy_reference() -> None:
    nlev = 5
    dt = 20.0
    cnpar = 0.6
    h = np.array([0.0, 1.0, 1.1, 0.9, 1.2, 1.0], dtype=np.float64)
    nu_y = np.array([0.0, 0.05, 0.07, 0.08, 0.06, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, 0.02, -0.01, 0.03, 0.01, 0.0], dtype=np.float64)
    q_sour = np.array([0.0, 0.002, -0.003, 0.004, -0.001, 0.0], dtype=np.float64)
    y = np.array([0.0, 0.9, 0.7, 0.4, 0.2, 0.0], dtype=np.float64)
    y_up = 0.3
    y_down = 1.1

    workspace = init_tridiagonal(nlev)
    y_field = _field(y)
    diff_face_kernel(
        nlev,
        dt,
        cnpar,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        _field(h),
        _field(nu_y),
        _field(l_sour),
        _field(q_sour),
        y_field,
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
    )

    expected = _reference_diff_face(
        nlev,
        dt,
        cnpar,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        h,
        nu_y,
        l_sour,
        q_sour,
        y,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    assert np.allclose(result[1:nlev], expected[1:nlev], rtol=1e-12, atol=1e-12)


def test_diff_face_two_layer_bugfix_matches_reference() -> None:
    nlev = 2
    dt = 12.0
    cnpar = 1.0
    h = np.array([0.0, 0.8, 1.1], dtype=np.float64)
    nu_y = np.array([0.0, 0.2, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, 0.01, 0.0], dtype=np.float64)
    q_sour = np.array([0.0, -0.002, 0.0], dtype=np.float64)
    y = np.array([0.0, 0.5, 0.0], dtype=np.float64)
    y_up = 0.04
    y_down = -0.03

    workspace = init_tridiagonal(nlev)
    nu_field = _field(nu_y)
    y_field = _field(y)
    diff_face_kernel(
        nlev,
        dt,
        cnpar,
        NEUMANN,
        NEUMANN,
        y_up,
        y_down,
        _field(h),
        nu_field,
        _field(l_sour),
        _field(q_sour),
        y_field,
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
    )

    expected = _reference_diff_face(
        nlev,
        dt,
        cnpar,
        NEUMANN,
        NEUMANN,
        y_up,
        y_down,
        h,
        nu_y,
        l_sour,
        q_sour,
        y,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    mutated_nu = np.array([nu_field[i] for i in range(nlev + 1)])
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert mutated_nu[0] == pytest.approx(mutated_nu[1])
    assert mutated_nu[nlev] == pytest.approx(mutated_nu[1])


def test_diff_face_multicolumn_ncols_one_matches_single_column() -> None:
    nlev = 4
    dt = 18.0
    cnpar = 0.5
    h = np.array([0.0, 1.0, 1.0, 1.2, 0.9], dtype=np.float64)
    nu_y = np.array([0.0, 0.06, 0.09, 0.07, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, -0.01, 0.02, -0.03, 0.0], dtype=np.float64)
    q_sour = np.array([0.0, 0.003, 0.001, -0.002, 0.0], dtype=np.float64)
    y = np.array([0.0, -0.2, 0.4, 0.8, 0.0], dtype=np.float64)
    y_up = 0.15
    y_down = -0.05

    single_workspace = init_tridiagonal(nlev)
    single_y = _field(y)
    multi_workspace = init_tridiagonal(nlev, n_cols=1)
    multi_y = _field(np.expand_dims(y, axis=0))

    diff_face_kernel(
        nlev,
        dt,
        cnpar,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        _field(h),
        _field(nu_y),
        _field(l_sour),
        _field(q_sour),
        single_y,
        single_workspace.au,
        single_workspace.bu,
        single_workspace.cu,
        single_workspace.du,
        single_workspace.ru,
        single_workspace.qu,
    )
    diff_face_multi_kernel(
        1,
        nlev,
        dt,
        cnpar,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        _field(np.expand_dims(h, axis=0)),
        _field(np.expand_dims(nu_y, axis=0)),
        _field(np.expand_dims(l_sour, axis=0)),
        _field(np.expand_dims(q_sour, axis=0)),
        multi_y,
        multi_workspace.au,
        multi_workspace.bu,
        multi_workspace.cu,
        multi_workspace.du,
        multi_workspace.ru,
        multi_workspace.qu,
    )

    single_result = np.array([single_y[i] for i in range(nlev + 1)])
    multi_result = np.array([multi_y[0, i] for i in range(nlev + 1)])
    assert np.allclose(multi_result, single_result, rtol=1e-12, atol=1e-12)


def test_diff_face_no_nan_inf() -> None:
    nlev = 5
    dt = 30.0
    cnpar = 0.6
    h = np.array([0.0, 1.0, 1.1, 0.9, 1.2, 1.0], dtype=np.float64)
    nu_y = np.array([0.0, 0.05, 0.07, 0.06, 0.08, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, 0.01, -0.02, 0.03, -0.01, 0.0], dtype=np.float64)
    q_sour = np.array([0.0, 0.002, -0.001, 0.003, -0.002, 0.0], dtype=np.float64)
    y = np.array([0.0, -0.3, 0.5, 1.2, 0.7, 0.0], dtype=np.float64)
    y_up = 0.2
    y_down = -0.1

    workspace = init_tridiagonal(nlev)
    y_field = _field(y)
    diff_face_kernel(
        nlev,
        dt,
        cnpar,
        NEUMANN,
        NEUMANN,
        y_up,
        y_down,
        _field(h),
        _field(nu_y),
        _field(l_sour),
        _field(q_sour),
        y_field,
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    assert np.isfinite(result[1:nlev]).all(), "diff_face produced NaN or Inf"
