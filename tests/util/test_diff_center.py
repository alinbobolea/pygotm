import numpy as np
import taichi as ti
from taichi_helpers import fill_field_from_array, read_field_array

from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.util.diff_center import DIRICHLET, NEUMANN, diff_center, diff_center_column
from pygotm.util.tridiagonal import init_tridiagonal


@ti_kernel
def diff_center_kernel(  # type: ignore[no-untyped-def]
    nlev: ti.i32,
    dt: ti.f64,
    cnpar: ti.f64,
    posconc: ti.i32,
    bc_up: ti.i32,
    bc_down: ti.i32,
    y_up: ti.f64,
    y_down: ti.f64,
    h: TemplateArg,
    nu_y: TemplateArg,
    l_sour: TemplateArg,
    q_sour: TemplateArg,
    tau_r: TemplateArg,
    y_obs: TemplateArg,
    y: TemplateArg,
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
):
    diff_center(
        nlev,
        dt,
        cnpar,
        posconc,
        h,
        bc_up,
        bc_down,
        y_up,
        y_down,
        nu_y,
        l_sour,
        q_sour,
        tau_r,
        y_obs,
        y,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )


@ti_kernel
def diff_center_multi_kernel(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    cnpar: ti.f64,
    posconc: ti.i32,
    bc_up: ti.i32,
    bc_down: ti.i32,
    y_up: ti.f64,
    y_down: ti.f64,
    h: TemplateArg,
    nu_y: TemplateArg,
    l_sour: TemplateArg,
    q_sour: TemplateArg,
    tau_r: TemplateArg,
    y_obs: TemplateArg,
    y: TemplateArg,
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
):
    for col in range(n_cols):
        diff_center_column(
            col,
            nlev,
            dt,
            cnpar,
            posconc,
            h,
            bc_up,
            bc_down,
            y_up,
            y_down,
            nu_y,
            l_sour,
            q_sour,
            tau_r,
            y_obs,
            y,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )


def _build_reference_matrix(
    nlev: int,
    dt: float,
    cnpar: float,
    posconc: int,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    tau_r: np.ndarray,
    y_obs: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    au = np.zeros(nlev + 1, dtype=np.float64)
    bu = np.zeros(nlev + 1, dtype=np.float64)
    cu = np.zeros(nlev + 1, dtype=np.float64)
    du = np.zeros(nlev + 1, dtype=np.float64)

    for i in range(2, nlev):
        c = 2.0 * dt * nu_y[i] / (h[i] + h[i + 1]) / h[i]
        a = 2.0 * dt * nu_y[i - 1] / (h[i] + h[i - 1]) / h[i]
        linear_source = dt * l_sour[i]

        cu[i] = -cnpar * c
        au[i] = -cnpar * a
        bu[i] = 1.0 + cnpar * (a + c) - linear_source
        du[i] = (1.0 - (1.0 - cnpar) * (a + c)) * y[i]
        du[i] += (1.0 - cnpar) * (a * y[i - 1] + c * y[i + 1])
        du[i] += dt * q_sour[i]

    if bc_up == NEUMANN:
        a = 2.0 * dt * nu_y[nlev - 1] / (h[nlev] + h[nlev - 1]) / h[nlev]
        linear_source = dt * l_sour[nlev]
        au[nlev] = -cnpar * a
        if posconc == 1 and y_up < 0.0:
            bu[nlev] = 1.0 - au[nlev] - linear_source - dt * y_up / y[nlev] / h[nlev]
            du[nlev] = y[nlev] + dt * q_sour[nlev]
            du[nlev] += (1.0 - cnpar) * a * (y[nlev - 1] - y[nlev])
        else:
            bu[nlev] = 1.0 - au[nlev] - linear_source
            du[nlev] = y[nlev] + dt * (q_sour[nlev] + y_up / h[nlev])
            du[nlev] += (1.0 - cnpar) * a * (y[nlev - 1] - y[nlev])
    else:
        au[nlev] = 0.0
        bu[nlev] = 1.0
        du[nlev] = y_up

    if bc_down == NEUMANN:
        c = 2.0 * dt * nu_y[1] / (h[1] + h[2]) / h[1]
        linear_source = dt * l_sour[1]
        cu[1] = -cnpar * c
        if posconc == 1 and y_down < 0.0:
            bu[1] = 1.0 - cu[1] - linear_source - dt * y_down / y[1] / h[1]
            du[1] = y[1] + dt * q_sour[1]
            du[1] += (1.0 - cnpar) * c * (y[2] - y[1])
        else:
            bu[1] = 1.0 - cu[1] - linear_source
            du[1] = y[1] + dt * (q_sour[1] + y_down / h[1])
            du[1] += (1.0 - cnpar) * c * (y[2] - y[1])
    else:
        cu[1] = 0.0
        bu[1] = 1.0
        du[1] = y_down

    if np.min(tau_r[1 : nlev + 1]) < 1.0e10:
        for i in range(1, nlev + 1):
            bu[i] += dt / tau_r[i]
            du[i] += dt / tau_r[i] * y_obs[i]

    lower = np.array([au[i] for i in range(1, nlev + 1)], dtype=np.float64)
    diagonal = np.array([bu[i] for i in range(1, nlev + 1)], dtype=np.float64)
    upper = np.array([cu[i] for i in range(1, nlev + 1)], dtype=np.float64)
    rhs = np.array([du[i] for i in range(1, nlev + 1)], dtype=np.float64)

    matrix = np.diag(diagonal)
    if nlev > 1:
        matrix += np.diag(lower[1:], k=-1)
        matrix += np.diag(upper[:-1], k=1)

    updated = y.copy()
    updated[1 : nlev + 1] = np.linalg.solve(matrix, rhs)
    return updated


def _field(values: np.ndarray) -> ti.Field:
    field = ti.field(dtype=ti.f64, shape=values.shape)
    for index in np.ndindex(values.shape):
        field[index] = values[index]
    return field


def test_diff_center_dirichlet_matches_numpy_reference() -> None:
    nlev = 4
    dt = 30.0
    cnpar = 0.7
    posconc = 0
    h = np.array([0.0, 1.0, 1.2, 0.8, 1.1], dtype=np.float64)
    nu_y = np.array([0.0, 0.15, 0.18, 0.12, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, -0.01, 0.02, -0.03, 0.01], dtype=np.float64)
    q_sour = np.array([0.0, 0.005, -0.001, 0.002, -0.003], dtype=np.float64)
    tau_r = np.array([1.0e12, 1.0e12, 600.0, 1.0e12, 300.0], dtype=np.float64)
    y_obs = np.array([0.0, 1.2, 1.0, 0.9, 0.8], dtype=np.float64)
    y = np.array([0.0, 1.5, 1.1, 0.7, 0.2], dtype=np.float64)
    y_up = 0.4
    y_down = 1.8

    workspace = init_tridiagonal(nlev)
    y_field = _field(y)
    diff_center_kernel(
        nlev,
        dt,
        cnpar,
        posconc,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        _field(h),
        _field(nu_y),
        _field(l_sour),
        _field(q_sour),
        _field(tau_r),
        _field(y_obs),
        y_field,
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
    )

    expected = _build_reference_matrix(
        nlev,
        dt,
        cnpar,
        posconc,
        h,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        nu_y,
        l_sour,
        q_sour,
        tau_r,
        y_obs,
        y,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    assert np.allclose(result[1:], expected[1:], rtol=1e-12, atol=1e-12)


def test_diff_center_neumann_patankar_matches_numpy_reference() -> None:
    nlev = 4
    dt = 10.0
    cnpar = 1.0
    posconc = 1
    h = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    nu_y = np.array([0.0, 0.08, 0.09, 0.1, 0.0], dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.array([0.0, 0.01, 0.0, 0.02, 0.0], dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e12, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)
    y = np.array([0.0, 0.8, 0.7, 0.6, 0.5], dtype=np.float64)
    y_up = -0.015
    y_down = -0.02

    workspace = init_tridiagonal(nlev)
    y_field = _field(y)
    diff_center_kernel(
        nlev,
        dt,
        cnpar,
        posconc,
        NEUMANN,
        NEUMANN,
        y_up,
        y_down,
        _field(h),
        _field(nu_y),
        _field(l_sour),
        _field(q_sour),
        _field(tau_r),
        _field(y_obs),
        y_field,
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
    )

    expected = _build_reference_matrix(
        nlev,
        dt,
        cnpar,
        posconc,
        h,
        NEUMANN,
        NEUMANN,
        y_up,
        y_down,
        nu_y,
        l_sour,
        q_sour,
        tau_r,
        y_obs,
        y,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    assert np.allclose(result[1:], expected[1:], rtol=1e-12, atol=1e-12)
    assert np.isfinite(result[1:]).all()
    assert np.all(result[1:] >= 0.0)


def test_diff_center_multicolumn_ncols_one_matches_single_column() -> None:
    nlev = 3
    dt = 15.0
    cnpar = 0.5
    posconc = 0
    h = np.array([0.0, 1.1, 0.9, 1.0], dtype=np.float64)
    nu_y = np.array([0.0, 0.12, 0.13, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, 0.01, -0.02, 0.03], dtype=np.float64)
    q_sour = np.array([0.0, 0.004, 0.005, -0.002], dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e12, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)
    y = np.array([0.0, 0.6, 0.3, -0.1], dtype=np.float64)
    y_up = 0.2
    y_down = -0.4

    single_workspace = init_tridiagonal(nlev)
    single_y = _field(y)
    multi_workspace = init_tridiagonal(nlev, n_cols=1)
    multi_y = _field(np.expand_dims(y, axis=0))

    diff_center_kernel(
        nlev,
        dt,
        cnpar,
        posconc,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        _field(h),
        _field(nu_y),
        _field(l_sour),
        _field(q_sour),
        _field(tau_r),
        _field(y_obs),
        single_y,
        single_workspace.au,
        single_workspace.bu,
        single_workspace.cu,
        single_workspace.du,
        single_workspace.ru,
        single_workspace.qu,
    )
    diff_center_multi_kernel(
        1,
        nlev,
        dt,
        cnpar,
        posconc,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        _field(np.expand_dims(h, axis=0)),
        _field(np.expand_dims(nu_y, axis=0)),
        _field(np.expand_dims(l_sour, axis=0)),
        _field(np.expand_dims(q_sour, axis=0)),
        _field(np.expand_dims(tau_r, axis=0)),
        _field(np.expand_dims(y_obs, axis=0)),
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


def test_diff_center_no_nan_inf() -> None:
    nlev = 5
    dt = 60.0
    cnpar = 0.5
    posconc = 0
    h = np.array([0.0, 1.0, 1.2, 0.8, 0.9, 1.1], dtype=np.float64)
    nu_y = np.array([0.0, 0.1, 0.15, 0.12, 0.08, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, -0.01, 0.02, -0.01, 0.03, 0.0], dtype=np.float64)
    q_sour = np.array([0.0, 0.005, -0.003, 0.002, -0.001, 0.0], dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e12, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)
    y = np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)
    y_up = 0.1
    y_down = 1.2

    workspace = init_tridiagonal(nlev)
    y_field = _field(y)
    diff_center_kernel(
        nlev,
        dt,
        cnpar,
        posconc,
        NEUMANN,
        NEUMANN,
        y_up,
        y_down,
        _field(h),
        _field(nu_y),
        _field(l_sour),
        _field(q_sour),
        _field(tau_r),
        _field(y_obs),
        y_field,
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
    )
    result = np.array([y_field[i] for i in range(nlev + 1)])
    assert np.isfinite(result[1:]).all(), "diff_center produced NaN or Inf"


def test_analytic_sinusoidal_diffusion() -> None:
    """Verify diff_center against analytic diffusion of sin(pi z / L)."""

    nlev = 50
    depth = 50.0
    kappa = 1.0e-3
    dt = 10.0
    nsteps = 1000
    cnpar = 0.5
    dz = depth / (nlev - 1)

    z = np.zeros(nlev + 1, dtype=np.float64)
    z[1:] = np.linspace(0.0, depth, nlev)

    h = np.zeros(nlev + 1, dtype=np.float64)
    h[1:] = dz
    y = np.zeros(nlev + 1, dtype=np.float64)
    y[1:] = np.sin(np.pi * z[1:] / depth)
    nu_y = np.full(nlev + 1, kappa, dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e15, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)

    h_field = ti.field(dtype=ti.f64, shape=(1, nlev + 1))
    nu_y_field = ti.field(dtype=ti.f64, shape=(1, nlev + 1))
    l_sour_field = ti.field(dtype=ti.f64, shape=(1, nlev + 1))
    q_sour_field = ti.field(dtype=ti.f64, shape=(1, nlev + 1))
    tau_r_field = ti.field(dtype=ti.f64, shape=(1, nlev + 1))
    y_obs_field = ti.field(dtype=ti.f64, shape=(1, nlev + 1))
    y_field = ti.field(dtype=ti.f64, shape=(1, nlev + 1))
    workspace = init_tridiagonal(nlev, n_cols=1)

    fill_field_from_array(h_field, h)
    fill_field_from_array(nu_y_field, nu_y)
    fill_field_from_array(l_sour_field, l_sour)
    fill_field_from_array(q_sour_field, q_sour)
    fill_field_from_array(tau_r_field, tau_r)
    fill_field_from_array(y_obs_field, y_obs)
    fill_field_from_array(y_field, y)

    for _ in range(nsteps):
        diff_center_multi_kernel(
            1,
            nlev,
            dt,
            cnpar,
            0,
            DIRICHLET,
            DIRICHLET,
            0.0,
            0.0,
            h_field,
            nu_y_field,
            l_sour_field,
            q_sour_field,
            tau_r_field,
            y_obs_field,
            y_field,
            workspace.au,
            workspace.bu,
            workspace.cu,
            workspace.du,
            workspace.ru,
            workspace.qu,
        )

    result = read_field_array(y_field)
    t_final = nsteps * dt
    analytic = np.zeros(nlev + 1, dtype=np.float64)
    analytic[1:] = np.exp(-kappa * (np.pi / depth) ** 2 * t_final) * np.sin(
        np.pi * z[1:] / depth
    )

    np.testing.assert_allclose(result[1:], analytic[1:], rtol=1.0e-2, atol=1.0e-6)
