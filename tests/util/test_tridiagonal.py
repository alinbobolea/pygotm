import numpy as np
import pytest
import taichi as ti

from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.util.tridiagonal import (
    clean_tridiagonal,
    init_tridiagonal,
    tridiagonal,
    tridiagonal_column,
)


@ti_kernel
def solve_single_kernel(  # type: ignore[no-untyped-def]
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
    value: TemplateArg,
    fi: ti.i32,
    lt: ti.i32,
):
    tridiagonal(au, bu, cu, du, ru, qu, value, fi, lt)


@ti_kernel
def solve_multi_kernel(  # type: ignore[no-untyped-def]
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
    value: TemplateArg,
    n_cols: ti.i32,
    fi: ti.i32,
    lt: ti.i32,
):
    for col in range(n_cols):
        tridiagonal_column(col, au, bu, cu, du, ru, qu, value, fi, lt)


def _dense_tridiagonal_matrix(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    matrix = np.diag(diagonal)
    matrix += np.diag(lower[1:], k=-1)
    matrix += np.diag(upper[:-1], k=1)
    return matrix


def test_workspace_allocates_expected_fields_and_clean_releases_them() -> None:
    workspace = init_tridiagonal(8, n_cols=3)

    assert workspace.shape == (3, 9)
    assert workspace.names() == ("au", "bu", "cu", "du", "ru", "qu")
    assert workspace.au.shape == (3, 9)
    assert workspace.bu.shape == (3, 9)

    clean_tridiagonal(workspace)

    assert workspace.names() == ()
    assert not hasattr(workspace, "au")


def test_single_column_solver_matches_numpy_for_offset_span() -> None:
    n = 9
    fi = 1
    lt = n - 1
    workspace = init_tridiagonal(n - 1)
    value = ti.field(dtype=ti.f64, shape=(n,))

    dx = 1.0 / n
    x = np.linspace(dx, 1.0 - dx, n - 1)
    rhs = np.pi**2 * np.sin(np.pi * x) * dx**2
    lower = np.full(n - 1, -1.0)
    diagonal = np.full(n - 1, 2.0)
    upper = np.full(n - 1, -1.0)

    for index in range(n - 1):
        storage_index = index + 1
        workspace.au[storage_index] = lower[index]
        workspace.bu[storage_index] = diagonal[index]
        workspace.cu[storage_index] = upper[index]
        workspace.du[storage_index] = rhs[index]

    solve_single_kernel(
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
        value,
        fi,
        lt,
    )

    expected = np.linalg.solve(_dense_tridiagonal_matrix(lower, diagonal, upper), rhs)
    result = np.array([value[index + 1] for index in range(n - 1)])
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_single_column_solver_handles_zero_based_span() -> None:
    n = 6
    fi = 0
    lt = n - 1
    workspace = init_tridiagonal(n - 1)
    value = ti.field(dtype=ti.f64, shape=(n,))

    lower = np.array([0.0, -0.4, -0.3, -0.2, -0.1, -0.2])
    diagonal = np.array([1.5, 1.7, 1.8, 1.6, 1.9, 1.4])
    upper = np.array([-0.2, -0.1, -0.4, -0.2, -0.3, 0.0])
    rhs = np.array([0.1, 0.4, -0.2, 0.7, -0.1, 0.3])

    for index in range(n):
        workspace.au[index] = lower[index]
        workspace.bu[index] = diagonal[index]
        workspace.cu[index] = upper[index]
        workspace.du[index] = rhs[index]

    solve_single_kernel(
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
        value,
        fi,
        lt,
    )

    expected = np.linalg.solve(_dense_tridiagonal_matrix(lower, diagonal, upper), rhs)
    result = np.array([value[index] for index in range(n)])
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(result).all()


def test_single_equation_path_is_supported() -> None:
    workspace = init_tridiagonal(0)
    value = ti.field(dtype=ti.f64, shape=(1,))

    workspace.bu[0] = 2.5
    workspace.du[0] = -0.75

    solve_single_kernel(
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
        value,
        0,
        0,
    )

    assert value[0] == pytest.approx(-0.3)


def test_tridiagonal_column_single_equation_path() -> None:
    """tridiagonal_column with fi==lt solves a single 1x1 linear equation."""

    nlev = 5
    workspace = init_tridiagonal(nlev, n_cols=1)
    value = ti.field(dtype=ti.f64, shape=(1, nlev + 1))

    workspace.bu[0, 3] = 4.0
    workspace.du[0, 3] = 8.0

    solve_multi_kernel(
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
        value,
        1,
        3,
        3,
    )

    assert value[0, 3] == pytest.approx(2.0, rel=1e-12)


def test_multi_column_ncols_one_matches_single_column() -> None:
    n = 5
    fi = 0
    lt = n - 1
    lower = np.array([0.0, -0.1, -0.2, -0.3, -0.2])
    diagonal = np.array([1.3, 1.4, 1.5, 1.6, 1.7])
    upper = np.array([-0.4, -0.3, -0.2, -0.1, 0.0])
    rhs = np.array([0.2, -0.5, 0.1, 0.8, -0.3])

    single = init_tridiagonal(n - 1)
    single_value = ti.field(dtype=ti.f64, shape=(n,))
    multi = init_tridiagonal(n - 1, n_cols=1)
    multi_value = ti.field(dtype=ti.f64, shape=(1, n))

    for index in range(n):
        single.au[index] = lower[index]
        single.bu[index] = diagonal[index]
        single.cu[index] = upper[index]
        single.du[index] = rhs[index]

        multi.au[0, index] = lower[index]
        multi.bu[0, index] = diagonal[index]
        multi.cu[0, index] = upper[index]
        multi.du[0, index] = rhs[index]

    solve_single_kernel(
        single.au,
        single.bu,
        single.cu,
        single.du,
        single.ru,
        single.qu,
        single_value,
        fi,
        lt,
    )
    solve_multi_kernel(
        multi.au,
        multi.bu,
        multi.cu,
        multi.du,
        multi.ru,
        multi.qu,
        multi_value,
        1,
        fi,
        lt,
    )

    single_result = np.array([single_value[index] for index in range(n)])
    multi_result = np.array([multi_value[0, index] for index in range(n)])
    assert np.allclose(multi_result, single_result, rtol=1e-12, atol=1e-12)
