import numpy as np
import pytest

from pygotm.util.tridiagonal import (
    TridiagonalBatchWorkspace,
    TridiagonalWorkspace,
    clean_tridiagonal,
    init_tridiagonal,
    tridiagonal,
)


def _dense_tridiagonal_matrix(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    matrix = np.diag(diagonal)
    matrix += np.diag(lower[1:], k=-1)
    matrix += np.diag(upper[:-1], k=1)
    return matrix


def _call_tridiagonal(
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    fi: int,
    lt: int,
) -> np.ndarray:
    """Helper: allocate work arrays, call tridiagonal, return value array."""
    n = len(au)
    ru = np.zeros(n, dtype=np.float64)
    qu = np.zeros(n, dtype=np.float64)
    value = np.zeros(n, dtype=np.float64)
    tridiagonal(au.copy(), bu.copy(), cu.copy(), du.copy(), ru, qu, value, fi, lt)
    return value


def test_workspace_construction() -> None:
    ws = TridiagonalWorkspace(nlev=8)
    assert ws.nlev == 8
    assert ws.au.shape == (9,)
    assert ws.bu.shape == (9,)
    assert ws.cu.shape == (9,)
    assert ws.du.shape == (9,)
    assert ws.ru.shape == (9,)
    assert ws.qu.shape == (9,)
    assert ws.au.dtype == np.float64


def test_batch_workspace_construction() -> None:
    ws = TridiagonalBatchWorkspace(nlev=10, batch_size=4)
    assert ws.au.shape == (4, 11)
    assert ws.au.dtype == np.float64


def test_init_tridiagonal_returns_workspace() -> None:
    ws = init_tridiagonal(nlev=5)
    assert isinstance(ws, TridiagonalWorkspace)
    assert ws.au.shape == (6,)


def test_clean_tridiagonal_is_noop() -> None:
    ws = init_tridiagonal(nlev=3)
    clean_tridiagonal(ws)  # must not raise


def test_single_column_solver_matches_numpy_for_offset_span() -> None:
    n = 9
    fi = 1
    lt = n - 1

    dx = 1.0 / n
    x = np.linspace(dx, 1.0 - dx, n - 1)
    rhs = np.pi**2 * np.sin(np.pi * x) * dx**2
    lower = np.full(n, -1.0)
    diagonal = np.full(n, 2.0)
    upper = np.full(n, -1.0)
    # zero out sentinels outside [fi, lt]
    lower[0] = 0.0
    upper[n - 1] = 0.0

    du = np.zeros(n, dtype=np.float64)
    du[fi : lt + 1] = rhs
    result = _call_tridiagonal(lower, diagonal, upper, du, fi, lt)

    rhs_inner = lower[1 : n - 1 + 1]  # align with interior
    mat = _dense_tridiagonal_matrix(lower[1:], diagonal[1:], upper[:-1])
    expected = np.linalg.solve(mat, rhs)
    np.testing.assert_allclose(result[fi : lt + 1], expected, rtol=1e-12, atol=1e-12)


def test_single_column_solver_handles_zero_based_span() -> None:
    n = 6
    fi = 0
    lt = n - 1

    lower = np.array([0.0, -0.4, -0.3, -0.2, -0.1, -0.2])
    diagonal = np.array([1.5, 1.7, 1.8, 1.6, 1.9, 1.4])
    upper = np.array([-0.2, -0.1, -0.4, -0.2, -0.3, 0.0])
    rhs = np.array([0.1, 0.4, -0.2, 0.7, -0.1, 0.3])

    result = _call_tridiagonal(
        lower.astype(np.float64),
        diagonal.astype(np.float64),
        upper.astype(np.float64),
        rhs.astype(np.float64),
        fi,
        lt,
    )

    expected = np.linalg.solve(_dense_tridiagonal_matrix(lower, diagonal, upper), rhs)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(result).all()


def test_single_equation_path() -> None:
    au = np.zeros(1, dtype=np.float64)
    bu = np.array([2.5])
    cu = np.zeros(1, dtype=np.float64)
    du = np.array([-0.75])
    result = _call_tridiagonal(au, bu, cu, du, 0, 0)
    assert result[0] == pytest.approx(-0.3)


def test_batch_parity() -> None:
    """Two identical columns through a batch loop must match single-column result."""
    nlev = 10
    fi, lt = 1, nlev
    batch_size = 2

    ws_s = TridiagonalWorkspace(nlev)
    ws_s.au[fi : lt + 1] = -1.0
    ws_s.bu[fi : lt + 1] = 2.0
    ws_s.cu[fi : lt + 1] = -1.0
    ws_s.du[fi : lt + 1] = np.linspace(0.1, 1.0, lt - fi + 1)

    val_s = np.zeros(nlev + 1, dtype=np.float64)
    tridiagonal(ws_s.au, ws_s.bu, ws_s.cu, ws_s.du, ws_s.ru, ws_s.qu, val_s, fi, lt)

    ws_b = TridiagonalBatchWorkspace(nlev, batch_size)
    for b in range(batch_size):
        ws_b.au[b, fi : lt + 1] = ws_s.au[fi : lt + 1]
        ws_b.bu[b, fi : lt + 1] = ws_s.bu[fi : lt + 1]
        ws_b.cu[b, fi : lt + 1] = ws_s.cu[fi : lt + 1]
        ws_b.du[b, fi : lt + 1] = ws_s.du[fi : lt + 1]

    val_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    for b in range(batch_size):
        tridiagonal(
            ws_b.au[b],
            ws_b.bu[b],
            ws_b.cu[b],
            ws_b.du[b],
            ws_b.ru[b],
            ws_b.qu[b],
            val_b[b],
            fi,
            lt,
        )

    for b in range(batch_size):
        np.testing.assert_array_equal(val_b[b], val_s)
