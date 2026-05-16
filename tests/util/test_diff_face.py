import numpy as np
import pytest

from pygotm.util.diff_face import DIRICHLET, NEUMANN, diff_face, diff_face_batch
from pygotm.util.tridiagonal import TridiagonalBatchWorkspace, TridiagonalWorkspace


def _call_diff_face(
    nlev: int,
    dt: float,
    cnpar: float,
    h: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    nu_y: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Call diff_face in-place; return (y_out, nu_y_out) after the step."""
    ws = TridiagonalWorkspace(nlev)
    y_out = y.copy()
    nu_out = nu_y.copy()
    diff_face(
        nlev,
        dt,
        cnpar,
        h,
        bc_up,
        bc_down,
        y_up,
        y_down,
        nu_out,
        l_sour,
        q_sour,
        y_out,
        ws.au,
        ws.bu,
        ws.cu,
        ws.du,
        ws.ru,
        ws.qu,
    )
    return y_out, nu_out


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


def test_dirichlet_matches_numpy_reference() -> None:
    nlev = 5
    dt = 20.0
    cnpar = 0.6
    h = np.array([0.0, 1.0, 1.1, 0.9, 1.2, 1.0], dtype=np.float64)
    nu_y = np.array([0.0, 0.05, 0.07, 0.08, 0.06, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, 0.02, -0.01, 0.03, 0.01, 0.0], dtype=np.float64)
    q_sour = np.array([0.0, 0.002, -0.003, 0.004, -0.001, 0.0], dtype=np.float64)
    y = np.array([0.0, 0.9, 0.7, 0.4, 0.2, 0.0], dtype=np.float64)
    y_up, y_down = 0.3, 1.1

    result, _ = _call_diff_face(
        nlev,
        dt,
        cnpar,
        h,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        nu_y,
        l_sour,
        q_sour,
        y,
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
    np.testing.assert_allclose(result[1:nlev], expected[1:nlev], rtol=1e-12, atol=1e-12)


def test_two_layer_bugfix_matches_reference() -> None:
    """nlev==2 triggers the Georg Umgiesser bug-fix that mutates nu_y in place."""
    nlev = 2
    dt = 12.0
    cnpar = 1.0
    h = np.array([0.0, 0.8, 1.1], dtype=np.float64)
    nu_y = np.array([0.0, 0.2, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, 0.01, 0.0], dtype=np.float64)
    q_sour = np.array([0.0, -0.002, 0.0], dtype=np.float64)
    y = np.array([0.0, 0.5, 0.0], dtype=np.float64)
    y_up, y_down = 0.04, -0.03

    result, nu_out = _call_diff_face(
        nlev,
        dt,
        cnpar,
        h,
        NEUMANN,
        NEUMANN,
        y_up,
        y_down,
        nu_y,
        l_sour,
        q_sour,
        y,
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
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert nu_out[0] == pytest.approx(nu_out[1])
    assert nu_out[nlev] == pytest.approx(nu_out[1])


def test_no_nan_inf() -> None:
    nlev = 5
    dt = 30.0
    cnpar = 0.6
    h = np.array([0.0, 1.0, 1.1, 0.9, 1.2, 1.0], dtype=np.float64)
    nu_y = np.array([0.0, 0.05, 0.07, 0.06, 0.08, 0.0], dtype=np.float64)
    l_sour = np.array([0.0, 0.01, -0.02, 0.03, -0.01, 0.0], dtype=np.float64)
    q_sour = np.array([0.0, 0.002, -0.001, 0.003, -0.002, 0.0], dtype=np.float64)
    y = np.array([0.0, -0.3, 0.5, 1.2, 0.7, 0.0], dtype=np.float64)
    y_up, y_down = 0.2, -0.1

    result, _ = _call_diff_face(
        nlev,
        dt,
        cnpar,
        h,
        NEUMANN,
        NEUMANN,
        y_up,
        y_down,
        nu_y,
        l_sour,
        q_sour,
        y,
    )
    assert np.isfinite(result[1:nlev]).all(), "diff_face produced NaN or Inf"


def test_batch_parity() -> None:
    """diff_face_batch with 2 identical columns must match single-column result."""
    nlev = 5
    dt = 20.0
    cnpar = 0.6
    batch_size = 2
    h = np.array([0.0, 1.0, 1.1, 0.9, 1.2, 1.0], dtype=np.float64)
    nu_y = np.array([0.0, 0.05, 0.07, 0.08, 0.06, 0.0], dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)
    y = np.array([0.0, 0.9, 0.7, 0.4, 0.2, 0.0], dtype=np.float64)
    y_up, y_down = 0.3, 1.1

    expected, _ = _call_diff_face(
        nlev,
        dt,
        cnpar,
        h,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        nu_y,
        l_sour,
        q_sour,
        y,
    )

    ws = TridiagonalBatchWorkspace(nlev, batch_size)
    h_b = np.tile(h, (batch_size, 1))
    nu_b = np.tile(nu_y, (batch_size, 1))
    ls_b = np.tile(l_sour, (batch_size, 1))
    qs_b = np.tile(q_sour, (batch_size, 1))
    y_b = np.tile(y, (batch_size, 1))

    diff_face_batch(
        batch_size,
        nlev,
        dt,
        cnpar,
        h_b,
        DIRICHLET,
        DIRICHLET,
        y_up,
        y_down,
        nu_b,
        ls_b,
        qs_b,
        y_b,
        ws.au,
        ws.bu,
        ws.cu,
        ws.du,
        ws.ru,
        ws.qu,
    )

    for b in range(batch_size):
        np.testing.assert_array_equal(y_b[b], expected)
