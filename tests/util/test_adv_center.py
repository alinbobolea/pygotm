import numpy as np

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
    AdvectionBatchWorkspace,
    AdvectionWorkspace,
    adv_center,
    adv_center_batch,
    clean_adv_center,
    init_adv_center,
)


def _call_adv_center(
    nlev: int,
    dt: float,
    h: np.ndarray,
    ho: np.ndarray,
    ww: np.ndarray,
    bc_up: int,
    bc_down: int,
    y_up: float,
    y_down: float,
    method: int,
    mode: int,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Call adv_center; return (y_out, cu_out)."""
    ws = AdvectionWorkspace(nlev)
    y_out = y.copy()
    adv_center(nlev, dt, h, ho, ww, bc_up, bc_down, y_up, y_down, method, mode, y_out, ws.cu)
    return y_out, ws.cu.copy()


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
            cu[k] = ww[k] * _adv_reconstruct_ref(method, courant, y_upstream, y_central, y_downstream)

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


def _adv_reconstruct_ref(scheme: int, cfl: float, fuu: float, fu: float, fd: float) -> float:
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
            limiter = min(2.0 * ratio, (1.0 / 3.0) * max(1.0 + 2.0 * ratio, 2.0 + ratio), 2.0)
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


def test_workspace_construction() -> None:
    ws = AdvectionWorkspace(6)
    assert ws.cu.shape == (7,)
    assert ws.cu.dtype == np.float64

    bws = AdvectionBatchWorkspace(6, batch_size=2)
    assert bws.cu.shape == (2, 7)
    assert bws.cu.dtype == np.float64


def test_init_adv_center_and_clean() -> None:
    ws = init_adv_center(5)
    assert isinstance(ws, AdvectionWorkspace)
    assert ws.cu.shape == (6,)
    clean_adv_center(ws)  # must not raise


import pytest


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
def test_matches_reference_across_limiters(
    method: int, fuu: float, fu: float, fd: float, cfl: float,
) -> None:
    nlev = 2
    dt = 0.2
    h = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    ho = h.copy()
    speed = np.array([0.0, cfl * 0.5 * (h[1] + h[2]) / dt, 0.0], dtype=np.float64)
    y = np.array([fuu, fu, fd], dtype=np.float64)

    result, _ = _call_adv_center(
        nlev, dt, h, ho, speed,
        ZERO_DIVERGENCE, VALUE, 0.0, fuu,
        method, CONSERVATIVE, y,
    )
    expected, _ = _adv_center_reference(
        nlev, dt, h, speed,
        ZERO_DIVERGENCE, VALUE, 0.0, fuu,
        method, CONSERVATIVE, y,
    )
    np.testing.assert_allclose(result[1:], expected[1:], rtol=1e-12, atol=1e-12)


def test_conservative_split_iterations_match_reference() -> None:
    nlev = 4
    dt = 2.5
    h = np.array([0.0, 1.0, 0.8, 1.1, 0.9], dtype=np.float64)
    ho = np.array([0.0, 1.2, 1.0, 1.0, 0.7], dtype=np.float64)
    ww = np.array([0.0, 1.2, -1.4, 0.8, 0.2], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.4, 1.6, 0.8], dtype=np.float64)
    y_up, y_down = 0.15, -0.1

    result, cu_result = _call_adv_center(
        nlev, dt, h, ho, ww, FLUX, ONE_SIDED, y_up, y_down, P2_PDM, CONSERVATIVE, y,
    )
    expected, expected_flux = _adv_center_reference(
        nlev, dt, h, ww, FLUX, ONE_SIDED, y_up, y_down, P2_PDM, CONSERVATIVE, y,
    )
    np.testing.assert_allclose(result[1:], expected[1:], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cu_result, expected_flux, rtol=1e-12, atol=1e-12)


def test_nonconservative_matches_reference() -> None:
    nlev = 4
    dt = 1.25
    h = np.array([0.0, 1.0, 1.1, 0.9, 1.2], dtype=np.float64)
    ho = np.array([0.0, 0.9, 1.0, 1.2, 1.1], dtype=np.float64)
    ww = np.array([-0.2, -0.4, 0.5, -0.3, 0.1], dtype=np.float64)
    y = np.array([0.0, 0.6, 0.9, 0.2, 0.7], dtype=np.float64)
    y_up, y_down = 1.1, -0.3

    result, _ = _call_adv_center(
        nlev, dt, h, ho, ww, VALUE, ZERO_DIVERGENCE, y_up, y_down, MUSCL, NON_CONSERVATIVE, y,
    )
    expected, _ = _adv_center_reference(
        nlev, dt, h, ww, VALUE, ZERO_DIVERGENCE, y_up, y_down, MUSCL, NON_CONSERVATIVE, y,
    )
    np.testing.assert_allclose(result[1:], expected[1:], rtol=1e-12, atol=1e-12)
    assert np.isfinite(result[1:]).all()


def test_no_nan_inf() -> None:
    nlev = 4
    h = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    ho = h.copy()
    ww = np.array([0.0, 0.3, -0.2, 0.4, -0.1], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.5, 0.8, 0.2], dtype=np.float64)

    result, _ = _call_adv_center(
        nlev, 1.0, h, ho, ww, FLUX, FLUX, 0.0, 0.0, UPSTREAM, CONSERVATIVE, y,
    )
    assert np.isfinite(result[1:]).all(), "adv_center produced NaN or Inf"


def test_batch_parity() -> None:
    """adv_center_batch with 2 identical columns must match single-column result."""
    nlev = 4
    dt = 1.0
    batch_size = 2
    h = np.array([0.0, 1.0, 0.9, 1.1, 1.0], dtype=np.float64)
    ho = h.copy()
    ww = np.array([0.0, 0.4, -0.2, 0.3, -0.1], dtype=np.float64)
    y = np.array([0.0, 1.2, 0.8, 0.5, 0.1], dtype=np.float64)
    y_up, y_down = 0.2, -0.05

    expected, _ = _call_adv_center(
        nlev, dt, h, ho, ww, FLUX, VALUE, y_up, y_down, UPSTREAM, CONSERVATIVE, y,
    )

    ws = AdvectionBatchWorkspace(nlev, batch_size)
    h_b = np.tile(h, (batch_size, 1))
    ho_b = np.tile(ho, (batch_size, 1))
    ww_b = np.tile(ww, (batch_size, 1))
    y_b = np.tile(y, (batch_size, 1))

    adv_center_batch(
        batch_size, nlev, dt, h_b, ho_b, ww_b,
        FLUX, VALUE, y_up, y_down,
        UPSTREAM, CONSERVATIVE, y_b, ws.cu,
    )

    for b in range(batch_size):
        np.testing.assert_array_equal(y_b[b], expected)
