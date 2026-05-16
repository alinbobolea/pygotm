import numpy as np

from pygotm.util.diff_center import DIRICHLET, NEUMANN, diff_center, diff_center_batch
from pygotm.util.tridiagonal import TridiagonalBatchWorkspace, TridiagonalWorkspace


def _make_ws(nlev: int) -> TridiagonalWorkspace:
    return TridiagonalWorkspace(nlev)


def _call_diff_center(
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
    ws = _make_ws(nlev)
    y_out = y.copy()
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
        y_out,
        ws.au,
        ws.bu,
        ws.cu,
        ws.du,
        ws.ru,
        ws.qu,
    )
    return y_out


def test_dirichlet_bc_recovers_prescribed_values() -> None:
    nlev = 4
    dt = 3600.0
    cnpar = 1.0
    h = np.ones(nlev + 1, dtype=np.float64)
    h[0] = 0.0
    nu_y = np.full(nlev + 1, 1e-4, dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e15, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)

    y_up = 2.0
    y_down = 0.5
    y = np.linspace(0.5, 2.0, nlev + 1)

    # Run many steps toward steady state
    for _ in range(500):
        y = _call_diff_center(
            nlev,
            dt,
            cnpar,
            0,
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

    assert y[nlev] == 2.0  # Dirichlet BC at top
    assert y[1] == 0.5  # Dirichlet BC at bottom


def test_neumann_zero_flux_gives_uniform_profile() -> None:
    """Zero-flux Neumann BC on both sides → uniform steady state."""
    nlev = 5
    dt = 1000.0
    cnpar = 1.0
    h = np.ones(nlev + 1, dtype=np.float64)
    h[0] = 0.0
    nu_y = np.full(nlev + 1, 1e-3, dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e15, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)

    y = np.array([0.0, 3.0, 1.0, 4.0, 1.5, 2.0], dtype=np.float64)
    y_mean = np.mean(y[1:])

    for _ in range(2000):
        y = _call_diff_center(
            nlev,
            dt,
            cnpar,
            0,
            h,
            NEUMANN,
            NEUMANN,
            0.0,
            0.0,
            nu_y,
            l_sour,
            q_sour,
            tau_r,
            y_obs,
            y,
        )

    np.testing.assert_allclose(y[1:], y_mean, rtol=1e-6, atol=1e-8)


def test_patankar_linearisation_prevents_negative_values() -> None:
    """posconc=1 with small negative flux: concentrations must stay positive.

    The Patankar (1980) linearisation moves a negative boundary flux to the
    implicit diagonal, preventing concentration from going negative.  The
    guard precondition (y > 0) is satisfied here by design: a tiny negative
    flux over a short run is insufficient to drive y to zero.
    """
    nlev = 4
    dt = 100.0  # short step — y stays well above zero
    cnpar = 1.0
    h = np.ones(nlev + 1, dtype=np.float64)
    h[0] = 0.0
    nu_y = np.full(nlev + 1, 1e-4, dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e15, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)

    y = np.array([0.0, 0.5, 0.4, 0.3, 0.2], dtype=np.float64)
    y_up = -1e-5  # small negative flux (loss) at surface
    y_down = -1e-5  # small negative flux (loss) at bottom

    for _ in range(20):
        y = _call_diff_center(
            nlev,
            dt,
            cnpar,
            1,
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

    assert np.all(y[1:] >= 0.0), "Patankar: concentrations must stay non-negative"
    assert np.isfinite(y[1:]).all()


def test_dirichlet_matches_numpy_solve() -> None:
    """Single step with Dirichlet BCs must match direct numpy tridiagonal solve."""
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
    y0 = np.array([0.0, 1.5, 1.1, 0.7, 0.2], dtype=np.float64)
    y_up = 0.4
    y_down = 1.8

    result = _call_diff_center(
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
        y0,
    )

    # Build reference tridiagonal matrix manually
    au = np.zeros(nlev + 1)
    bu = np.zeros(nlev + 1)
    cu = np.zeros(nlev + 1)
    du = np.zeros(nlev + 1)

    for i in range(2, nlev):
        c = 2.0 * dt * nu_y[i] / (h[i] + h[i + 1]) / h[i]
        a = 2.0 * dt * nu_y[i - 1] / (h[i] + h[i - 1]) / h[i]
        ls = dt * l_sour[i]
        cu[i] = -cnpar * c
        au[i] = -cnpar * a
        bu[i] = 1.0 + cnpar * (a + c) - ls
        du[i] = (1.0 - (1.0 - cnpar) * (a + c)) * y0[i]
        du[i] += (1.0 - cnpar) * (a * y0[i - 1] + c * y0[i + 1])
        du[i] += dt * q_sour[i]
    au[nlev] = 0.0
    bu[nlev] = 1.0
    du[nlev] = y_up
    cu[1] = 0.0
    bu[1] = 1.0
    du[1] = y_down
    if np.min(tau_r[1:]) < 1.0e10:
        for i in range(1, nlev + 1):
            bu[i] += dt / tau_r[i]
            du[i] += dt / tau_r[i] * y_obs[i]

    lo = au[1:]
    dia = bu[1:]
    up = cu[1:]
    rhs = du[1:]
    mat = np.diag(dia) + np.diag(lo[1:], k=-1) + np.diag(up[:-1], k=1)
    expected = np.linalg.solve(mat, rhs)
    np.testing.assert_allclose(result[1:], expected, rtol=1e-12, atol=1e-12)


def test_no_nan_inf() -> None:
    nlev = 5
    dt = 60.0
    cnpar = 0.5
    h = np.array([0.0, 1.0, 1.2, 0.8, 0.9, 1.1], dtype=np.float64)
    nu_y = np.array([0.0, 0.1, 0.15, 0.12, 0.08, 0.0], dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e12, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)
    y = np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)

    result = _call_diff_center(
        nlev,
        dt,
        cnpar,
        0,
        h,
        NEUMANN,
        NEUMANN,
        0.1,
        1.2,
        nu_y,
        l_sour,
        q_sour,
        tau_r,
        y_obs,
        y,
    )
    assert np.isfinite(result[1:]).all()


def test_analytic_sinusoidal_diffusion() -> None:
    """Verify diff_center against analytic diffusion decay of sin(pi z / L)."""
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

    for _ in range(nsteps):
        y = _call_diff_center(
            nlev,
            dt,
            cnpar,
            0,
            h,
            DIRICHLET,
            DIRICHLET,
            0.0,
            0.0,
            nu_y,
            l_sour,
            q_sour,
            tau_r,
            y_obs,
            y,
        )

    t_final = nsteps * dt
    analytic = np.zeros(nlev + 1, dtype=np.float64)
    analytic[1:] = np.exp(-kappa * (np.pi / depth) ** 2 * t_final) * np.sin(
        np.pi * z[1:] / depth
    )
    np.testing.assert_allclose(y[1:], analytic[1:], rtol=1e-2, atol=1e-6)


def test_batch_parity() -> None:
    """diff_center_batch with 2 identical columns must match single-column result."""
    nlev = 6
    dt = 100.0
    cnpar = 0.5
    batch_size = 2
    h = np.array([0.0, 1.0, 1.1, 0.9, 1.0, 0.8, 1.2], dtype=np.float64)
    nu_y = np.full(nlev + 1, 1e-3, dtype=np.float64)
    l_sour = np.zeros(nlev + 1, dtype=np.float64)
    q_sour = np.zeros(nlev + 1, dtype=np.float64)
    tau_r = np.full(nlev + 1, 1.0e15, dtype=np.float64)
    y_obs = np.zeros(nlev + 1, dtype=np.float64)
    y0 = np.array([0.0, 2.0, 1.8, 1.5, 1.2, 0.8, 0.3], dtype=np.float64)
    y_up, y_down = 0.3, 2.0

    # single-column reference
    expected = _call_diff_center(
        nlev,
        dt,
        cnpar,
        0,
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
        y0,
    )

    # batch with two identical columns
    ws = TridiagonalBatchWorkspace(nlev, batch_size)
    h_b = np.tile(h, (batch_size, 1))
    nu_b = np.tile(nu_y, (batch_size, 1))
    ls_b = np.tile(l_sour, (batch_size, 1))
    qs_b = np.tile(q_sour, (batch_size, 1))
    tr_b = np.tile(tau_r, (batch_size, 1))
    yo_b = np.tile(y_obs, (batch_size, 1))
    y_b = np.tile(y0, (batch_size, 1))

    diff_center_batch(
        batch_size,
        nlev,
        dt,
        cnpar,
        0,
        h_b,
        NEUMANN,
        NEUMANN,
        y_up,
        y_down,
        nu_b,
        ls_b,
        qs_b,
        tr_b,
        yo_b,
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
