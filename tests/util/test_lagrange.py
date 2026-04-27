"""Tests for util/lagrange.py — Step 1.8 of GOTM translation plan."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygotm.util.lagrange import RND_VAR, VISC_BACK, lagrange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NLEV = 10
DEPTH = 50.0  # [m]


def _uniform_grid(nlev: int = NLEV, depth: float = DEPTH) -> np.ndarray:
    """Return zlev[0..nlev] uniformly spaced from -depth (bottom) to 0 (surface)."""
    return np.linspace(-depth, 0.0, nlev + 1)


def _const_nuh(nlev: int = NLEV, value: float = 1e-4) -> np.ndarray:
    """Return nuh[0..nlev] constant at `value` [m²/s]."""
    return np.full(nlev + 1, value)


def _zero_nuh(nlev: int = NLEV) -> np.ndarray:
    return np.zeros(nlev + 1)


def _find_layer(zp: float, zlev: np.ndarray) -> int:
    """Return the 1-based layer index containing particle at zp."""
    nlev = len(zlev) - 1
    for i in range(1, nlev + 1):
        if zlev[i - 1] <= zp <= zlev[i]:
            return i
    return nlev  # fallback


def _make_particles(
    n: int, zlev: np.ndarray, depth: float = DEPTH
) -> tuple[np.ndarray, np.ndarray]:
    """Create n particles uniformly distributed in the water column."""
    nlev = len(zlev) - 1
    zp = np.linspace(-depth + 1.0, -1.0, n)
    zi = np.array([_find_layer(z, zlev) for z in zp], dtype=int)
    return zi, zp


# ---------------------------------------------------------------------------
# Import / smoke
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.util.lagrange import lagrange  # noqa: F401


def test_smoke_single_particle() -> None:
    zlev = _uniform_grid()
    nuh = _const_nuh()
    active = np.array([True])
    zi = np.array([5], dtype=int)
    zp = np.array([-DEPTH / 2.0])

    lagrange(NLEV, 100.0, zlev, nuh, 0.0, 1, active, zi, zp)

    assert zi.shape == (1,)
    assert zp.shape == (1,)


def test_smoke_multiple_particles() -> None:
    nlev = NLEV
    zlev = _uniform_grid(nlev)
    nuh = _const_nuh(nlev)
    npar = 20
    zi, zp = _make_particles(npar, zlev)
    active = np.ones(npar, dtype=bool)

    lagrange(nlev, 600.0, zlev, nuh, 0.0, npar, active, zi, zp,
             rng=np.random.default_rng(42))

    assert zi.shape == (npar,)
    assert zp.shape == (npar,)


# ---------------------------------------------------------------------------
# Physical bounds: particles must stay within [-depth, 0]
# ---------------------------------------------------------------------------


def test_particles_remain_in_domain_calm() -> None:
    """All particles must stay within [-depth, 0] under typical diffusion."""
    zlev = _uniform_grid()
    nuh = _const_nuh()
    npar = 50
    zi, zp = _make_particles(npar, zlev)
    active = np.ones(npar, dtype=bool)

    for _ in range(100):
        lagrange(NLEV, 60.0, zlev, nuh, 0.0, npar, active, zi, zp,
                 rng=np.random.default_rng(7))

    assert np.all(zp >= -DEPTH)
    assert np.all(zp <= 0.0)


def test_particles_remain_in_domain_strong_upwelling() -> None:
    """Large upward velocity must not push particles above surface."""
    zlev = _uniform_grid()
    nuh = _const_nuh()
    npar = 20
    zi, zp = _make_particles(npar, zlev)
    active = np.ones(npar, dtype=bool)

    for _ in range(50):
        lagrange(NLEV, 60.0, zlev, nuh, 5.0, npar, active, zi, zp,
                 rng=np.random.default_rng(13))

    assert np.all(zp >= -DEPTH)
    assert np.all(zp <= 0.0)


def test_particles_remain_in_domain_strong_downwelling() -> None:
    """Large downward velocity must not push particles below bottom."""
    zlev = _uniform_grid()
    nuh = _const_nuh()
    npar = 20
    zi, zp = _make_particles(npar, zlev)
    active = np.ones(npar, dtype=bool)

    for _ in range(50):
        lagrange(NLEV, 60.0, zlev, nuh, -5.0, npar, active, zi, zp,
                 rng=np.random.default_rng(99))

    assert np.all(zp >= -DEPTH)
    assert np.all(zp <= 0.0)


# ---------------------------------------------------------------------------
# Deterministic case: zero viscosity → pure drift, no random spread
# ---------------------------------------------------------------------------


def test_zero_viscosity_zero_velocity_particle_stationary() -> None:
    """With nuh=0 and w=0, visc=0 → step=0 → particle does not move."""
    zlev = _uniform_grid()
    nuh = _zero_nuh()
    zi_init = np.array([5], dtype=int)
    zp_init = np.array([zlev[4] + 0.1])  # interior of layer 5
    zi = zi_init.copy()
    zp = zp_init.copy()
    active = np.array([True])

    lagrange(NLEV, 100.0, zlev, nuh, 0.0, 1, active, zi, zp,
             rng=np.random.default_rng(0))

    assert zp[0] == pytest.approx(zp_init[0], abs=1e-15)
    assert zi[0] == zi_init[0]


def test_zero_viscosity_pure_upward_drift() -> None:
    """With nuh=0 and w>0, particle drifts upward by exactly w*dt each step."""
    zlev = np.linspace(-100.0, 0.0, 11)
    nuh = _zero_nuh(nlev=10)
    w = 0.1  # m/s upward
    dt = 5.0  # s
    expected_step = w * dt

    zi = np.array([5], dtype=int)
    zp_start = np.array([zlev[4] + 0.01])  # well inside layer 5
    zp = zp_start.copy()
    active = np.array([True])

    lagrange(10, dt, zlev, nuh, w, 1, active, zi, zp,
             rng=np.random.default_rng(0))

    assert zp[0] == pytest.approx(zp_start[0] + expected_step, rel=1e-12)


def test_zero_viscosity_pure_downward_drift() -> None:
    """With nuh=0 and w<0, particle drifts downward by exactly w*dt each step."""
    zlev = np.linspace(-100.0, 0.0, 11)
    nuh = _zero_nuh(nlev=10)
    w = -0.1  # m/s downward
    dt = 5.0  # s
    expected_step = w * dt

    zi = np.array([6], dtype=int)
    zp_start = np.array([zlev[5] + 0.1])  # well inside layer 6
    zp = zp_start.copy()
    active = np.array([True])

    lagrange(10, dt, zlev, nuh, w, 1, active, zi, zp,
             rng=np.random.default_rng(0))

    assert zp[0] == pytest.approx(zp_start[0] + expected_step, rel=1e-12)


# ---------------------------------------------------------------------------
# Reflective boundary conditions
# ---------------------------------------------------------------------------


def test_reflective_bc_at_surface() -> None:
    """Particle pushed above z=0 must be reflected back into the domain."""
    zlev = _uniform_grid(nlev=4, depth=10.0)  # levels: -10, -7.5, -5, -2.5, 0
    nuh = _zero_nuh(nlev=4)
    w = 100.0  # enormous upwelling — will definitely exceed surface
    dt = 1.0

    zi = np.array([4], dtype=int)
    zp = np.array([-0.5])  # very close to surface
    active = np.array([True])

    lagrange(4, dt, zlev, nuh, w, 1, active, zi, zp,
             rng=np.random.default_rng(0))

    assert zp[0] <= 0.0
    assert zp[0] >= -10.0


def test_reflective_bc_at_bottom() -> None:
    """Particle pushed below -depth must be reflected back into the domain."""
    zlev = _uniform_grid(nlev=4, depth=10.0)
    nuh = _zero_nuh(nlev=4)
    w = -100.0  # enormous downwelling
    dt = 1.0

    zi = np.array([1], dtype=int)
    zp = np.array([-9.5])  # very close to bottom
    active = np.array([True])

    lagrange(4, dt, zlev, nuh, w, 1, active, zi, zp,
             rng=np.random.default_rng(0))

    assert zp[0] >= -10.0
    assert zp[0] <= 0.0


# ---------------------------------------------------------------------------
# Layer index tracking
# ---------------------------------------------------------------------------


def test_layer_index_consistent_with_position() -> None:
    """After each step, zi[n] must be the layer that contains zp[n]."""
    zlev = _uniform_grid()
    nuh = _const_nuh(value=1e-3)
    npar = 30
    zi, zp = _make_particles(npar, zlev)
    active = np.ones(npar, dtype=bool)

    rng = np.random.default_rng(42)
    for _ in range(20):
        lagrange(NLEV, 120.0, zlev, nuh, 0.0, npar, active, zi, zp, rng=rng)

    for n in range(npar):
        i = zi[n]
        assert 1 <= i <= NLEV
        assert zlev[i - 1] <= zp[n] <= zlev[i] + 1e-14


def test_layer_index_bounds() -> None:
    """zi must always be in [1, nlev] after the walk."""
    zlev = _uniform_grid()
    nuh = _const_nuh(value=1e-2)  # large diffusivity → energetic random walk
    npar = 50
    zi, zp = _make_particles(npar, zlev)
    active = np.ones(npar, dtype=bool)

    rng = np.random.default_rng(17)
    for _ in range(50):
        lagrange(NLEV, 300.0, zlev, nuh, 0.0, npar, active, zi, zp, rng=rng)

    assert np.all(zi >= 1)
    assert np.all(zi <= NLEV)


def test_layer_index_updates_when_particle_crosses_layer_boundary() -> None:
    """A particle that drifts from layer 3 to layer 4 must have zi updated."""
    zlev = np.linspace(-50.0, 0.0, 11)  # 10 layers, 5 m each
    nuh = _zero_nuh(nlev=10)
    w = 0.0
    dt = 1.0

    # Place particle at top of layer 3: just below zlev[3]
    zp_start = zlev[2] + 0.01  # inside layer 3 (above zlev[2], below zlev[3])
    zi = np.array([3], dtype=int)
    zp = np.array([zp_start])
    active = np.array([True])

    # Manually craft a large upward w to cross into layer 4
    # With nuh=0, step = dt * w → set w large enough to cross zlev[3]
    gap_to_next = zlev[3] - zp_start + 0.1
    w_cross = gap_to_next / dt

    lagrange(10, dt, zlev, nuh, w_cross, 1, active, zi, zp,
             rng=np.random.default_rng(0))

    assert zi[0] == 4
    assert zp[0] > zlev[3]
    assert zp[0] <= zlev[4]


# ---------------------------------------------------------------------------
# Diffusion step magnitude sanity check
# ---------------------------------------------------------------------------


def test_diffusion_step_magnitude() -> None:
    """The RMS step should equal sqrt(2 * nuh * dt) (Fickian diffusion)."""
    zlev = _uniform_grid(nlev=4, depth=100.0)
    nuh_val = 1e-3
    nuh = _const_nuh(nlev=4, value=nuh_val)
    dt = 600.0
    npar = 10_000

    # Place all particles at the centre of layer 2
    zi = np.full(npar, 2, dtype=int)
    zp_start = -75.0  # well inside domain so no reflections in one step
    zp = np.full(npar, zp_start)
    active = np.ones(npar, dtype=bool)

    lagrange(4, dt, zlev, nuh, 0.0, npar, active, zi, zp,
             rng=np.random.default_rng(123))

    steps = zp - zp_start
    rms = math.sqrt(np.mean(steps**2))

    # Expected RMS from Visser scheme: sqrt(2 * nuh * dt)
    expected_rms = math.sqrt(2.0 * nuh_val * dt)
    assert rms == pytest.approx(expected_rms, rel=0.05)


# ---------------------------------------------------------------------------
# NaN / Inf guard
# ---------------------------------------------------------------------------


def test_no_nan_inf_standard_run() -> None:
    """No NaN or Inf in zp or zi after a typical run."""
    zlev = _uniform_grid()
    nuh = _const_nuh()
    npar = 40
    zi, zp = _make_particles(npar, zlev)
    active = np.ones(npar, dtype=bool)

    for _ in range(50):
        lagrange(NLEV, 300.0, zlev, nuh, 0.0, npar, active, zi, zp,
                 rng=np.random.default_rng(5))

    assert not np.any(np.isnan(zp))
    assert not np.any(np.isinf(zp))
    assert not np.any(np.isnan(zi.astype(float)))


def test_no_nan_inf_zero_viscosity() -> None:
    """Zero viscosity must not produce NaN (sqrt(0) should be 0)."""
    zlev = _uniform_grid()
    nuh = _zero_nuh()
    npar = 10
    zi, zp = _make_particles(npar, zlev)
    active = np.ones(npar, dtype=bool)

    lagrange(NLEV, 100.0, zlev, nuh, 0.0, npar, active, zi, zp,
             rng=np.random.default_rng(3))

    assert not np.any(np.isnan(zp))
    assert not np.any(np.isinf(zp))


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_constants_match_fortran() -> None:
    """Module constants must match Fortran PARAMETER values."""
    assert VISC_BACK == pytest.approx(0.0e-6, abs=1e-20)
    assert RND_VAR == pytest.approx(0.333333333, rel=1e-9)


# ---------------------------------------------------------------------------
# Reproducibility with seeded RNG
# ---------------------------------------------------------------------------


def test_reproducible_with_seeded_rng() -> None:
    """Same seed must produce identical results."""
    zlev = _uniform_grid()
    nuh = _const_nuh()
    npar = 10
    active = np.ones(npar, dtype=bool)

    zi1, zp1 = _make_particles(npar, zlev)
    zi2 = zi1.copy()
    zp2 = zp1.copy()

    lagrange(NLEV, 300.0, zlev, nuh, 0.01, npar, active, zi1, zp1,
             rng=np.random.default_rng(42))
    lagrange(NLEV, 300.0, zlev, nuh, 0.01, npar, active, zi2, zp2,
             rng=np.random.default_rng(42))

    np.testing.assert_array_equal(zp1, zp2)
    np.testing.assert_array_equal(zi1, zi2)
