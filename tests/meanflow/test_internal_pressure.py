"""Tests for pygotm.meanflow.internal_pressure — baroclinic pressure gradient.

The subroutine computes the internal (baroclinic) pressure gradient and writes
the result into output arrays ``idpdx`` and ``idpdy`` (shape nlev+1, 1-indexed
layers 1..nlev).

Two computation modes are supported:

  int_press_type == 1:
      Horizontal T/S gradient method. Buoyancy gradients are estimated via a
      finite-difference perturbation of the equation of state, then vertically
      integrated from the surface downward via a trapezoidal scheme.

  int_press_type == 2:
      Plume method. Uses the existing buoyancy profile with a prescribed
      along-slope tilt.  Two sub-types:
        plume_type == 1: surface plume   idpdx(i) = slope_x*(buoy(i)-buoy(1))
        plume_type == 2: bottom plume    idpdx(i) = -slope_x*(buoy(nlev)-buoy(i))

Tests verify:
  - int_press_type==0 leaves idpdx/idpdy unchanged (no-op)
  - Type 1 with zero T/S gradient → zero output
  - Type 1 with uniform T gradient and linear EOS → analytic trapezoidal result
  - Type 1 x and y channels are independent
  - Type 2 surface-plume: zero at bottom layer, proportional to buoy difference elsewhere
  - Type 2 bottom-plume: zero at surface layer, correct sign at bottom
  - Type 2 analytic: linear buoyancy profile → linear output
  - No NaN or Inf for valid inputs
  - Boundary level k=0 is never modified
"""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.meanflow.internal_pressure import (
    INT_PRESS_GRADIENTS,
    INT_PRESS_NONE,
    INT_PRESS_PLUME,
    PLUME_BOTTOM,
    PLUME_SURFACE,
    internal_pressure,
)
from pygotm.meanflow.meanflow import MeanflowState, init_meanflow, post_init_meanflow
from pygotm.meanflow.updategrid import updategrid
from pygotm.util.density import METHOD_LINEAR_USER, DensityState, init_density

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NLEV = 20
_DEPTH = 100.0  # m (deep enough for pressure to matter in linear EOS)
_DT = 3600.0
_GRAVITY = 9.81
# Reference density for linear EOS tests
_RHO0 = 1027.0
# Thermal expansion [1/K] and haline contraction [kg/g] coefficients
_ALPHA0 = 2.0e-4
_BETA0 = 7.0e-4


def _make_state(nlev: int = _NLEV, depth: float = _DEPTH) -> MeanflowState:
    state = MeanflowState()
    init_meanflow(state, gravity=_GRAVITY)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    return state


def _make_density(
    nlev: int = _NLEV,
    T0: float = 10.0,
    S0: float = 35.0,
    rho0: float = _RHO0,
    alpha0: float = _ALPHA0,
    beta0: float = _BETA0,
) -> DensityState:
    ds = DensityState()
    ds.density_method = METHOD_LINEAR_USER
    ds.T0 = T0
    ds.S0 = S0
    ds.rho0 = rho0
    ds._rhob = rho0
    ds.alpha0 = alpha0
    ds.beta0 = beta0
    init_density(ds, nlev)
    return ds


def _make_outputs(nlev: int = _NLEV) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(nlev + 1), np.zeros(nlev + 1)


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.meanflow.internal_pressure import internal_pressure as _ip  # noqa: F401

    assert callable(_ip)


def test_constants_defined() -> None:
    assert INT_PRESS_NONE == 0
    assert INT_PRESS_GRADIENTS == 1
    assert INT_PRESS_PLUME == 2
    assert PLUME_SURFACE == 1
    assert PLUME_BOTTOM == 2


# ---------------------------------------------------------------------------
# 2. int_press_type == 0 — no-op
# ---------------------------------------------------------------------------


def test_type0_noop_idpdx() -> None:
    """int_press_type=0 must leave idpdx unchanged."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    idpdx, idpdy = _make_outputs(nlev)
    idpdx[1 : nlev + 1] = 0.5  # sentinel
    sentinel = idpdx.copy()

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_NONE,
    )

    np.testing.assert_array_equal(
        idpdx, sentinel, err_msg="Type 0 must not modify idpdx"
    )


def test_type0_noop_idpdy() -> None:
    """int_press_type=0 must leave idpdy unchanged."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    idpdx, idpdy = _make_outputs(nlev)
    idpdy[1 : nlev + 1] = -0.3  # sentinel
    sentinel = idpdy.copy()

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_NONE,
    )

    np.testing.assert_array_equal(
        idpdy, sentinel, err_msg="Type 0 must not modify idpdy"
    )


# ---------------------------------------------------------------------------
# 3. Type 1 — zero T/S gradient → zero pressure gradient
# ---------------------------------------------------------------------------


def test_type1_zero_gradient_idpdx() -> None:
    """With all-zero T/S gradients, idpdx must remain zero."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.T is not None
    assert state.S is not None
    state.T[1 : nlev + 1] = 10.0
    state.S[1 : nlev + 1] = 35.0

    idpdx, idpdy = _make_outputs(nlev)
    dsdx = np.zeros(nlev + 1)
    dtdx = np.zeros(nlev + 1)
    dsdy = np.zeros(nlev + 1)
    dtdy = np.zeros(nlev + 1)

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_GRADIENTS,
        dsdx=dsdx,
        dtdx=dtdx,
        dsdy=dsdy,
        dtdy=dtdy,
    )

    np.testing.assert_allclose(
        idpdx[1 : nlev + 1],
        0.0,
        atol=1e-14,
        err_msg="Zero gradient must give zero idpdx",
    )


def test_type1_zero_gradient_idpdy() -> None:
    """With all-zero T/S gradients, idpdy must remain zero."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.T is not None
    assert state.S is not None
    state.T[1 : nlev + 1] = 10.0
    state.S[1 : nlev + 1] = 35.0

    idpdx, idpdy = _make_outputs(nlev)
    dsdx = np.zeros(nlev + 1)
    dtdx = np.zeros(nlev + 1)
    dsdy = np.zeros(nlev + 1)
    dtdy = np.zeros(nlev + 1)

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_GRADIENTS,
        dsdx=dsdx,
        dtdx=dtdx,
        dsdy=dsdy,
        dtdy=dtdy,
    )

    np.testing.assert_allclose(
        idpdy[1 : nlev + 1],
        0.0,
        atol=1e-14,
        err_msg="Zero gradient must give zero idpdy",
    )


# ---------------------------------------------------------------------------
# 4. Type 1 — analytic verification with linear EOS and uniform T gradient
# ---------------------------------------------------------------------------


def test_type1_analytic_uniform_T_gradient_idpdx() -> None:
    """Uniform dT/dx with linear EOS → analytic trapezoidal integral of dB/dx.

    With linear EOS: get_rho(S, T, p) = rho0*(1 - alpha0*(T-T0) + beta0*(S-S0))
    Buoyancy: B = -g*(rho - rho0)/rho0
    For uniform dTdx = G_T, dsdx = 0:
        Br = rho0*(1 - alpha0*(T + dx*G_T - T0) + beta0*(S-S0))
        Bl = rho0*(1 - alpha0*(T - T0)           + beta0*(S-S0))
        dxB = (Br_buoy - Bl_buoy)/dx
            = -g*(Br - Bl)/(rho0*dx)
            = -g*(-alpha0*dx*G_T*rho0)/(rho0*dx)
            = g*alpha0*G_T = const

    Trapezoidal integration (uniform layers h = depth/nlev = D):
        idpdx(nlev) = 0.5*D*C
        idpdx(k)    = (nlev - k + 0.5)*D*C   for k in 1..nlev
    where C = g*alpha0*G_T.
    """
    nlev = 10
    depth = 100.0
    state = _make_state(nlev=nlev, depth=depth)
    ds = _make_density(nlev=nlev)
    assert state.T is not None
    assert state.S is not None
    T0 = ds.T0
    S0 = ds.S0
    state.T[1 : nlev + 1] = T0
    state.S[1 : nlev + 1] = S0

    G_T = 1e-3  # K/m temperature gradient
    dtdx = np.full(nlev + 1, G_T)
    dsdx = np.zeros(nlev + 1)
    dtdy = np.zeros(nlev + 1)
    dsdy = np.zeros(nlev + 1)

    idpdx, idpdy = _make_outputs(nlev)

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_GRADIENTS,
        dsdx=dsdx,
        dtdx=dtdx,
        dsdy=dsdy,
        dtdy=dtdy,
    )

    assert state.h is not None
    D = state.h[1]  # uniform layer thickness = depth/nlev
    C = _GRAVITY * _ALPHA0 * G_T  # expected uniform buoyancy gradient

    expected = np.zeros(nlev + 1)
    for k in range(1, nlev + 1):
        expected[k] = (nlev - k + 0.5) * D * C

    np.testing.assert_allclose(
        idpdx[1 : nlev + 1],
        expected[1 : nlev + 1],
        rtol=1e-10,
        err_msg="Type 1 uniform T gradient: analytic trapezoidal mismatch",
    )


def test_type1_analytic_uniform_S_gradient_idpdy() -> None:
    """Uniform dS/dy with linear EOS → analytic trapezoidal integral.

    With linear EOS: get_rho(S+dSS, T, p) = rho0*(1 - alpha0*(T-T0) + beta0*(S+dSS-S0))
    dB/dy = -g/rho0 * (Br - Bl)/dy = -g/rho0 * rho0*beta0*G_S = -g*beta0*G_S

    idpdy(k) = (nlev - k + 0.5)*D*(-g*beta0*G_S)
    """
    nlev = 10
    depth = 100.0
    state = _make_state(nlev=nlev, depth=depth)
    ds = _make_density(nlev=nlev)
    assert state.T is not None
    assert state.S is not None
    state.T[1 : nlev + 1] = ds.T0
    state.S[1 : nlev + 1] = ds.S0

    G_S = 5e-4  # g/(kg·m) salinity gradient
    dsdy = np.full(nlev + 1, G_S)
    dsdx = np.zeros(nlev + 1)
    dtdx = np.zeros(nlev + 1)
    dtdy = np.zeros(nlev + 1)

    idpdx, idpdy = _make_outputs(nlev)

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_GRADIENTS,
        dsdx=dsdx,
        dtdx=dtdx,
        dsdy=dsdy,
        dtdy=dtdy,
    )

    assert state.h is not None
    D = state.h[1]
    C = (
        -_GRAVITY * _BETA0 * G_S
    )  # negative: salinity increases density → decreases buoyancy

    expected = np.zeros(nlev + 1)
    for k in range(1, nlev + 1):
        expected[k] = (nlev - k + 0.5) * D * C

    np.testing.assert_allclose(
        idpdy[1 : nlev + 1],
        expected[1 : nlev + 1],
        rtol=1e-10,
        err_msg="Type 1 uniform S gradient: analytic trapezoidal mismatch",
    )


# ---------------------------------------------------------------------------
# 5. Type 1 — x and y channels are independent
# ---------------------------------------------------------------------------


def test_type1_x_gradient_does_not_affect_idpdy() -> None:
    """Setting only dT/dx must leave idpdy = 0."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.T is not None
    assert state.S is not None
    state.T[1 : nlev + 1] = ds.T0
    state.S[1 : nlev + 1] = ds.S0

    dtdx = np.full(nlev + 1, 1e-3)
    dsdx = np.zeros(nlev + 1)
    dtdy = np.zeros(nlev + 1)
    dsdy = np.zeros(nlev + 1)

    idpdx, idpdy = _make_outputs(nlev)

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_GRADIENTS,
        dsdx=dsdx,
        dtdx=dtdx,
        dsdy=dsdy,
        dtdy=dtdy,
    )

    np.testing.assert_allclose(
        idpdy[1 : nlev + 1], 0.0, atol=1e-14, err_msg="dT/dx must not affect idpdy"
    )


def test_type1_y_gradient_does_not_affect_idpdx() -> None:
    """Setting only dT/dy must leave idpdx = 0."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.T is not None
    assert state.S is not None
    state.T[1 : nlev + 1] = ds.T0
    state.S[1 : nlev + 1] = ds.S0

    dtdy = np.full(nlev + 1, 1e-3)
    dsdx = np.zeros(nlev + 1)
    dtdx = np.zeros(nlev + 1)
    dsdy = np.zeros(nlev + 1)

    idpdx, idpdy = _make_outputs(nlev)

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_GRADIENTS,
        dsdx=dsdx,
        dtdx=dtdx,
        dsdy=dsdy,
        dtdy=dtdy,
    )

    np.testing.assert_allclose(
        idpdx[1 : nlev + 1], 0.0, atol=1e-14, err_msg="dT/dy must not affect idpdx"
    )


# ---------------------------------------------------------------------------
# 6. Type 2 — surface plume (plume_type=1)
# ---------------------------------------------------------------------------


def test_type2_surface_plume_zero_at_bottom() -> None:
    """Surface plume: idpdx(1) = slope_x*(buoy(1)-buoy(1)) = 0."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.buoy is not None
    # Linear buoyancy: 0 at k=1, 0.1 at k=nlev
    for k in range(1, nlev + 1):
        state.buoy[k] = 0.1 * (k - 1) / (nlev - 1)

    idpdx, idpdy = _make_outputs(nlev)
    slope_x = 1e-3
    slope_y = 0.0

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_PLUME,
        plume_type=PLUME_SURFACE,
        plume_slope_x=slope_x,
        plume_slope_y=slope_y,
    )

    assert idpdx[1] == pytest.approx(
        0.0, abs=1e-14
    ), "Surface plume: idpdx(1) must be zero"


def test_type2_surface_plume_maximum_at_surface() -> None:
    """Surface plume: idpdx(nlev) = slope_x*(buoy(nlev)-buoy(1))."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.buoy is not None
    buoy_bot = 0.0
    buoy_top = 0.1
    for k in range(1, nlev + 1):
        state.buoy[k] = buoy_bot + (buoy_top - buoy_bot) * (k - 1) / (nlev - 1)

    idpdx, idpdy = _make_outputs(nlev)
    slope_x = 2e-3

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_PLUME,
        plume_type=PLUME_SURFACE,
        plume_slope_x=slope_x,
        plume_slope_y=0.0,
    )

    expected_top = slope_x * (buoy_top - buoy_bot)
    assert idpdx[nlev] == pytest.approx(expected_top, rel=1e-12)


def test_type2_surface_plume_formula() -> None:
    """Surface plume: idpdx(i) = slope_x*(buoy(i)-buoy(1)) for all i."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.buoy is not None

    rng = np.random.default_rng(42)
    state.buoy[1 : nlev + 1] = rng.uniform(-0.05, 0.05, nlev)
    buoy_ref = state.buoy.copy()

    idpdx, idpdy = _make_outputs(nlev)
    slope_x = 1.5e-3

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_PLUME,
        plume_type=PLUME_SURFACE,
        plume_slope_x=slope_x,
        plume_slope_y=0.0,
    )

    expected = np.array(
        [slope_x * (buoy_ref[k] - buoy_ref[1]) for k in range(nlev + 1)]
    )
    np.testing.assert_allclose(
        idpdx[1 : nlev + 1],
        expected[1 : nlev + 1],
        rtol=1e-14,
        err_msg="Surface plume formula mismatch",
    )


# ---------------------------------------------------------------------------
# 7. Type 2 — bottom plume (plume_type=2)
# ---------------------------------------------------------------------------


def test_type2_bottom_plume_zero_at_surface() -> None:
    """Bottom plume: idpdx(nlev) = -slope_x*(buoy(nlev)-buoy(nlev)) = 0."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.buoy is not None
    for k in range(1, nlev + 1):
        state.buoy[k] = 0.1 * (k - 1) / (nlev - 1)

    idpdx, idpdy = _make_outputs(nlev)
    slope_x = 1e-3

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_PLUME,
        plume_type=PLUME_BOTTOM,
        plume_slope_x=slope_x,
        plume_slope_y=0.0,
    )

    assert idpdx[nlev] == pytest.approx(
        0.0, abs=1e-14
    ), "Bottom plume: idpdx(nlev) must be zero"


def test_type2_bottom_plume_formula() -> None:
    """Bottom plume: idpdx(i) = -slope_x*(buoy(nlev)-buoy(i)) for all i."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.buoy is not None

    rng = np.random.default_rng(7)
    state.buoy[1 : nlev + 1] = rng.uniform(-0.05, 0.05, nlev)
    buoy_ref = state.buoy.copy()

    idpdx, idpdy = _make_outputs(nlev)
    slope_x = 2e-3

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_PLUME,
        plume_type=PLUME_BOTTOM,
        plume_slope_x=slope_x,
        plume_slope_y=0.0,
    )

    expected = np.array(
        [-slope_x * (buoy_ref[nlev] - buoy_ref[k]) for k in range(nlev + 1)]
    )
    np.testing.assert_allclose(
        idpdx[1 : nlev + 1],
        expected[1 : nlev + 1],
        rtol=1e-14,
        err_msg="Bottom plume formula mismatch",
    )


# ---------------------------------------------------------------------------
# 8. Type 2 — both x and y slope components applied simultaneously
# ---------------------------------------------------------------------------


def test_type2_surface_plume_both_components() -> None:
    """Surface plume with non-zero slope_x and slope_y: both idpdx and idpdy updated."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.buoy is not None

    for k in range(1, nlev + 1):
        state.buoy[k] = 0.05 * k / nlev
    buoy_ref = state.buoy.copy()

    idpdx, idpdy = _make_outputs(nlev)
    slope_x = 1e-3
    slope_y = -2e-3

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_PLUME,
        plume_type=PLUME_SURFACE,
        plume_slope_x=slope_x,
        plume_slope_y=slope_y,
    )

    for k in range(1, nlev + 1):
        assert idpdx[k] == pytest.approx(
            slope_x * (buoy_ref[k] - buoy_ref[1]), rel=1e-12
        )
        assert idpdy[k] == pytest.approx(
            slope_y * (buoy_ref[k] - buoy_ref[1]), rel=1e-12
        )


# ---------------------------------------------------------------------------
# 9. No NaN or Inf for valid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "int_press_type,plume_type",
    [
        (INT_PRESS_NONE, 0),
        (INT_PRESS_GRADIENTS, 0),
        (INT_PRESS_PLUME, PLUME_SURFACE),
        (INT_PRESS_PLUME, PLUME_BOTTOM),
    ],
)
def test_no_nan_inf(int_press_type: int, plume_type: int) -> None:
    """No NaN or Inf in idpdx or idpdy for any valid (type, plume_type)."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.T is not None
    assert state.S is not None
    assert state.buoy is not None

    state.T[1 : nlev + 1] = 10.0
    state.S[1 : nlev + 1] = 35.0
    state.buoy[1 : nlev + 1] = np.linspace(-0.02, 0.02, nlev)

    idpdx, idpdy = _make_outputs(nlev)
    dsdx = np.full(nlev + 1, 1e-4)
    dtdx = np.full(nlev + 1, 1e-3)
    dsdy = np.full(nlev + 1, -5e-5)
    dtdy = np.full(nlev + 1, 5e-4)

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=int_press_type,
        dsdx=dsdx,
        dtdx=dtdx,
        dsdy=dsdy,
        dtdy=dtdy,
        plume_type=plume_type,
        plume_slope_x=1e-3,
        plume_slope_y=-1e-3,
    )

    assert np.all(np.isfinite(idpdx[1 : nlev + 1])), "NaN/Inf in idpdx"
    assert np.all(np.isfinite(idpdy[1 : nlev + 1])), "NaN/Inf in idpdy"


# ---------------------------------------------------------------------------
# 10. k=0 boundary level is never modified
# ---------------------------------------------------------------------------


def test_boundary_k0_not_modified() -> None:
    """idpdx[0] and idpdy[0] must not be touched (unused face below seabed)."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    assert state.T is not None
    assert state.S is not None
    assert state.buoy is not None

    state.T[1 : nlev + 1] = 10.0
    state.S[1 : nlev + 1] = 35.0
    state.buoy[1 : nlev + 1] = np.linspace(0.0, 0.05, nlev)

    idpdx, idpdy = _make_outputs(nlev)
    idpdx[0] = 99.0
    idpdy[0] = 88.0

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
        int_press_type=INT_PRESS_GRADIENTS,
        dsdx=np.full(nlev + 1, 1e-4),
        dtdx=np.full(nlev + 1, 1e-3),
        dsdy=np.zeros(nlev + 1),
        dtdy=np.zeros(nlev + 1),
    )

    assert idpdx[0] == 99.0, "idpdx[0] must not be modified"
    assert idpdy[0] == 88.0, "idpdy[0] must not be modified"


# ---------------------------------------------------------------------------
# 11. Smoke test — default call (all zeros, type 0)
# ---------------------------------------------------------------------------


def test_smoke_default_call() -> None:
    """internal_pressure() with default arguments must not raise."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)
    idpdx, idpdy = _make_outputs(nlev)

    internal_pressure(
        state=state,
        density=ds,
        nlev=nlev,
        idpdx=idpdx,
        idpdy=idpdy,
    )  # int_press_type defaults to INT_PRESS_NONE
