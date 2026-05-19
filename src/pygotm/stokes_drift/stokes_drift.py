# ruff: noqa: E501
"""
Stokes drift — translation of ``stokes_drift.F90``.

Provides subroutines to compute Stokes drift profiles from various input
sources.  The active method is selected per-component via integer flags:

* ``NOTHING`` (0) — Stokes drift disabled.
* ``CONSTANT`` (1) — constant surface Stokes drift ``us0``/``vs0``.
* ``FROMFILE`` (2) — profile or scalar read from file.
* ``EXPONENTIAL`` (3) — exponential profile from surface Stokes drift and
  penetration depth ``ds`` (calls :func:`~pygotm.stokes_drift.stokes_drift_exp.stokes_drift_exp`).
* ``THEORYWAVE`` (4) — Li et al. (2017) empirical theory-wave spectrum
  (calls :func:`~pygotm.stokes_drift.stokes_drift_theory.stokes_drift_theory`).

Fortran public data members — ``usprof``, ``vsprof``, ``dusdz``, ``dvsdz``
(profile inputs), ``us0``, ``vs0``, ``ds``, ``uwnd``, ``vwnd`` (scalar inputs),
``La_Turb`` (McWilliams et al., 1997), ``La_SL`` (Harcourt and D'Asaro, 2008),
``La_SLP_VR12`` (Van Roekel et al., 2012), ``La_SLP_RWH16`` (Reichl et al.,
2016), ``EFactor_LWF16`` (Li et al., 2016), ``EFactor_RWH16`` (Reichl et al.,
2016), ``theta_WW``, ``theta_WL`` — are consolidated in :class:`StokesDriftState`.

Public interface: :func:`init_stokes_drift`, :func:`init_stokes_drift_yaml`,
:func:`post_init_stokes_drift`, :func:`do_stokes_drift`,
:func:`langmuir_number`, :func:`clean_stokes_drift`, :class:`StokesDriftState`.

Original authors: Qing Li.
"""

import math
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np

from pygotm.input.input import ProfileInput, ScalarInput
from pygotm.stokes_drift.stokes_drift_exp import stokes_drift_exp
from pygotm.stokes_drift.stokes_drift_theory import stokes_drift_theory

__all__ = [
    "CONSTANT",
    "EXPONENTIAL",
    "FROMFILE",
    "FROMUS",
    "NOTHING",
    "THEORYWAVE",
    "StokesDriftState",
    "clean_stokes_drift",
    "do_stokes_drift",
    "init_stokes_drift",
    "init_stokes_drift_yaml",
    "langmuir_number",
    "post_init_stokes_drift",
]

NOTHING: int = 0
CONSTANT: int = 1
FROMFILE: int = 2
EXPONENTIAL: int = 3
THEORYWAVE: int = 4
FROMUS: int = 3

_SMALL: float = 1.0e-12
_LARGE: float = 1.0 / _SMALL
_KAPPA: float = 0.4


@dataclass
class StokesDriftState:
    """State owned by the translated ``stokes_drift`` module."""

    usprof_method: int = NOTHING
    vsprof_method: int = NOTHING
    dusdz_method: int = NOTHING
    dvsdz_method: int = NOTHING
    uwnd_method: int = NOTHING
    vwnd_method: int = NOTHING
    us0_method: int = NOTHING
    vs0_method: int = NOTHING
    ds_method: int = NOTHING

    us0: float = 0.0
    vs0: float = 0.0
    ds: float = 5.0
    uwnd: float = 0.0
    vwnd: float = 0.0

    usprof: np.ndarray | None = None
    vsprof: np.ndarray | None = None
    dusdz: np.ndarray | None = None
    dvsdz: np.ndarray | None = None
    stokes_srf: np.ndarray | None = None

    us0_input: ScalarInput | None = None
    vs0_input: ScalarInput | None = None
    ds_input: ScalarInput | None = None
    uwnd_input: ScalarInput | None = None
    vwnd_input: ScalarInput | None = None
    usprof_input: ProfileInput | None = None
    vsprof_input: ProfileInput | None = None
    dusdz_input: ProfileInput | None = None
    dvsdz_input: ProfileInput | None = None

    La_Turb: float = _LARGE
    La_SL: float = _LARGE
    La_SLP_VR12: float = _LARGE
    La_SLP_RWH16: float = _LARGE
    EFactor_LWF16: float = 1.0
    EFactor_RWH16: float = 1.0
    theta_WW: float = 0.0
    theta_WL: float = 0.0


def init_stokes_drift(state: StokesDriftState, **overrides: int | float) -> None:
    """Initialise Stokes drift method selectors and scalar defaults."""

    for name, value in overrides.items():
        if not hasattr(state, name):
            msg = f"unknown Stokes drift override {name!r}"
            raise AttributeError(msg)
        setattr(state, name, value)


def _method_from_token(value: object, default: int = NOTHING) -> int:
    text = "" if value is None else str(value).strip().lower().replace("-", "_")
    if text in {"", "off", "none"}:
        return NOTHING
    if text == "constant":
        return CONSTANT
    if text in {"file", "from_file"}:
        return FROMFILE
    if text == "exponential":
        return EXPONENTIAL
    if text in {"empirical", "theorywave", "theory_wave"}:
        return THEORYWAVE
    if text in {"us", "vs", "fromus", "from_us"}:
        return FROMUS
    return default


def init_stokes_drift_yaml(
    state: StokesDriftState,
    settings: dict[str, Any] | None = None,
) -> None:
    """Apply GOTM YAML-style Stokes drift method selectors.

    The driver owns input registration. This helper mirrors the Fortran method
    selection rules and is useful for tests and non-driver callers.
    """

    raw = {} if settings is None else settings
    state.usprof_method = _method_from_token(
        dict(raw.get("us", {})).get("method")
        if isinstance(raw.get("us"), dict)
        else None
    )
    state.vsprof_method = _method_from_token(
        dict(raw.get("vs", {})).get("method")
        if isinstance(raw.get("vs"), dict)
        else None
    )
    state.dusdz_method = _method_from_token(
        dict(raw.get("dusdz", {})).get("method")
        if isinstance(raw.get("dusdz"), dict)
        else None
    )
    state.dvsdz_method = _method_from_token(
        dict(raw.get("dvsdz", {})).get("method")
        if isinstance(raw.get("dvsdz"), dict)
        else None
    )


def post_init_stokes_drift(state: StokesDriftState, nlev: int) -> None:
    """Allocate memory and initialise Stokes drift diagnostics."""

    state.usprof = np.zeros(nlev + 1, dtype=np.float64)
    state.vsprof = np.zeros(nlev + 1, dtype=np.float64)
    state.dusdz = np.zeros(nlev + 1, dtype=np.float64)
    state.dvsdz = np.zeros(nlev + 1, dtype=np.float64)
    state.stokes_srf = np.zeros(nlev + 1, dtype=np.float64)
    state.La_Turb = _LARGE
    state.La_SL = _LARGE
    state.La_SLP_VR12 = _LARGE
    state.La_SLP_RWH16 = _LARGE
    state.theta_WW = 0.0
    state.theta_WL = 0.0
    state.EFactor_LWF16 = 1.0
    state.EFactor_RWH16 = 1.0


def _sync_registered_inputs(state: StokesDriftState) -> None:
    if state.us0_input is not None:
        state.us0 = float(state.us0_input.value)
    if state.vs0_input is not None:
        state.vs0 = float(state.vs0_input.value)
    if state.ds_input is not None:
        state.ds = float(state.ds_input.value)
    if state.uwnd_input is not None:
        state.uwnd = float(state.uwnd_input.value)
    if state.vwnd_input is not None:
        state.vwnd = float(state.vwnd_input.value)

    if state.usprof_input is not None and state.usprof_input.data is not None:
        assert state.usprof is not None
        state.usprof[:] = state.usprof_input.data
    if state.vsprof_input is not None and state.vsprof_input.data is not None:
        assert state.vsprof is not None
        state.vsprof[:] = state.vsprof_input.data
    if state.dusdz_input is not None and state.dusdz_input.data is not None:
        assert state.dusdz is not None
        state.dusdz[:] = state.dusdz_input.data
    if state.dvsdz_input is not None and state.dvsdz_input.data is not None:
        assert state.dvsdz is not None
        state.dvsdz[:] = state.dvsdz_input.data


@numba.njit(cache=True)
def _compute_stokes_shear(
    nlev: int,
    z: np.ndarray,
    usprof: np.ndarray,
    vsprof: np.ndarray,
    dusdz: np.ndarray,
    dvsdz: np.ndarray,
) -> None:
    for k in range(1, nlev):
        dz = z[k + 1] - z[k]
        dusdz[k] = (usprof[k + 1] - usprof[k]) / dz
        dvsdz[k] = (vsprof[k + 1] - vsprof[k]) / dz
    if nlev > 1:
        dusdz[0] = dusdz[1]
        dusdz[nlev] = dusdz[nlev - 1]
        dvsdz[0] = dvsdz[1]
        dvsdz[nlev] = dvsdz[nlev - 1]


def do_stokes_drift(
    state: StokesDriftState,
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    gravity: float,
    u10: float,
    v10: float,
) -> None:
    """Wrapper for all subroutines that calculate the Stokes drift profile."""

    assert state.usprof is not None
    assert state.vsprof is not None
    assert state.dusdz is not None
    assert state.dvsdz is not None
    assert state.stokes_srf is not None

    _sync_registered_inputs(state)

    profile_method = (
        state.usprof_method if state.usprof_method != NOTHING else state.vsprof_method
    )

    if profile_method == EXPONENTIAL:
        stokes_drift_exp(
            nlev,
            z,
            zi,
            state.us0,
            state.vs0,
            state.ds,
            state.usprof,
            state.vsprof,
        )
    elif profile_method == THEORYWAVE:
        wind_u = u10 if state.uwnd_method == NOTHING else state.uwnd
        wind_v = v10 if state.vwnd_method == NOTHING else state.vwnd
        state.us0, state.vs0, state.ds = stokes_drift_theory(
            nlev,
            z,
            zi,
            wind_u,
            wind_v,
            gravity,
            state.stokes_srf,
            state.usprof,
            state.vsprof,
        )
    else:
        ustran = 0.0
        for k in range(1, nlev + 1):
            ustran += math.sqrt(state.usprof[k] ** 2 + state.vsprof[k] ** 2) * (
                zi[k] - zi[k - 1]
            )
        denom = max(_SMALL, math.sqrt(state.us0**2 + state.vs0**2))
        state.ds = ustran / denom

    if state.dusdz_method == FROMUS or state.dvsdz_method == FROMUS:
        _compute_stokes_shear(
            nlev,
            z,
            state.usprof,
            state.vsprof,
            state.dusdz,
            state.dvsdz,
        )


@numba.njit(cache=True)
def _langmuir_number_kernel(
    nlev: int,
    zi: np.ndarray,
    usprof: np.ndarray,
    vsprof: np.ndarray,
    us0: float,
    vs0: float,
    hsw: float,
    u_taus: float,
    hbl: float,
    u10: float,
    v10: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    us_srf = math.sqrt(us0 * us0 + vs0 * vs0)
    hsl = 0.2 * hbl

    ksl = 1
    for k in range(nlev, 0, -1):
        if zi[nlev] - zi[k - 1] >= hsl:
            ksl = k
            break

    kbl = 1
    for k in range(nlev, 0, -1):
        if zi[nlev] - zi[k - 1] >= hbl:
            kbl = k
            break

    if ksl < nlev:
        ussl = usprof[ksl] * (hsl + zi[ksl])
        vssl = vsprof[ksl] * (hsl + zi[ksl])
        for k in range(nlev, ksl, -1):
            dz = zi[k] - zi[k - 1]
            ussl += usprof[k] * dz
            vssl += vsprof[k] * dz
        ussl /= hsl
        vssl /= hsl
    else:
        ussl = usprof[nlev]
        vssl = vsprof[nlev]

    if us_srf > 1.0e-4 and u_taus > 1.0e-4:
        la_turb = math.sqrt(u_taus / us_srf)
        la_sl = math.sqrt(
            u_taus
            / abs(
                math.sqrt(ussl * ussl + vssl * vssl)
                - math.sqrt(usprof[kbl] ** 2 + vsprof[kbl] ** 2)
                + _SMALL * _SMALL
            )
        )
        theta_ww = math.atan2(vssl, ussl) - math.atan2(v10, u10)
        z0 = max(0.02, hsw) * 4.0
        theta_wl = math.atan(
            math.sin(theta_ww)
            / (
                u_taus / us_srf / _KAPPA * math.log(max(hbl / z0, 1.0))
                + math.cos(theta_ww)
            )
        )
        la_slp_vr12 = la_sl * math.sqrt(
            abs(math.cos(theta_wl)) / (abs(math.cos(theta_ww - theta_wl)) + _SMALL)
        )
        la_slp_rwh16 = la_sl * math.sqrt(
            1.0 / (abs(math.cos(theta_ww - theta_wl)) + _SMALL)
        )
    else:
        la_turb = _LARGE
        la_sl = _LARGE
        la_slp_vr12 = _LARGE
        la_slp_rwh16 = _LARGE
        theta_ww = 0.0
        theta_wl = 0.0

    efactor_lwf16 = min(
        2.0,
        abs(math.cos(theta_wl))
        * math.sqrt(
            1.0 + (1.5 * la_slp_vr12) ** (-2.0) + (5.4 * la_slp_vr12) ** (-4.0)
        ),
    )
    efactor_rwh16 = min(2.25, 1.0 + 1.0 / la_slp_rwh16)
    return (
        la_turb,
        la_sl,
        la_slp_vr12,
        la_slp_rwh16,
        theta_ww,
        theta_wl,
        efactor_lwf16,
        efactor_rwh16,
    )


def langmuir_number(
    state: StokesDriftState,
    nlev: int,
    zi: np.ndarray,
    hsw: float,
    u_taus: float,
    hbl: float,
    u10: float,
    v10: float,
) -> None:
    """Compute Langmuir numbers and enhancement factors from Stokes drift."""

    assert state.usprof is not None
    assert state.vsprof is not None
    (
        state.La_Turb,
        state.La_SL,
        state.La_SLP_VR12,
        state.La_SLP_RWH16,
        state.theta_WW,
        state.theta_WL,
        state.EFactor_LWF16,
        state.EFactor_RWH16,
    ) = _langmuir_number_kernel(
        nlev,
        zi,
        state.usprof,
        state.vsprof,
        state.us0,
        state.vs0,
        hsw,
        u_taus,
        hbl,
        u10,
        v10,
    )


def clean_stokes_drift(state: StokesDriftState) -> None:
    """Release Stokes drift arrays and registered input handles."""

    state.usprof = None
    state.vsprof = None
    state.dusdz = None
    state.dvsdz = None
    state.stokes_srf = None
