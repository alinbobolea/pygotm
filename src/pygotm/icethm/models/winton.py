"""Winton three-layer sea-ice thermodynamics.

This module follows GOTM/STIM's ``stim_winton.F90`` and
``winton/ice_thm.F90`` implementation. The model uses a zero heat-capacity snow
layer over two sea-ice layers; the upper layer has the Winton brine heat-capacity
term and the lower layer has fixed heat capacity.
"""

import math

import numba
import numpy as np

from pygotm.icethm._util import freezing_temperature
from pygotm.icethm.constants import OPT_DEP_ICE

_DI: float = 905.0
_DS: float = 330.0
_DW: float = 1030.0
_KI: float = 2.03
_KS: float = 0.31
_CI: float = 2100.0
_LI: float = 334_000.0
_TFI: float = -0.054
_ALB_SNO: float = 0.85
_ALB_ICE: float = 0.5826
_THIN_ICE_ALB: float = 0.06
_PEN_ICE: float = 0.3
_T_RANGE_MELT: float = 1.0
_H_LO_LIM: float = 0.0
_STIM_RHO_ICE: float = 910.0
_STIM_L_ICE: float = 333_500.0
_STIM_CW: float = 4.18e6


@numba.njit(cache=True)
def ice_optics(
    hice: float,
    hsnow: float,
    Ts: float,
    albedo_ice: np.ndarray,
    transmissivity: np.ndarray,
) -> float:
    """Update Winton albedo/transmissivity and return penetrating solar fraction."""

    if hice <= 0.0:
        albedo_ice[0] = 0.0
        transmissivity[0] = 1.0
        return 0.0

    snow_cover = hsnow / (hsnow + 0.02)
    snow_albedo = _ALB_SNO
    ice_albedo = _ALB_ICE
    f_h = min(math.atan(5.0 * hice) / math.atan(2.5), 1.0)
    if Ts + _T_RANGE_MELT > _TFI:
        melt_factor = min((Ts + _T_RANGE_MELT - _TFI) / _T_RANGE_MELT, 1.0)
        snow_albedo -= 0.1235 * melt_factor
        ice_albedo -= 0.075 * melt_factor
    ice_albedo = f_h * ice_albedo + (1.0 - f_h) * _THIN_ICE_ALB
    albedo_ice[0] = snow_cover * snow_albedo + (1.0 - snow_cover) * ice_albedo
    transmissivity[0] = math.exp(-hice / OPT_DEP_ICE)
    return (1.0 - snow_cover) * _PEN_ICE


@numba.njit(cache=True)
def _e_to_melt_hs(hs: float) -> float:
    return _DS * _LI * hs


@numba.njit(cache=True)
def _e_to_melt_h1(h1: float, t1: float) -> float:
    return _DI * h1 * (_CI - _LI / t1) * (_TFI - t1)


@numba.njit(cache=True)
def _e_to_melt_h2(h2: float, t2: float) -> float:
    return _DI * h2 * (_LI + _CI * (_TFI - t2))


@numba.njit(cache=True)
def _add_to_top(h: float, t: float, h1: float, t1: float) -> tuple[float, float]:
    if h <= 0.0:
        return h1, t1
    f1 = h1 / (h1 + h)
    mixed = f1 * (t1 + _LI * _TFI / (_CI * t1)) + (1.0 - f1) * t
    t1 = (mixed - math.sqrt(mixed * mixed - 4.0 * _TFI * _LI / _CI)) / 2.0
    h1 += h
    return h1, t1


@numba.njit(cache=True)
def _add_to_bot(h: float, t: float, h2: float, t2: float) -> tuple[float, float]:
    if h <= 0.0:
        return h2, t2
    t2 = (h2 * t2 + h * t) / (h2 + h)
    h2 += h
    return h2, t2


@numba.njit(cache=True)
def _even_up(
    h1: float,
    t1: float,
    h2: float,
    t2: float,
) -> tuple[float, float, float, float]:
    half_total = (h1 + h2) / 2.0
    if h1 > half_total:
        h2, t2 = _add_to_bot(h1 - half_total, t1 + _LI * _TFI / (_CI * t1), h2, t2)
        h1 = h2
    elif h2 > half_total:
        h1, t1 = _add_to_top(h2 - half_total, t2, h1, t1)
        h2 = h1
    if t2 > _TFI:
        denom = _LI * t1 + (_CI * t1 - _LI) * (_TFI - t1)
        dh = h2 * _CI * (t2 - _TFI) * t1 / denom
        t2 = _TFI
        h1 -= dh
        h2 -= dh
    return h1, t1, h2, t2


@numba.njit(cache=True)
def ice3lay_temp(
    Tf: float,
    dt: float,
    A_flux: float,
    B_flux: float,
    I_absorbed: float,
    Hice: np.ndarray,
    Hsnow: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    Tice_surface: np.ndarray,
    surface_energy: np.ndarray,
    bottom_energy: np.ndarray,
    ocean_ice_heat_flux: float,
) -> None:
    """Advance ice temperatures using the GOTM/STIM Winton equations."""

    hice = Hice[0]
    if hice <= 0.0 or dt <= 0.0:
        return

    hsnow = Hsnow[0]
    tsf = 0.0 if hsnow > 0.0 else _TFI
    hie = max(hice, _H_LO_LIM)
    k12 = 4.0 * _KI * _KS / (_KS + 4.0 * _KI * hsnow / hie)
    hi2 = hie * hie
    denom_lower = 6.0 * dt * 2.0 * _KI + _DI * hi2 * _CI
    a10 = (
        _DI * hi2 * _CI / (2.0 * dt)
        + 2.0 * _KI * (4.0 * dt * 2.0 * _KI + _DI * hi2 * _CI) / denom_lower
    )
    b10 = (
        -_DI * hi2 * (_CI * T1[0] + _LI * _TFI / T1[0]) / (2.0 * dt)
        - I_absorbed * hie
        - 2.0
        * _KI
        * (4.0 * dt * 2.0 * _KI * Tf + _DI * hi2 * _CI * T2[0])
        / denom_lower
    )

    a1 = a10 + k12 * B_flux * hie / (k12 + B_flux * hie)
    b1 = b10 + A_flux * k12 * hie / (k12 + B_flux * hie)
    c1 = _DI * hi2 * _LI * _TFI / (2.0 * dt)
    disc = b1 * b1 - 4.0 * a1 * c1
    if disc < 0.0:
        disc = 0.0
    T1[0] = -(math.sqrt(disc) + b1) / (2.0 * a1)
    Ts = (k12 * T1[0] - A_flux * hie) / (k12 + B_flux * hie)

    if Ts > _TFI:
        a1 = a10 + k12
        b1 = b10 - k12 * tsf
        disc = b1 * b1 - 4.0 * a1 * c1
        if disc < 0.0:
            disc = 0.0
        T1[0] = -(math.sqrt(disc) + b1) / (2.0 * a1)
        Ts = tsf
        surface_energy[0] += (k12 * (T1[0] - Ts) / hie - (A_flux + B_flux * Ts)) * dt

    T1[0] -= Tf
    T2[0] -= Tf
    T2[0] = (2.0 * dt * 2.0 * _KI * T1[0] + _DI * hi2 * _CI * T2[0]) / denom_lower
    T1[0] += Tf
    T2[0] += Tf
    bottom_energy[0] += (ocean_ice_heat_flux + 4.0 * _KI * (T2[0] - Tf) / hie) * dt

    if T2[0] > _TFI:
        bottom_energy[0] += _e_to_melt_h2(hie / 2.0, _TFI) - _e_to_melt_h2(
            hie / 2.0, T2[0]
        )
        T2[0] = _TFI
    if T1[0] > _TFI:
        surface_energy[0] += _e_to_melt_h1(hie / 2.0, _TFI) - _e_to_melt_h1(
            hie / 2.0, T1[0]
        )
        T1[0] = _TFI
    Tice_surface[0] = Ts


@numba.njit(cache=True)
def ice3lay_resize(
    Tf: float,
    snow: float,
    frazil: float,
    evap: float,
    Hice: np.ndarray,
    Hsnow: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    surface_energy: np.ndarray,
    bottom_energy: np.ndarray,
    ocean_ice_flux: np.ndarray,
) -> None:
    """Resize snow/ice layers using GOTM/STIM's ``ice3lay_resize`` equations."""

    hs = Hsnow[0]
    hi = Hice[0]
    t1 = T1[0]
    t2 = T2[0]
    h2o_to_ocn = _DS * hs + _DI * hi + snow - evap
    h2o_from_ocn = 0.0

    h1 = hi / 2.0
    h2 = hi / 2.0
    hs += snow / _DS
    h2, t2 = _add_to_bot(frazil / _e_to_melt_h2(1.0, Tf), Tf, h2, t2)

    if bottom_energy[0] < 0.0:
        h2, t2 = _add_to_bot(-bottom_energy[0] / _e_to_melt_h2(1.0, Tf), Tf, h2, t2)
    if h1 == 0.0:
        t1 = Tf

    snow_energy = _e_to_melt_hs(hs)
    upper_energy = snow_energy + _e_to_melt_h1(h1, t1)
    total_energy = upper_energy + _e_to_melt_h2(h2, t2)
    if surface_energy[0] <= snow_energy:
        pass
    elif surface_energy[0] <= upper_energy:
        h1 -= (surface_energy[0] - snow_energy) / _e_to_melt_h1(1.0, t1)
        hs = 0.0
    elif surface_energy[0] <= total_energy:
        h2 -= (surface_energy[0] - upper_energy) / _e_to_melt_h2(1.0, t2)
        hs = 0.0
        h1 = 0.0
    else:
        hs = 0.0
        h1 = 0.0
        h2 = 0.0

    if bottom_energy[0] > 0.0:
        lower_energy = _e_to_melt_h2(h2, t2)
        ice_energy = _e_to_melt_h1(h1, t1) + lower_energy
        all_energy = _e_to_melt_hs(hs) + ice_energy
        if bottom_energy[0] < lower_energy:
            h2 -= bottom_energy[0] / _e_to_melt_h2(1.0, t2)
        elif bottom_energy[0] < ice_energy:
            h1 -= (bottom_energy[0] - lower_energy) / _e_to_melt_h1(1.0, t1)
            h2 = 0.0
        elif bottom_energy[0] < all_energy:
            h1 = 0.0
            h2 = 0.0
        else:
            hs = 0.0
            h1 = 0.0
            h2 = 0.0

    hi = h1 + h2
    hw = (_DI * hi + _DS * hs) / _DW
    if hw > hi:
        h1, t1 = _add_to_top(hw - hi, _TFI, h1, t1)

    h1, t1, h2, t2 = _even_up(h1, t1, h2, t2)
    hi = h1 + h2
    if hi == 0.0:
        t1 = 0.0
        t2 = 0.0

    h2o_to_ocn = h2o_to_ocn + h2o_from_ocn - _DS * hs - _DI * hi
    Hice[0] = max(0.0, hi)
    Hsnow[0] = hs
    T1[0] = t1
    T2[0] = t2
    ocean_ice_flux[0] = h2o_to_ocn


@numba.njit(cache=True)
def step_winton(
    T_w: float,
    S_w: float,
    h_sfc: float,
    dt: float,
    Qsw: float,
    Ql: float,
    Qh: float,
    Qe: float,
    precip: float,
    surface_flux_a: float,
    surface_flux_b: float,
    Hice: np.ndarray,
    Hsnow: np.ndarray,
    Hfrazil: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    Tice_surface: np.ndarray,
    ice_cover: np.ndarray,
    albedo_ice: np.ndarray,
    transmissivity: np.ndarray,
    Tf: np.ndarray,
    ocean_ice_heat_flux: np.ndarray,
    ocean_ice_flux: np.ndarray,
    ocean_ice_salt_flux: np.ndarray,
    surface_ice_energy: np.ndarray,
    bottom_ice_energy: np.ndarray,
) -> None:
    """Advance Winton three-layer ice state by one timestep."""

    _ = precip
    tf = freezing_temperature(S_w)
    Tf[0] = tf
    surface_ice_energy[0] = 0.0
    bottom_ice_energy[0] = 0.0
    frazil = 0.0
    Hfrazil[0] = 0.0

    if ice_cover[0] > 0:
        pen = ice_optics(Hice[0], Hsnow[0], Tice_surface[0], albedo_ice, transmissivity)
        absorbed_shortwave = Qsw * (1.0 - transmissivity[0])
        if surface_flux_a == 0.0 and surface_flux_b == 0.0:
            surface_flux_a = -(Qe + Qh + Ql)
            surface_flux_b = -4.0
        A_flux = surface_flux_a - absorbed_shortwave - Tice_surface[0] * surface_flux_b
        ice3lay_temp(
            tf,
            dt,
            A_flux,
            surface_flux_b,
            pen * absorbed_shortwave,
            Hice,
            Hsnow,
            T1,
            T2,
            Tice_surface,
            surface_ice_energy,
            bottom_ice_energy,
            ocean_ice_heat_flux[0],
        )
    else:
        frazil = -(T_w - tf) * h_sfc * _STIM_CW
        if frazil > 0.0:
            Hfrazil[0] = frazil / (_STIM_RHO_ICE * _STIM_L_ICE)

    if ice_cover[0] > 0 or frazil > 0.0:
        ice3lay_resize(
            tf,
            0.0,
            frazil,
            0.0,
            Hice,
            Hsnow,
            T1,
            T2,
            surface_ice_energy,
            bottom_ice_energy,
            ocean_ice_flux,
        )

    Hsnow[0] = 0.0
    if Hice[0] > 0.0:
        ice_cover[0] = 2
    else:
        ice_cover[0] = 0
    ocean_ice_salt_flux[0] = 0.0
