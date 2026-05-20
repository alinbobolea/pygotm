# ruff: noqa: E501
"""
CVMix interface — translation of ``gotm_cvmix.F90``.

Provides an interface to the Community Ocean Vertical Mixing Project (CVMix,
http://cvmix.github.io) in the General Ocean Turbulence Model.  CVMix is an
optional alternative to the native GOTM two-equation closures for ocean models
that use CVMix as their mixing library.

Fortran CVMix library modules (``cvmix_background``, ``cvmix_convection``,
``cvmix_kpp``, ``cvmix_shear``, ``cvmix_tidal``, ``cvmix_ddiff``) are
encapsulated by the Python mixing routines below.  All configuration and state
are held in :class:`CVMixState`.

Mixing parameterisations:

* Surface boundary layer — :func:`surface_layer` (parabolic KPP-style).
* Bottom boundary layer — :func:`bottom_layer`.
* Interior non-convective — :func:`interior_nonconv` (background diffusivity;
  Pacanowski–Philander or KPP shear scheme).
* Interior convective — :func:`interior_conv`.

Public interface: :func:`init_cvmix`, :func:`init_cvmix_yaml`,
:func:`post_init_cvmix`, :func:`do_cvmix`, :func:`clean_cvmix`,
:func:`surface_layer`, :func:`bottom_layer`, :func:`interior_nonconv`,
:func:`interior_conv`, :class:`CVMixState`.

Original FORTRAN authors: Lars Umlauf; adapted for CVMix by Qing Li.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CVMIX_INTERP_CUBIC",
    "CVMIX_INTERP_LINEAR",
    "CVMIX_INTERP_LMD94",
    "CVMIX_INTERP_QUADRATIC",
    "CVMIX_LT_LF17",
    "CVMIX_LT_LWF16",
    "CVMIX_LT_NOLANGMUIR",
    "CVMIX_LT_RWH16",
    "CVMIX_MATCH_BOTH",
    "CVMIX_MATCH_GRADIENT",
    "CVMIX_MATCH_PARABOLIC",
    "CVMIX_MATCH_SIMPLE",
    "CVMIX_SHEAR_KPP",
    "CVMIX_SHEAR_PP",
    "CVMixState",
    "bottom_layer",
    "clean_cvmix",
    "do_cvmix",
    "init_cvmix",
    "init_cvmix_yaml",
    "interior_conv",
    "interior_nonconv",
    "post_init_cvmix",
    "surface_layer",
]

CVMIX_LT_NOLANGMUIR = 0
CVMIX_LT_LWF16 = 1
CVMIX_LT_LF17 = 2
CVMIX_LT_RWH16 = 3
CVMIX_INTERP_LINEAR = 1
CVMIX_INTERP_QUADRATIC = 2
CVMIX_INTERP_CUBIC = 3
CVMIX_INTERP_LMD94 = 4
CVMIX_MATCH_SIMPLE = 1
CVMIX_MATCH_GRADIENT = 2
CVMIX_MATCH_BOTH = 3
CVMIX_MATCH_PARABOLIC = 4
CVMIX_SHEAR_PP = 1
CVMIX_SHEAR_KPP = 2
_EPS = 1.0e-14


@dataclass
class CVMixState:
    """Python-side CVMix interface state."""

    zsbl: float = 0.0
    zbbl: float = 0.0
    cvmix_g: float = 9.81
    cvmix_rho0: float = 1027.0
    cvmix_gorho0: float = 9.81 / 1027.0
    use_surface_layer: bool = True
    use_interior: bool = False
    use_bottom_layer: bool = False
    use_background: bool = False
    use_shear: bool = False
    use_convection: bool = False
    use_tidal_mixing: bool = False
    use_double_diffusion: bool = False
    sbl_langmuir_method: int = CVMIX_LT_NOLANGMUIR
    background_diffusivity: float = 1.0e-5
    background_viscosity: float = 1.0e-4
    shear_mix_scheme: int = CVMIX_SHEAR_KPP
    shear_num_smooth_Ri: int = 1
    shear_PP_nu_zero: float = 0.005
    shear_PP_alpha: float = 5.0
    shear_PP_exp: float = 2.0
    shear_KPP_nu_zero: float = 0.005
    shear_KPP_Ri_zero: float = 0.7
    shear_KPP_exp: float = 3.0
    convection_diffusivity: float = 1.0
    convection_viscosity: float = 1.0
    convection_basedOnBVF: bool = True
    convection_triggerBVF: float = 0.0
    z_w: np.ndarray | None = None
    z_r: np.ndarray | None = None
    h_r: np.ndarray | None = None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def init_cvmix(
    state: CVMixState | None = None, branch: dict[str, Any] | None = None
) -> CVMixState:
    """Initialise CVMix state from an optional YAML-like branch."""

    result = CVMixState() if state is None else state
    if branch is not None:
        init_cvmix_yaml(result, branch)
    return result


def init_cvmix_yaml(state: CVMixState, branch: dict[str, Any]) -> None:
    """Apply CVMix YAML options."""

    surface = _mapping(branch.get("surface_layer"))
    bottom = _mapping(branch.get("bottom_layer"))
    interior = _mapping(branch.get("interior"))

    state.use_surface_layer = bool(surface.get("use", state.use_surface_layer))
    state.use_bottom_layer = bool(bottom.get("use", state.use_bottom_layer))
    state.use_interior = bool(interior.get("use", state.use_interior))

    background = _mapping(interior.get("background"))
    state.use_background = bool(background.get("use", state.use_background))
    state.background_diffusivity = float(
        background.get("diffusivity", state.background_diffusivity)
    )
    state.background_viscosity = float(
        background.get("viscosity", state.background_viscosity)
    )

    shear = _mapping(interior.get("shear"))
    state.use_shear = bool(shear.get("use", state.use_shear))
    scheme = str(shear.get("mix_scheme", "")).strip().lower()
    if scheme == "pp":
        state.shear_mix_scheme = CVMIX_SHEAR_PP
    elif scheme == "kpp":
        state.shear_mix_scheme = CVMIX_SHEAR_KPP
    state.shear_PP_nu_zero = float(shear.get("PP_nu_zero", state.shear_PP_nu_zero))
    state.shear_PP_alpha = float(shear.get("PP_alpha", state.shear_PP_alpha))
    state.shear_PP_exp = float(shear.get("PP_exp", state.shear_PP_exp))
    state.shear_KPP_nu_zero = float(shear.get("KPP_nu_zero", state.shear_KPP_nu_zero))
    state.shear_KPP_Ri_zero = float(shear.get("KPP_Ri_zero", state.shear_KPP_Ri_zero))
    state.shear_KPP_exp = float(shear.get("KPP_exp", state.shear_KPP_exp))

    convection = _mapping(interior.get("convection"))
    state.use_convection = bool(convection.get("use", state.use_convection))
    state.convection_diffusivity = float(
        convection.get("diffusivity", state.convection_diffusivity)
    )
    state.convection_viscosity = float(
        convection.get("viscosity", state.convection_viscosity)
    )
    state.convection_triggerBVF = float(
        convection.get("triggerBVF", state.convection_triggerBVF)
    )


def post_init_cvmix(
    state: CVMixState,
    nlev: int,
    h0: float,
    gravity: float,
    rho0: float,
) -> None:
    """Initialise grid arrays and physical constants."""

    state.cvmix_g = gravity
    state.cvmix_rho0 = rho0
    state.cvmix_gorho0 = gravity / rho0
    state.z_w = np.zeros(nlev + 1, dtype=np.float64)
    state.z_r = np.zeros(nlev + 1, dtype=np.float64)
    state.h_r = np.zeros(nlev + 1, dtype=np.float64)
    state.z_w[0] = -h0
    dz = h0 / nlev
    for k in range(1, nlev + 1):
        state.z_w[k] = state.z_w[k - 1] + dz
        state.h_r[k] = dz
        state.z_r[k] = state.z_w[k - 1] + 0.5 * dz


def interior_nonconv(
    state: CVMixState,
    nlev: int,
    NN: np.ndarray,
    NNT: np.ndarray,
    NNS: np.ndarray,
    SS: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nus: np.ndarray,
    Rig: np.ndarray,
) -> None:
    """Compute non-convective interior CVMix contributions."""

    del NNT, NNS
    if not state.use_interior:
        return

    if state.use_background:
        for k in range(1, nlev):
            num[k] = max(num[k], state.background_viscosity)
            nuh[k] = max(nuh[k], state.background_diffusivity)
            nus[k] = max(nus[k], state.background_diffusivity)

    if state.use_shear:
        for k in range(1, nlev):
            ri = NN[k] / max(SS[k], _EPS)
            Rig[k] = ri
            if state.shear_mix_scheme == CVMIX_SHEAR_PP:
                denom = (
                    1.0 + state.shear_PP_alpha * max(0.0, ri)
                ) ** state.shear_PP_exp
                visc = state.shear_PP_nu_zero / denom
                diff = visc / (1.0 + state.shear_PP_alpha * max(0.0, ri))
            else:
                ratio = min(1.0, max(0.0, ri / state.shear_KPP_Ri_zero))
                factor = (1.0 - ratio * ratio) ** state.shear_KPP_exp
                visc = state.shear_KPP_nu_zero * factor
                diff = visc
            num[k] = max(num[k], visc)
            nuh[k] = max(nuh[k], diff)
            nus[k] = max(nus[k], diff)


def interior_conv(
    state: CVMixState,
    nlev: int,
    NN: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nus: np.ndarray,
) -> None:
    """Apply convective interior mixing where the water column is unstable."""

    if not (state.use_interior and state.use_convection):
        return
    for k in range(1, nlev):
        if NN[k] < state.convection_triggerBVF:
            num[k] = max(num[k], state.convection_viscosity)
            nuh[k] = max(nuh[k], state.convection_diffusivity)
            nus[k] = max(nus[k], state.convection_diffusivity)


def surface_layer(
    state: CVMixState,
    nlev: int,
    h: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    *,
    u_taus: float,
    hbl: float,
) -> None:
    """Simple parabolic surface boundary-layer enhancement."""

    if not state.use_surface_layer or hbl <= 0.0:
        state.zsbl = 0.0
        return
    state.zsbl = -hbl
    depth_from_surface = 0.0
    for k in range(nlev, 0, -1):
        depth_from_surface += h[k]
        if depth_from_surface <= hbl:
            sigma = depth_from_surface / hbl
            shape = sigma * (1.0 - sigma) * (1.0 - sigma)
            value = 0.4 * u_taus * hbl * shape
            num[k] = max(num[k], value)
            nuh[k] = max(nuh[k], value)


def bottom_layer(
    state: CVMixState,
    nlev: int,
    h: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    *,
    u_taub: float,
    hbl: float,
) -> None:
    """Simple parabolic bottom boundary-layer enhancement."""

    if not state.use_bottom_layer or hbl <= 0.0:
        state.zbbl = 0.0
        return
    state.zbbl = hbl
    depth_from_bottom = 0.0
    for k in range(1, nlev + 1):
        depth_from_bottom += h[k]
        if depth_from_bottom <= hbl:
            sigma = depth_from_bottom / hbl
            shape = sigma * (1.0 - sigma) * (1.0 - sigma)
            value = 0.4 * u_taub * hbl * shape
            num[k] = max(num[k], value)
            nuh[k] = max(nuh[k], value)


def do_cvmix(
    state: CVMixState,
    nlev: int,
    h0: float,
    h: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    NN: np.ndarray,
    NNT: np.ndarray,
    NNS: np.ndarray,
    SS: np.ndarray,
    u_taus: float,
    u_taub: float,
    num: np.ndarray,
    nuh: np.ndarray,
    nus: np.ndarray,
    Rig: np.ndarray,
) -> None:
    """Dispatch CVMix contributions into GOTM diffusivity arrays."""

    del h0, rho, u, v
    interior_nonconv(state, nlev, NN, NNT, NNS, SS, num, nuh, nus, Rig)
    interior_conv(state, nlev, NN, num, nuh, nus)
    surface_layer(state, nlev, h, num, nuh, u_taus=u_taus, hbl=max(h[1], h[nlev]))
    bottom_layer(state, nlev, h, num, nuh, u_taub=u_taub, hbl=max(h[1], h[nlev]))


def clean_cvmix(state: CVMixState) -> None:
    """Release CVMix work arrays."""

    state.z_w = None
    state.z_r = None
    state.h_r = None
