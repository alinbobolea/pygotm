r"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: gotm --- the general framework \label{sec:gotm}
!
! !DESCRIPTION:
! This is 'where it all happens'. This module provides the internal
! routines {\tt init\_gotm()} to initialise the whole model and
! {\tt time\_loop()} to manage the time-stepping of all fields.
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import gsw
import numpy as np

from pygotm.airsea.airsea import (
    AirSeaDriverState,
    clean_airsea,
    do_airsea,
    init_airsea,
    integrated_fluxes,
    post_init_airsea,
    set_sst,
    set_ssuv,
)
from pygotm.airsea.airsea_fluxes import FAIRALL, KONDO
from pygotm.airsea.airsea_variables import (
    BERLIAND_BERLIAND,
    BIGNAMI,
    CLARK,
    COGLEY,
    CONST,
    HASTENRATH_LAMB,
    JOSEY1,
    JOSEY2,
    PAYNE,
)
from pygotm.config import GotmConfig, GotmSettings, load_config
from pygotm.config.settings import InputSetting, _canonical_token
from pygotm.extras.seagrass.seagrass import (
    SeagrassState,
    do_seagrass,
    end_seagrass,
    init_seagrass,
    post_init_seagrass,
)
from pygotm.fabm.config import FABMConfig, load_fabm_config
from pygotm.fabm.engine import FABMEngine
from pygotm.fabm.fabm_loop import (
    _record_scalar_diagnostics,
    run_fabm_chunk,
)
from pygotm.gotm.diagnostics import (
    DiagnosticsState,
    clean_diagnostics,
    do_diagnostics,
    init_diagnostics,
)
from pygotm.gotm.register_all_variables import (
    FieldRegistry,
    do_register_all_variables,
    fm,
    snapshot_registry,
)
from pygotm.gotm.runtime_builder import (
    RuntimeBundle,
    build_runtime_from_run,
)
from pygotm.gotm.time_loop import run_compiled_time_loop
from pygotm.icethm import (
    IceParams,
    IceState,
    init_ice,
    make_ice_params_from_mapping,
    step_ice,
)
from pygotm.input.input import (
    ProfileInput,
    ScalarInput,
    close_input,
    do_input,
    init_input,
    register_input,
)
from pygotm.meanflow.coriolis import coriolis
from pygotm.meanflow.external_pressure import external_pressure
from pygotm.meanflow.friction import friction
from pygotm.meanflow.internal_pressure import internal_pressure
from pygotm.meanflow.meanflow import (
    MeanflowState,
    clean_meanflow,
    init_meanflow,
    post_init_meanflow,
)
from pygotm.meanflow.salinity import salinity
from pygotm.meanflow.shear import shear
from pygotm.meanflow.stratification import stratification
from pygotm.meanflow.temperature import temperature
from pygotm.meanflow.uequation import uequation
from pygotm.meanflow.updategrid import updategrid
from pygotm.meanflow.vequation import vequation
from pygotm.meanflow.wequation import wequation
from pygotm.observations.observations import (
    ObservationsState,
    clean_observations,
    get_all_obs,
    init_observations,
    post_init_observations,
)
from pygotm.stokes_drift.stokes_drift import (
    CONSTANT as STOKES_CONSTANT,
)
from pygotm.stokes_drift.stokes_drift import (
    EXPONENTIAL as STOKES_EXPONENTIAL,
)
from pygotm.stokes_drift.stokes_drift import (
    FROMFILE as STOKES_FROMFILE,
)
from pygotm.stokes_drift.stokes_drift import (
    FROMUS as STOKES_FROMUS,
)
from pygotm.stokes_drift.stokes_drift import (
    NOTHING as STOKES_NOTHING,
)
from pygotm.stokes_drift.stokes_drift import (
    THEORYWAVE as STOKES_THEORYWAVE,
)
from pygotm.stokes_drift.stokes_drift import (
    StokesDriftState,
    clean_stokes_drift,
    do_stokes_drift,
    init_stokes_drift,
    post_init_stokes_drift,
)
from pygotm.turbulence.turbulence import (
    CCH02,
    CHCD01A,
    CHCD01B,
    GL78,
    KC94,
    LDOR96,
    LIST,
    MY82,
    Blackadar,
    Bougeault_Andre,
    Constant,
    Munk_Anderson,
    Parabolic,
    Robert_Ouellet,
    Schumann_Gerz,
    Triangular,
    TurbulenceState,
    Xing_Davies,
    clean_turbulence,
    diss_eq,
    do_turbulence,
    epsb_algebraic,
    epsb_dynamic,
    first_order,
    generic_eq,
    init_turbulence,
    kb_algebraic,
    kb_dynamic,
    length_eq,
    logarithmic,
    no_model,
    omega_eq,
    post_init_turbulence,
    quasi_Eq,
    quasi_Eq_H15,
    run_variances,
    second_order,
    tke_keps,
    tke_local_eq,
    tke_MY,
    weak_Eq_Kb,
    weak_Eq_Kb_Eq,
)
from pygotm.util.density import (
    CP0,
    METHOD_LINEAR_TEOS10,
    METHOD_LINEAR_USER,
    METHOD_TEOS10,
    DensityState,
    clean_density,
    do_density,
    init_density,
)
from pygotm.util.time import GotmTime

__all__ = [
    "GotmRun",
    "finalize_gotm",
    "initialize_gotm",
    "initialize_gotm_from_settings",
    "integrate_gotm",
    "integrate_gotm_compiled",
]

_GRID_METHOD = {"analytical": 0, "file_sigma": 1, "file_h": 2}
_DENSITY_METHOD = {
    "full_teos10": METHOD_TEOS10,
    # Reference YAMLs use full_teos-10, normalised by _canonical_token.
    "full_teos_10": METHOD_TEOS10,
    "linear_teos10": METHOD_LINEAR_TEOS10,
    "linear_teos_10": METHOD_LINEAR_TEOS10,
    "linear_custom": METHOD_LINEAR_USER,
}
_FLUX_METHOD = {"off": 0, "kondo": KONDO, "fairall": FAIRALL}
_HUM_METHOD = {"relative": 1, "wet_bulb": 2, "dew_point": 3, "specific": 4}
_ALBEDO_METHOD = {"constant": CONST, "payne": PAYNE, "cogley": COGLEY}
_LONGWAVE_METHOD = {
    "constant": 2,
    "file": 2,
    "clark": CLARK,
    "hastenrath_lamb": HASTENRATH_LAMB,
    "bignami": BIGNAMI,
    "berliand_berliand": BERLIAND_BERLIAND,
    "josey1": JOSEY1,
    "josey2": JOSEY2,
}
_TURB_METHOD = {
    "no_model": no_model,
    "first_order": first_order,
    "second_order": second_order,
    "cvmix": 100,
}
_TKE_METHOD = {"local_eq": tke_local_eq, "tke": tke_keps, "mellor_yamada": tke_MY}
_LEN_SCALE_METHOD = {
    "parabolic": Parabolic,
    "triangular": Triangular,
    "xing_davies": Xing_Davies,
    "robert_ouellet": Robert_Ouellet,
    "blackadar": Blackadar,
    "bougeault_andre": Bougeault_Andre,
    "dissipation": diss_eq,
    "gls": generic_eq,
    "omega": omega_eq,
    "mellor_yamada": length_eq,
}
_STAB_METHOD = {
    "constant": Constant,
    "munk_anderson": Munk_Anderson,
    "schumann_gerz": Schumann_Gerz,
}
_BC_METHOD = {"dirichlet": 0, "neumann": 1}
_BC_TYPE = {"logarithmic": logarithmic, "tke_injection": 2}
_SCND_METHOD = {
    "quasi_eq": quasi_Eq,
    "weak_eq_kb_eq": weak_Eq_Kb_Eq,
    "weak_eq_kb": weak_Eq_Kb,
    "quasi_eq_h15": quasi_Eq_H15,
}
_SCND_COEFF = {
    "custom": LIST,
    "gibson_launder": GL78,
    "mellor_yamada": MY82,
    "kantha_clayson": KC94,
    "luyten": LDOR96,
    "canuto_a": CHCD01A,
    "canuto_b": CHCD01B,
    "cheng": CCH02,
}
_MY_LENGTH = {"parabolic": 1, "triangular": 2, "linear": 3}
_IW_MODEL = {"off": 0, "mellor": 1, "large": 2}
_KB_METHOD = {"algebraic": kb_algebraic, "prognostic": kb_dynamic}
_EPSB_METHOD = {"algebraic": epsb_algebraic, "prognostic": epsb_dynamic}


@dataclass
class SurfaceInputs:
    """Registered scalar inputs used by the single-column runtime."""

    heat: ScalarInput | None = None
    tx: ScalarInput | None = None
    ty: ScalarInput | None = None
    u10: ScalarInput | None = None
    v10: ScalarInput | None = None
    airp: ScalarInput | None = None
    airt: ScalarInput | None = None
    hum: ScalarInput | None = None
    cloud: ScalarInput | None = None
    precip: ScalarInput | None = None
    swr: ScalarInput | None = None
    longwave: ScalarInput | None = None
    sst_obs: ScalarInput | None = None
    sss_obs: ScalarInput | None = None


@dataclass
class OutputSchedule:
    interval_steps: int = 1
    capture_initial: bool = True
    time_method: str = "point"
    k_start: int = 1
    k1_start: int = 1


@dataclass
class GotmRun:
    """Single-column GOTM runtime state."""

    settings: GotmSettings
    document: dict[str, Any]
    yaml_path: Path
    time: GotmTime
    meanflow: MeanflowState
    density: DensityState
    observations: ObservationsState
    airsea: AirSeaDriverState
    stokes_drift: StokesDriftState
    seagrass: SeagrassState
    turbulence: TurbulenceState
    diagnostics: DiagnosticsState
    fabm_config: FABMConfig | None
    fabm_engine: FABMEngine | None
    surface_inputs: SurfaceInputs
    registry: FieldRegistry
    nlev: int
    dt: float
    cnpar: float
    latitude: float
    longitude: float
    depth: float
    output_schedule: OutputSchedule
    ice_params: IceParams | None = None
    ice_state: IceState | None = None
    current_i0: float = 0.0
    initialized: bool = False
    friction_first: list[bool] = field(default_factory=lambda: [True])
    snapshot_times: list[str] = field(default_factory=list)
    snapshots: list[dict[str, float | np.ndarray]] = field(default_factory=list)
    accum_snapshot: dict[str, float | np.ndarray] = field(default_factory=dict)
    accum_count: int = 0


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _surface_value(input_: ScalarInput | None, default: float = 0.0) -> float:
    if input_ is None:
        return default
    return float(input_.value)


def _maybe_register_scalar(
    name: str,
    raw: dict[str, Any] | None,
    *,
    default_method: str = "constant",
) -> ScalarInput | None:
    setting = InputSetting.model_validate({} if raw is None else raw)
    method = _canonical_token(setting.method, default_method)
    if method == "off":
        return None
    if method not in {"constant", "file"}:
        msg = f"unsupported scalar input method {method!r} for {name!r}"
        raise NotImplementedError(msg)

    input_ = ScalarInput(
        name=name,
        method=0 if method == "constant" else 2,
        path=setting.file,
        index=setting.column,
        constant_value=setting.constant_value,
        scale_factor=setting.scale_factor,
        add_offset=setting.offset,
        method_constant=0,
        method_file=2,
    )
    register_input(input_)
    return input_


def _input_token(raw: dict[str, Any] | None, default: str = "constant") -> str:
    setting = InputSetting.model_validate({} if raw is None else raw)
    return _canonical_token(setting.method, default)


def _register_stokes_scalar(
    name: str,
    raw: dict[str, Any] | None,
) -> ScalarInput | None:
    if not raw:
        return None
    setting = InputSetting.model_validate({} if raw is None else raw)
    method = _canonical_token(setting.method, "off")
    if method == "off":
        return None
    if method not in {"constant", "file"}:
        msg = f"unsupported scalar Stokes input method {method!r} for {name!r}"
        raise NotImplementedError(msg)
    input_ = ScalarInput(
        name=name,
        method=STOKES_CONSTANT if method == "constant" else STOKES_FROMFILE,
        path=setting.file,
        index=setting.column,
        constant_value=setting.constant_value,
        scale_factor=setting.scale_factor,
        add_offset=setting.offset,
        method_constant=STOKES_CONSTANT,
        method_file=STOKES_FROMFILE,
    )
    register_input(input_)
    return input_


def _register_stokes_profile(
    name: str,
    raw: dict[str, Any] | None,
) -> tuple[int, ProfileInput | None]:
    if not raw:
        return STOKES_NOTHING, None
    setting = InputSetting.model_validate({} if raw is None else raw)
    method = _canonical_token(setting.method, "off")
    if method in {"off", "none"}:
        return STOKES_NOTHING, None
    if method == "file":
        input_ = ProfileInput(
            name=name,
            method=STOKES_FROMFILE,
            path=setting.file,
            index=setting.column,
            constant_value=setting.constant_value,
            scale_factor=setting.scale_factor,
            add_offset=setting.offset,
            method_constant=STOKES_CONSTANT,
            method_file=STOKES_FROMFILE,
        )
        register_input(input_)
        return STOKES_FROMFILE, input_
    if method == "exponential":
        return STOKES_EXPONENTIAL, None
    if method in {"empirical", "theorywave", "theory_wave"}:
        return STOKES_THEORYWAVE, None
    if method in {"us", "vs"}:
        return STOKES_FROMUS, None
    msg = f"unsupported Stokes profile method {method!r} for {name!r}"
    raise NotImplementedError(msg)


def _configure_stokes_drift_from_document(
    state: StokesDriftState,
    document: dict[str, Any],
) -> None:
    waves = _mapping(document.get("waves"))
    raw = _mapping(waves.get("stokes_drift"))
    init_stokes_drift(state)

    us0_raw = _mapping(raw.get("us0"))
    vs0_raw = _mapping(raw.get("vs0"))
    ds_raw = _mapping(_mapping(raw.get("exponential")).get("ds"))
    if "ds" in raw:
        ds_raw = _mapping(raw.get("ds"))
    empirical = _mapping(raw.get("empirical"))
    uwnd_raw = _mapping(empirical.get("uwnd"))
    vwnd_raw = _mapping(empirical.get("vwnd"))

    state.us0_input = _register_stokes_scalar("us0", us0_raw)
    state.vs0_input = _register_stokes_scalar("vs0", vs0_raw)
    state.ds_input = _register_stokes_scalar("ds", ds_raw) if ds_raw else None
    state.uwnd_input = _register_stokes_scalar("uwnd", uwnd_raw) if uwnd_raw else None
    state.vwnd_input = _register_stokes_scalar("vwnd", vwnd_raw) if vwnd_raw else None

    state.us0_method = (
        STOKES_NOTHING if state.us0_input is None else int(state.us0_input.method)
    )
    state.vs0_method = (
        STOKES_NOTHING if state.vs0_input is None else int(state.vs0_input.method)
    )
    state.ds_method = (
        STOKES_NOTHING if state.ds_input is None else int(state.ds_input.method)
    )
    state.uwnd_method = (
        STOKES_NOTHING if state.uwnd_input is None else int(state.uwnd_input.method)
    )
    state.vwnd_method = (
        STOKES_NOTHING if state.vwnd_input is None else int(state.vwnd_input.method)
    )

    state.usprof_method, state.usprof_input = _register_stokes_profile(
        "us",
        _mapping(raw.get("us")),
    )
    state.vsprof_method, state.vsprof_input = _register_stokes_profile(
        "vs",
        _mapping(raw.get("vs")),
    )
    state.dusdz_method, state.dusdz_input = _register_stokes_profile(
        "dusdz",
        _mapping(raw.get("dusdz")),
    )
    state.dvsdz_method, state.dvsdz_input = _register_stokes_profile(
        "dvsdz",
        _mapping(raw.get("dvsdz")),
    )


def _configure_seagrass_from_document(
    state: SeagrassState,
    document: dict[str, Any],
) -> None:
    raw = _mapping(document.get("seagrass"))
    init_seagrass(
        state,
        method=int(raw.get("method", 0)),
        grassfile=str(raw.get("file", "seagrass.dat") or "seagrass.dat"),
        alpha=float(raw.get("alpha", 0.0)),
    )


def _configure_meanflow_from_document(
    meanflow: MeanflowState,
    document: dict[str, Any],
) -> None:
    surface = _mapping(document.get("surface"))
    roughness = _mapping(surface.get("roughness"))
    bottom = _mapping(document.get("bottom"))

    if "charnock" in roughness:
        meanflow.charnock = bool(roughness["charnock"])
    if "charnock_val" in roughness:
        meanflow.charnock_val = float(roughness["charnock_val"])
    if "z0s_min" in roughness:
        meanflow.z0s_min = float(roughness["z0s_min"])
    if "calc_bottom_stress" in bottom:
        meanflow.calc_bottom_stress = bool(bottom["calc_bottom_stress"])
    if "h0b" in bottom:
        meanflow.h0b = float(bottom["h0b"])


def _configure_density_from_document(
    density: DensityState,
    document: dict[str, Any],
) -> float:
    eqstate = _mapping(document.get("equation_of_state"))
    method_name = _canonical_token(eqstate.get("method"), "full_teos10")
    try:
        density.density_method = _DENSITY_METHOD[method_name]
    except KeyError as exc:
        msg = f"unsupported equation_of_state.method {method_name!r}"
        raise NotImplementedError(msg) from exc

    if "rho0" in eqstate:
        density.rho0 = float(eqstate["rho0"])

    linear = _mapping(eqstate.get("linear"))
    if "T0" in linear:
        density.T0 = float(linear["T0"])
    if "S0" in linear:
        density.S0 = float(linear["S0"])
    if "p0" in linear:
        density.p0 = float(linear["p0"])
    if "alpha" in linear:
        density.alpha0 = float(linear["alpha"])
    if "beta" in linear:
        density.beta0 = float(linear["beta"])
    return float(linear.get("cp", CP0))


def _configure_airsea_from_document(
    airsea: AirSeaDriverState,
    document: dict[str, Any],
) -> tuple[SurfaceInputs, IceParams]:
    surface = _mapping(document.get("surface"))
    fluxes = _mapping(surface.get("fluxes"))
    swr = _mapping(surface.get("swr"))
    longwave = _mapping(surface.get("longwave_radiation"))
    albedo = _mapping(surface.get("albedo"))
    hum = _mapping(surface.get("hum"))
    ice = _mapping(surface.get("ice"))

    flux_method = _canonical_token(fluxes.get("method"), "off")
    ssuv_method = _canonical_token(surface.get("ssuv_method"), "relative")
    hum_type = _canonical_token(hum.get("type"), "relative")
    swr_method = _canonical_token(swr.get("method"), "constant")
    longwave_method = _canonical_token(longwave.get("method"), "clark")
    albedo_method = _canonical_token(albedo.get("method"), "payne")
    ice_model = _canonical_token(ice.get("model"), "simple")

    if ssuv_method not in {"absolute", "relative"}:
        msg = f"unsupported surface.ssuv_method {ssuv_method!r}"
        raise NotImplementedError(msg)
    if ice_model not in {
        "no_ice",
        "simple",
        "basal_melt",
        "lebedev",
        "mylake",
        "winton",
    }:
        msg = f"unsupported surface.ice.model {ice_model!r}"
        raise NotImplementedError(msg)
    ice_params = make_ice_params_from_mapping(ice)

    init_airsea(
        airsea,
        fluxes_method=_FLUX_METHOD.get(flux_method, 0),
        hum_method=_HUM_METHOD[hum_type],
        shortwave_method=3 if swr_method == "calculate" else 1,
        shortwave_type=int(swr.get("type", 1)),
        shortwave_scale_factor=float(swr.get("scale_factor", 1.0)),
        longwave_method=_LONGWAVE_METHOD[longwave_method],
        longwave_type=int(longwave.get("type", 1)),
        albedo_method=_ALBEDO_METHOD[albedo_method],
        const_albedo=float(albedo.get("constant_value", 0.0)),
        heat_scale_factor=float(_mapping(fluxes.get("heat")).get("scale_factor", 1.0)),
        ssuv_method=0 if ssuv_method == "absolute" else 1,
    )
    airsea.calc_evaporation = bool(surface.get("calc_evaporation", False))

    inputs = SurfaceInputs()
    if flux_method == "off":
        inputs.heat = _maybe_register_scalar("heat_input", _mapping(fluxes.get("heat")))
        inputs.tx = _maybe_register_scalar("tx_input", _mapping(fluxes.get("tx")))
        inputs.ty = _maybe_register_scalar("ty_input", _mapping(fluxes.get("ty")))
    else:
        inputs.u10 = _maybe_register_scalar("u10_input", _mapping(surface.get("u10")))
        inputs.v10 = _maybe_register_scalar("v10_input", _mapping(surface.get("v10")))
        inputs.airp = _maybe_register_scalar(
            "airp_input", _mapping(surface.get("airp"))
        )
        inputs.airt = _maybe_register_scalar(
            "airt_input", _mapping(surface.get("airt"))
        )
        inputs.hum = _maybe_register_scalar("hum_input", hum)
        inputs.cloud = _maybe_register_scalar(
            "cloud_input", _mapping(surface.get("cloud"))
        )

    inputs.precip = _maybe_register_scalar(
        "precip_input", _mapping(surface.get("precip"))
    )
    if swr_method != "calculate":
        inputs.swr = _maybe_register_scalar("I_0_input", swr)
    if longwave_method in {"constant", "file"}:
        inputs.longwave = _maybe_register_scalar("ql_input", longwave)
    inputs.sst_obs = (
        _maybe_register_scalar("sst_obs_input", _mapping(surface.get("sst")))
        if "sst" in surface
        else None
    )
    inputs.sss_obs = (
        _maybe_register_scalar("sss_input", _mapping(surface.get("sss")))
        if "sss" in surface
        else None
    )
    return inputs, ice_params


def _configure_turbulence_from_document(
    state: TurbulenceState,
    document: dict[str, Any],
) -> None:
    raw = _mapping(document.get("turbulence"))
    overrides: dict[str, bool | int | float] = {}

    turb_method = _canonical_token(raw.get("turb_method"), "second_order")
    overrides["turb_method"] = _TURB_METHOD.get(turb_method, second_order)
    overrides["tke_method"] = _TKE_METHOD[
        _canonical_token(raw.get("tke_method"), "tke")
    ]
    overrides["len_scale_method"] = _LEN_SCALE_METHOD[
        _canonical_token(raw.get("len_scale_method"), "dissipation")
    ]
    overrides["stab_method"] = _STAB_METHOD[
        _canonical_token(raw.get("stab_method"), "schumann_gerz")
    ]

    bc = _mapping(raw.get("bc"))
    for key in ("k_ubc", "k_lbc", "psi_ubc", "psi_lbc"):
        if key in bc:
            overrides[key] = _BC_METHOD[_canonical_token(bc[key], "neumann")]
    for key in ("ubc_type", "lbc_type"):
        if key in bc:
            overrides[key] = _BC_TYPE[_canonical_token(bc[key], "logarithmic")]

    turb_param = _mapping(raw.get("turb_param"))
    for key, attr in (
        ("cm0_fix", "cm0_fix"),
        ("Prandtl0_fix", "Prandtl0_fix"),
        ("cw", "cw"),
        ("compute_kappa", "compute_kappa"),
        ("kappa", "kappa"),
        ("compute_c3", "compute_c3"),
        ("Ri_st", "ri_st"),
        ("length_lim", "length_lim"),
        ("galp", "galp"),
        ("const_num", "const_num"),
        ("const_nuh", "const_nuh"),
        ("k_min", "k_min"),
        ("eps_min", "eps_min"),
        ("kb_min", "kb_min"),
        ("epsb_min", "epsb_min"),
    ):
        if key in turb_param:
            overrides[attr] = turb_param[key]

    generic = _mapping(raw.get("generic"))
    for key in (
        "compute_param",
        "gen_m",
        "gen_n",
        "gen_p",
        "cpsi1",
        "cpsi2",
        "cpsi3minus",
        "cpsi3plus",
        "cpsix",
        "cpsi4",
        "sig_kpsi",
        "sig_psi",
        "gen_d",
        "gen_alpha",
        "gen_l",
    ):
        if key in generic:
            overrides[key] = generic[key]

    keps = _mapping(raw.get("keps"))
    for key in (
        "ce1",
        "ce2",
        "ce3minus",
        "ce3plus",
        "cex",
        "ce4",
        "sig_k",
        "sig_e",
        "sig_peps",
    ):
        if key in keps:
            overrides[key] = keps[key]

    kw = _mapping(raw.get("kw"))
    for key in ("cw1", "cw2", "cw3minus", "cw3plus", "cwx", "cw4", "sig_kw", "sig_w"):
        if key in kw:
            overrides[key] = kw[key]

    my = _mapping(raw.get("my"))
    for key in ("e1", "e2", "e3", "ex", "e6", "sq", "sl"):
        if key in my:
            overrides[key] = my[key]
    if "length" in my:
        overrides["my_length"] = _MY_LENGTH[_canonical_token(my["length"], "parabolic")]

    scnd = _mapping(raw.get("scnd"))
    overrides["scnd_method"] = _SCND_METHOD[
        _canonical_token(scnd.get("method"), "weak_eq_kb_eq")
    ]
    overrides["kb_method"] = _KB_METHOD[
        _canonical_token(scnd.get("kb_method"), "algebraic")
    ]
    overrides["epsb_method"] = _EPSB_METHOD[
        _canonical_token(scnd.get("epsb_method"), "algebraic")
    ]
    overrides["scnd_coeff"] = _SCND_COEFF[
        _canonical_token(scnd.get("scnd_coeff"), "canuto_a")
    ]
    for key in (
        "cc1",
        "cc2",
        "cc3",
        "cc4",
        "cc5",
        "cc6",
        "ct1",
        "ct2",
        "ct3",
        "ct4",
        "ct5",
        "ctt",
    ):
        if key in scnd:
            overrides[key] = scnd[key]

    iw = _mapping(raw.get("iw"))
    overrides["iw_model"] = _IW_MODEL[_canonical_token(iw.get("method"), "off")]
    for key in ("alpha", "klim", "rich_cr", "numshear", "num", "nuh"):
        if key in iw:
            attribute = {
                "klim": "klimiw",
                "num": "numiw",
                "nuh": "nuhiw",
            }.get(key, key)
            overrides[attribute] = iw[key]

    init_turbulence(state, overrides=overrides)


def _configure_output_schedule(document: dict[str, Any], dt: float) -> OutputSchedule:
    raw_output = _mapping(document.get("output"))
    specs = [value for value in raw_output.values() if isinstance(value, dict)]
    if not specs:
        return OutputSchedule()

    interval_steps = 1
    current_time_method = "point"
    k_start = 1
    k1_start = 1
    for spec in specs:
        if not bool(spec.get("is_active", True)):
            continue

        time_method = _canonical_token(spec.get("time_method"), "point")
        if time_method not in {"point", "mean", "integrated"}:
            msg = f"unsupported output time_method {time_method!r}"
            raise NotImplementedError(msg)
        current_time_method = time_method

        time_unit = _canonical_token(spec.get("time_unit"), "day")
        time_step = max(1, int(spec.get("time_step", 1)))
        if time_unit == "dt":
            interval = time_step
        elif time_unit == "second":
            interval = max(1, round(time_step / dt))
        elif time_unit == "hour":
            interval = max(1, round(time_step * 3600.0 / dt))
        elif time_unit == "day":
            interval = max(1, round(time_step * 86400.0 / dt))
        else:
            msg = f"unsupported output time_unit {time_unit!r}"
            raise NotImplementedError(msg)

        interval_steps = (
            interval if interval_steps == 1 else min(interval_steps, interval)
        )
        k_start = max(k_start, int(spec.get("k_start", k_start)))
        k1_start = max(k1_start, int(spec.get("k1_start", k1_start)))

    return OutputSchedule(
        interval_steps=interval_steps,
        time_method=current_time_method,
        k_start=k_start,
        k1_start=k1_start,
    )


def _apply_initial_profiles(run: GotmRun) -> None:
    meanflow = run.meanflow
    observations = run.observations

    assert meanflow.z is not None
    assert meanflow.u is not None
    assert meanflow.v is not None
    assert meanflow.T is not None
    assert meanflow.Ti is not None
    assert meanflow.Tp is not None
    assert meanflow.Tobs is not None
    assert meanflow.S is not None
    assert meanflow.Sp is not None
    assert meanflow.Sobs is not None
    assert observations.tprof_input.data is not None
    assert observations.sprof_input.data is not None
    assert observations.uprof_input.data is not None
    assert observations.vprof_input.data is not None

    z = meanflow.z[1 : run.nlev + 1]
    pressure = -z

    if observations.initial_salinity_type == 1:
        meanflow.Sp[1 : run.nlev + 1] = observations.sprof_input.data[1 : run.nlev + 1]
        meanflow.S[1 : run.nlev + 1] = gsw.SA_from_SP(
            meanflow.Sp[1 : run.nlev + 1],
            pressure,
            run.longitude,
            run.latitude,
        )
    else:
        meanflow.S[1 : run.nlev + 1] = observations.sprof_input.data[1 : run.nlev + 1]
        meanflow.Sp[1 : run.nlev + 1] = gsw.SP_from_SA(
            meanflow.S[1 : run.nlev + 1],
            pressure,
            run.longitude,
            run.latitude,
        )

    if observations.initial_temperature_type == 1:
        meanflow.Ti[1 : run.nlev + 1] = observations.tprof_input.data[1 : run.nlev + 1]
        meanflow.T[1 : run.nlev + 1] = gsw.CT_from_t(
            meanflow.S[1 : run.nlev + 1],
            meanflow.Ti[1 : run.nlev + 1],
            pressure,
        )
        meanflow.Tp[1 : run.nlev + 1] = gsw.pt_from_t(
            meanflow.S[1 : run.nlev + 1],
            meanflow.Ti[1 : run.nlev + 1],
            pressure,
            0.0,
        )
    elif observations.initial_temperature_type == 2:
        meanflow.Tp[1 : run.nlev + 1] = observations.tprof_input.data[1 : run.nlev + 1]
        meanflow.T[1 : run.nlev + 1] = gsw.CT_from_pt(
            meanflow.S[1 : run.nlev + 1],
            meanflow.Tp[1 : run.nlev + 1],
        )
        meanflow.Ti[1 : run.nlev + 1] = gsw.t_from_CT(
            meanflow.S[1 : run.nlev + 1],
            meanflow.T[1 : run.nlev + 1],
            pressure,
        )
    else:
        meanflow.T[1 : run.nlev + 1] = observations.tprof_input.data[1 : run.nlev + 1]
        meanflow.Tp[1 : run.nlev + 1] = gsw.pt_from_CT(
            meanflow.S[1 : run.nlev + 1],
            meanflow.T[1 : run.nlev + 1],
        )
        meanflow.Ti[1 : run.nlev + 1] = gsw.t_from_CT(
            meanflow.S[1 : run.nlev + 1],
            meanflow.T[1 : run.nlev + 1],
            pressure,
        )

    meanflow.Sobs[:] = meanflow.S
    meanflow.Tobs[:] = meanflow.T
    meanflow.u[1 : run.nlev + 1] = observations.uprof_input.data[1 : run.nlev + 1]
    meanflow.v[1 : run.nlev + 1] = observations.vprof_input.data[1 : run.nlev + 1]


def _update_relaxation_targets(run: GotmRun) -> None:
    meanflow = run.meanflow
    observations = run.observations

    assert meanflow.z is not None
    assert meanflow.Sobs is not None
    assert meanflow.Tobs is not None
    assert meanflow.S is not None

    z = meanflow.z[1 : run.nlev + 1]
    pressure = -z

    if (
        observations.sprof_input.method != 0
        and observations.sprof_input.data is not None
    ):
        if observations.initial_salinity_type == 1:
            meanflow.Sobs[1 : run.nlev + 1] = gsw.SA_from_SP(
                observations.sprof_input.data[1 : run.nlev + 1],
                pressure,
                run.longitude,
                run.latitude,
            )
        else:
            meanflow.Sobs[1 : run.nlev + 1] = observations.sprof_input.data[
                1 : run.nlev + 1
            ]

    if (
        observations.tprof_input.method != 0
        and observations.tprof_input.data is not None
    ):
        if observations.initial_temperature_type == 1:
            meanflow.Tobs[1 : run.nlev + 1] = gsw.CT_from_t(
                meanflow.Sobs[1 : run.nlev + 1],
                observations.tprof_input.data[1 : run.nlev + 1],
                pressure,
            )
        elif observations.initial_temperature_type == 2:
            meanflow.Tobs[1 : run.nlev + 1] = gsw.CT_from_pt(
                meanflow.Sobs[1 : run.nlev + 1],
                observations.tprof_input.data[1 : run.nlev + 1],
            )
        else:
            meanflow.Tobs[1 : run.nlev + 1] = observations.tprof_input.data[
                1 : run.nlev + 1
            ]


def _accumulate_snapshot(run: GotmRun) -> None:
    snap = snapshot_registry(run.registry)
    if run.accum_count == 0:
        run.accum_snapshot = snap
    else:
        for k, v in snap.items():
            run.accum_snapshot[k] = run.accum_snapshot[k] + v
    run.accum_count += 1


def _save_snapshot(run: GotmRun) -> None:
    run.snapshot_times.append(run.time.timestr)
    method = run.output_schedule.time_method
    if method == "point" or run.accum_count == 0:
        run.snapshots.append(snapshot_registry(run.registry))
    elif method == "mean":
        n = run.accum_count
        run.snapshots.append({k: v / n for k, v in run.accum_snapshot.items()})
        run.accum_snapshot.clear()
        run.accum_count = 0
    else:
        run.snapshots.append(dict(run.accum_snapshot))
        run.accum_snapshot.clear()
        run.accum_count = 0


def initialize_gotm_from_settings(
    settings: GotmSettings,
    *,
    yaml_path: str | Path = "gotm.yaml",
    list_fields: bool = False,
    document: dict[str, Any] | None = None,
) -> GotmRun:
    """Initialise the single-column runtime from a parsed settings model."""

    resolved_document = (
        deepcopy(document)
        if document is not None
        else settings.model_dump(by_alias=True, exclude_none=True)
    )
    time_state = GotmTime(
        timestep=settings.time.dt,
        timefmt=settings.time.method,
        start=settings.time.start,
        stop=settings.time.stop,
    )
    time_state.MaxN = settings.time.max_steps
    time_state.init_time()

    nlev = settings.grid.nlev
    depth = settings.location.depth

    init_input(nlev)

    meanflow = MeanflowState()
    init_meanflow(meanflow)
    _configure_meanflow_from_document(meanflow, resolved_document)
    meanflow.depth = depth
    meanflow.grid_method = _GRID_METHOD[settings.grid.method]
    meanflow.ddu = settings.grid.ddu
    meanflow.ddl = settings.grid.ddl
    meanflow.grid_file = settings.grid.file
    post_init_meanflow(meanflow, nlev, settings.location.latitude)
    updategrid(meanflow, nlev, settings.time.dt, zeta=0.0)

    density = DensityState()
    custom_cp = _configure_density_from_document(density, resolved_document)
    init_density(density, nlev)
    if density.density_method == METHOD_LINEAR_USER:
        density.cp = custom_cp

    observations = ObservationsState()
    init_observations(observations, settings)
    assert meanflow.z is not None
    assert meanflow.zi is not None
    assert meanflow.h is not None
    post_init_observations(
        observations,
        depth,
        nlev,
        meanflow.z,
        meanflow.zi,
        meanflow.h,
        meanflow.gravity,
        density,
    )

    airsea = AirSeaDriverState()
    surface_inputs, ice_params = _configure_airsea_from_document(
        airsea,
        resolved_document,
    )
    post_init_airsea(airsea, settings.location.latitude, settings.location.longitude)

    stokes = StokesDriftState()
    _configure_stokes_drift_from_document(stokes, resolved_document)
    post_init_stokes_drift(stokes, nlev)

    seagrass = SeagrassState()
    _configure_seagrass_from_document(seagrass, resolved_document)
    assert meanflow.h is not None
    post_init_seagrass(seagrass, nlev, meanflow.h)

    fabm_config = load_fabm_config(resolved_document, yaml_path)

    do_input(time_state.julianday, time_state.secondsofday, nlev, meanflow.z)
    get_all_obs(
        observations,
        time_state.julianday,
        time_state.secondsofday,
        nlev,
        meanflow.z,
        fsecs=time_state.fsecs,
    )
    meanflow.zeta = float(observations.zeta_input.value)
    updategrid(meanflow, nlev, settings.time.dt, zeta=meanflow.zeta)

    turbulence = TurbulenceState()
    _configure_turbulence_from_document(turbulence, resolved_document)
    post_init_turbulence(turbulence, nlev)

    _apply_initial_profiles(
        GotmRun(
            settings=settings,
            document=resolved_document,
            yaml_path=Path(yaml_path),
            time=time_state,
            meanflow=meanflow,
            density=density,
            observations=observations,
            airsea=airsea,
            stokes_drift=stokes,
            seagrass=seagrass,
            turbulence=turbulence,
            diagnostics=DiagnosticsState(),
            fabm_config=fabm_config,
            fabm_engine=None,
            surface_inputs=surface_inputs,
            registry=FieldRegistry(),
            nlev=nlev,
            dt=settings.time.dt,
            cnpar=settings.time.cnpar,
            latitude=settings.location.latitude,
            longitude=settings.location.longitude,
            depth=depth,
            output_schedule=OutputSchedule(),
        )
    )

    assert meanflow.S is not None
    assert meanflow.T is not None
    ice_state = init_ice(
        ice_params,
        T_air_init=0.0,
        S_sfc_init=float(meanflow.S[nlev]),
    )

    diagnostics = DiagnosticsState()
    init_diagnostics(diagnostics, nlev)

    output_schedule = _configure_output_schedule(resolved_document, settings.time.dt)
    run = GotmRun(
        settings=settings,
        document=resolved_document,
        yaml_path=Path(yaml_path),
        time=time_state,
        meanflow=meanflow,
        density=density,
        observations=observations,
        airsea=airsea,
        stokes_drift=stokes,
        seagrass=seagrass,
        turbulence=turbulence,
        diagnostics=diagnostics,
        fabm_config=fabm_config,
        fabm_engine=None,
        surface_inputs=surface_inputs,
        registry=FieldRegistry(),
        nlev=nlev,
        dt=settings.time.dt,
        cnpar=settings.time.cnpar,
        latitude=settings.location.latitude,
        longitude=settings.location.longitude,
        depth=depth,
        output_schedule=output_schedule,
        ice_params=ice_params,
        ice_state=ice_state,
        initialized=True,
    )
    run.registry = do_register_all_variables(
        settings.location.latitude,
        settings.location.longitude,
        nlev,
        observations=observations,
        diagnostics=diagnostics,
        meanflow=meanflow,
        airsea=airsea,
        density=density,
        turbulence=turbulence,
        stokes_drift=stokes,
        ice_state=ice_state,
        surface_inputs=surface_inputs,
        i0_provider=lambda: run.current_i0,
    )
    assert meanflow.z is not None
    assert meanflow.zi is not None
    assert meanflow.buoy is not None
    assert density.rho_p is not None
    do_airsea(
        airsea,
        yearday=time_state.yearday,
        secs=time_state.secondsofday,
        airp=_surface_value(surface_inputs.airp),
        airt=_surface_value(surface_inputs.airt),
        hum=_surface_value(surface_inputs.hum),
        cloud=_surface_value(surface_inputs.cloud),
        u10=_surface_value(surface_inputs.u10),
        v10=_surface_value(surface_inputs.v10),
        precip=_surface_value(surface_inputs.precip),
        shortwave=_surface_value(surface_inputs.swr),
        heat=_surface_value(surface_inputs.heat),
        tx=_surface_value(surface_inputs.tx),
        ty=_surface_value(surface_inputs.ty),
        longwave=_surface_value(surface_inputs.longwave),
        sst_obs=(
            _surface_value(surface_inputs.sst_obs)
            if surface_inputs.sst_obs is not None
            else None
        ),
    )
    run.current_i0 = airsea.shortwave * (1.0 - airsea.albedo - airsea.bio_albedo)
    do_stokes_drift(
        stokes,
        nlev,
        meanflow.z,
        meanflow.zi,
        meanflow.gravity,
        _surface_value(surface_inputs.u10),
        _surface_value(surface_inputs.v10),
    )
    shear(meanflow, nlev, settings.time.cnpar, stokes.dusdz, stokes.dvsdz)
    do_density(density, nlev, meanflow.S, meanflow.T, -meanflow.z, -meanflow.zi)
    meanflow.buoy[1 : nlev + 1] = (
        -meanflow.gravity * (density.rho_p[1 : nlev + 1] - density.rho0) / density.rho0
    )
    stratification(meanflow, density, nlev)
    if list_fields:
        _save_snapshot(run)
    return run


def initialize_gotm(
    yaml_file: str | Path = "gotm.yaml",
    *,
    write_yaml_path: str = "",
    write_yaml_detail: int = 1,
    write_schema_path: str = "",
    output_id: str = "",
    list_fields: bool = False,
    ignore_unknown_config: bool = False,
) -> GotmRun:
    """Initialise configuration, physics state, and output registry."""

    del write_yaml_detail, output_id, ignore_unknown_config
    config_path = Path(yaml_file)
    config_only = bool(write_yaml_path or write_schema_path)

    if config_path.exists():
        config = load_config(config_path)
    elif config_only:
        config = GotmConfig.from_settings(GotmSettings(), source_path=config_path)
    else:
        msg = f"configuration file {config_path} not found"
        raise FileNotFoundError(msg)

    if write_yaml_path:
        config.save(write_yaml_path)
    if write_schema_path:
        Path(write_schema_path).write_text(
            json.dumps(GotmSettings.model_json_schema(), indent=2),
            encoding="utf-8",
        )

    return initialize_gotm_from_settings(
        config.resolved_settings(),
        yaml_path=config.source_path or config_path,
        list_fields=list_fields,
        document=config.resolved_document(),
    )


def _integrate_gotm_python(
    run: GotmRun,
    *,
    max_steps: int | None = None,
    output: bool = True,
) -> None:
    """Legacy Python timestep loop retained for focused migration parity tests."""

    if not run.initialized:
        raise RuntimeError("run has not been initialised")

    meanflow = run.meanflow
    observations = run.observations
    turbulence = run.turbulence
    airsea = run.airsea
    stokes = run.stokes_drift
    seagrass = run.seagrass
    density = run.density
    surface_inputs = run.surface_inputs

    assert meanflow.z is not None
    assert meanflow.zi is not None
    assert meanflow.h is not None
    assert meanflow.S is not None
    assert meanflow.T is not None
    assert meanflow.u is not None
    assert meanflow.v is not None
    assert meanflow.buoy is not None
    assert meanflow.drag is not None
    assert meanflow.xP is not None
    assert meanflow.NN is not None
    assert meanflow.SS is not None
    assert meanflow.SSU is not None
    assert meanflow.SSV is not None
    assert observations.idpdx is not None
    assert observations.idpdy is not None
    assert observations.SRelaxTau is not None
    assert observations.TRelaxTau is not None
    assert density.rho_p is not None
    assert turbulence.num is not None
    assert turbulence.nuh is not None
    assert turbulence.nus is not None
    assert turbulence.nucl is not None
    assert turbulence.gamu is not None
    assert turbulence.gamv is not None
    assert turbulence.gamh is not None
    assert turbulence.gams is not None
    assert turbulence.tke is not None
    assert stokes.dusdz is not None
    assert stokes.dvsdz is not None
    h = meanflow.h
    salinity_profile = meanflow.S
    temperature_profile = meanflow.T
    buoyancy = meanflow.buoy
    drag = meanflow.drag
    nn_profile = meanflow.NN
    ss_profile = meanflow.SS
    rho_p = density.rho_p

    run.snapshot_times.clear()
    run.snapshots.clear()
    last_step = run.time.MaxN if max_steps is None else min(run.time.MaxN, max_steps)

    if output and run.output_schedule.capture_initial:
        _save_snapshot(run)
    if last_step < run.time.MinN:
        return

    for step in range(run.time.MinN, last_step + 1):
        run.time.update_time(step)
        do_input(run.time.julianday, run.time.secondsofday, run.nlev, meanflow.z)
        get_all_obs(
            observations,
            run.time.julianday,
            run.time.secondsofday,
            run.nlev,
            meanflow.z,
            fsecs=run.time.fsecs,
        )

        meanflow.zeta = float(observations.zeta_input.value)

        if airsea.fluxes_method != 0:
            surface_temperature = float(
                gsw.t_from_CT(
                    meanflow.S[run.nlev],
                    meanflow.T[run.nlev],
                    0.0,
                )
            )
            set_sst(airsea, surface_temperature)
            set_ssuv(airsea, float(meanflow.u[run.nlev]), float(meanflow.v[run.nlev]))

        do_airsea(
            airsea,
            yearday=run.time.yearday,
            secs=run.time.secondsofday,
            airp=_surface_value(surface_inputs.airp),
            airt=_surface_value(surface_inputs.airt),
            hum=_surface_value(surface_inputs.hum),
            cloud=_surface_value(surface_inputs.cloud),
            u10=_surface_value(surface_inputs.u10),
            v10=_surface_value(surface_inputs.v10),
            precip=_surface_value(surface_inputs.precip),
            shortwave=_surface_value(surface_inputs.swr),
            heat=_surface_value(surface_inputs.heat),
            tx=_surface_value(surface_inputs.tx),
            ty=_surface_value(surface_inputs.ty),
            longwave=_surface_value(surface_inputs.longwave),
            sst_obs=(
                _surface_value(surface_inputs.sst_obs)
                if surface_inputs.sst_obs is not None
                else None
            ),
        )

        run.current_i0 = airsea.shortwave * (1.0 - airsea.albedo - airsea.bio_albedo)
        swf = airsea.precip + airsea.evap
        shf = -airsea.heat
        ssf = float(meanflow.S[run.nlev]) * swf
        if run.ice_params is not None and run.ice_state is not None:
            ice_state = run.ice_state
            diff_t_up = -shf / (density.rho0 * density.cp)
            diff_t_up = step_ice(
                int(run.ice_params.model),
                float(meanflow.T[run.nlev]),
                float(meanflow.S[run.nlev]),
                _surface_value(surface_inputs.airt),
                float(meanflow.h[run.nlev]),
                run.dt,
                diff_t_up,
                airsea.shortwave,
                airsea.ql,
                airsea.qh,
                airsea.qe,
                airsea.precip,
                float(meanflow.u_taus),
                ice_state.Hice,
                ice_state.Hsnow,
                ice_state.Hfrazil,
                ice_state.T1,
                ice_state.T2,
                ice_state.Tice_surface,
                ice_state.fdd,
                ice_state.ice_cover,
                ice_state.Tf,
                ice_state.albedo_ice,
                ice_state.transmissivity,
                ice_state.ocean_ice_flux,
                ice_state.ocean_ice_heat_flux,
                ice_state.ocean_ice_salt_flux,
                ice_state.surface_ice_energy,
                ice_state.bottom_ice_energy,
                ice_state.melt_rate,
                ice_state.T_melt,
                ice_state.S_melt,
            )
            shf = -diff_t_up * density.rho0 * density.cp
            ssf -= float(ice_state.ocean_ice_salt_flux[0])
            meanflow.Hice = float(ice_state.Hice[0])
            if ice_state.ice_cover[0] == 2:
                open_water_shortwave = airsea.shortwave * (
                    1.0 - float(ice_state.albedo_ice[0]) - airsea.bio_albedo
                )
                run.current_i0 = open_water_shortwave * float(
                    ice_state.transmissivity[0]
                )
        airsea.tx = airsea.tx / density.rho0
        airsea.ty = airsea.ty / density.rho0
        tx = airsea.tx
        ty = airsea.ty
        integrated_fluxes(airsea, run.dt, shortwave=run.current_i0)

        updategrid(meanflow, run.nlev, run.dt, meanflow.zeta)
        do_stokes_drift(
            stokes,
            run.nlev,
            meanflow.z,
            meanflow.zi,
            meanflow.gravity,
            _surface_value(surface_inputs.u10),
            _surface_value(surface_inputs.v10),
        )
        adjusted_w_height = wequation(
            meanflow,
            run.nlev,
            run.dt,
            observations.w_adv_input.method,
            float(observations.w_adv_input.value),
            float(observations.w_height_input.value),
        )
        observations.w_height_input.value = adjusted_w_height
        coriolis(meanflow, run.nlev, run.dt)
        do_seagrass(
            seagrass,
            run.nlev,
            run.dt,
            meanflow.u,
            meanflow.v,
            meanflow.h,
            meanflow.drag,
            meanflow.xP,
        )

        uequation(
            meanflow,
            run.nlev,
            run.dt,
            run.cnpar,
            tx,
            turbulence.num,
            turbulence.nucl,
            turbulence.gamu,
            ext_method=observations.ext_press_mode,
            dpdx=float(observations.dpdx_input.value),
            idpdx=observations.idpdx,
            dusdz=stokes.dusdz,
            w_adv_active=observations.w_adv_input.method != 0,
            w_adv_discr=observations.w_adv_discr,
            vel_relax_tau=observations.SRelaxTau * 0.0 + observations.vel_relax_tau,
            vel_relax_ramp=observations.vel_relax_ramp,
            uprof=observations.uprof_input.data,
            plume_active=(
                observations.int_press_type == 2 and observations.plume_type == 1
            ),
            seagrass_active=seagrass.seagrass_calc,
        )
        vequation(
            meanflow,
            run.nlev,
            run.dt,
            run.cnpar,
            ty,
            turbulence.num,
            turbulence.nucl,
            turbulence.gamv,
            ext_method=observations.ext_press_mode,
            dpdy=float(observations.dpdy_input.value),
            idpdy=observations.idpdy,
            dvsdz=stokes.dvsdz,
            w_adv_active=observations.w_adv_input.method != 0,
            w_adv_discr=observations.w_adv_discr,
            vel_relax_tau=observations.SRelaxTau * 0.0 + observations.vel_relax_tau,
            vel_relax_ramp=observations.vel_relax_ramp,
            vprof=observations.vprof_input.data,
            plume_active=(
                observations.int_press_type == 2 and observations.plume_type == 1
            ),
        )
        external_pressure(
            meanflow,
            run.nlev,
            observations.ext_press_mode,
            float(observations.dpdx_input.value),
            float(observations.dpdy_input.value),
            h_press=float(observations.h_press_input.value),
        )
        observations.idpdx.fill(0.0)
        observations.idpdy.fill(0.0)
        internal_pressure(
            meanflow,
            density,
            run.nlev,
            observations.idpdx,
            observations.idpdy,
            int_press_type=observations.int_press_type,
            dsdx=observations.dsdx_input.data,
            dsdy=observations.dsdy_input.data,
            dtdx=observations.dtdx_input.data,
            dtdy=observations.dtdy_input.data,
            plume_type=observations.plume_type,
            plume_slope_x=observations.plume_slope_x,
            plume_slope_y=observations.plume_slope_y,
        )
        friction(
            meanflow,
            run.nlev,
            kappa=turbulence.kappa,
            tx=tx,
            ty=ty,
            plume_type=1 if observations.plume_type == 1 else 0,
            rho0=density.rho0,
            _first=run.friction_first,
        )

        if observations.sprof_input.method != 0:
            salinity(
                meanflow,
                run.nlev,
                run.dt,
                run.cnpar,
                swf,
                ssf,
                turbulence.nus,
                turbulence.gams,
                Sobs=meanflow.Sobs,
                tau_r=observations.SRelaxTau,
                dsdx=observations.dsdx_input.data,
                dsdy=observations.dsdy_input.data,
                w_adv_active=observations.w_adv_input.method != 0,
                w_adv_discr=observations.w_adv_discr,
                s_adv=observations.s_adv,
            )
        if observations.tprof_input.method != 0:
            temperature(
                meanflow,
                run.nlev,
                run.dt,
                run.cnpar,
                run.current_i0,
                swf,
                shf,
                turbulence.nuh,
                turbulence.gamh,
                rho0=density.rho0,
                cp=density.cp,
                A=float(observations.A_input.value),
                g1=float(observations.g1_input.value),
                g2=float(observations.g2_input.value),
                Tobs=meanflow.Tobs,
                tau_r=observations.TRelaxTau,
                dtdx=observations.dtdx_input.data,
                dtdy=observations.dtdy_input.data,
                w_adv_active=observations.w_adv_input.method != 0,
                w_adv_discr=observations.w_adv_discr,
                t_adv=observations.t_adv,
            )

        _update_relaxation_targets(run)
        shear(meanflow, run.nlev, run.cnpar, stokes.dusdz, stokes.dvsdz)
        do_density(
            density,
            run.nlev,
            salinity_profile,
            temperature_profile,
            -meanflow.z,
            -meanflow.zi,
        )
        buoyancy[1 : run.nlev + 1] = (
            -meanflow.gravity * (rho_p[1 : run.nlev + 1] - density.rho0) / density.rho0
        )
        stratification(meanflow, density, run.nlev)
        do_turbulence(
            turbulence,
            run.nlev,
            run.dt,
            meanflow.depth,
            meanflow.u_taus,
            meanflow.u_taub,
            meanflow.z0s,
            meanflow.z0b,
            h,
            nn_profile,
            ss_profile,
            xP=meanflow.xP,
            SSCSTK=meanflow.SSCSTK,
            SSSTK=meanflow.SSSTK,
        )
        run_variances(turbulence, run.nlev, meanflow.SSU, meanflow.SSV)
        do_diagnostics(
            run.diagnostics,
            run.nlev,
            tx=tx,
            ty=ty,
            drag=drag,
            h=h,
            u=meanflow.u,
            v=meanflow.v,
            NN=nn_profile,
            SS=ss_profile,
            buoy=buoyancy,
            tke=turbulence.tke,
            num=turbulence.num,
            nucl=turbulence.nucl,
        )

        if output and run.output_schedule.time_method != "point":
            _accumulate_snapshot(run)

        if output and (
            step % run.output_schedule.interval_steps == 0 or step == last_step
        ):
            _save_snapshot(run)


def integrate_gotm(
    run: GotmRun,
    *,
    max_steps: int | None = None,
    output: bool = True,
) -> None:
    """Advance supported single-column cases through the compiled runtime."""

    if not run.initialized:
        raise RuntimeError("run has not been initialised")
    integrate_gotm_compiled(run, max_steps=max_steps, output=output)


def integrate_gotm_compiled(
    run: GotmRun,
    *,
    max_steps: int | None = None,
    output: bool = True,
    chunk_size: int | None = None,
) -> RuntimeBundle:
    """Advance supported single-column cases through the compiled runtime."""

    if not run.initialized:
        raise RuntimeError("run has not been initialised")

    fabm_active = run.fabm_config is not None and run.fabm_config.use
    bundle = build_runtime_from_run(run, max_steps=max_steps, output=output)
    nt = bundle.params.nt
    nlev = bundle.params.nlev

    if not fabm_active:
        written = bundle.run()
        if written < 0:
            msg = f"compiled GOTM runtime failed with status {written}"
            raise RuntimeError(msg)
        if output and written != bundle.output.nout:
            msg = (
                f"compiled GOTM runtime wrote {written} outputs, "
                f"expected {bundle.output.nout}"
            )
            raise RuntimeError(msg)
        _copy_runtime_state_to_run(run, bundle)
        run.time.update_time(nt)
        return bundle

    # --- FABM active: chunked interleaved physics+FABM loop ---

    output_every = bundle.output.output_every

    # Determine effective chunk_size
    if chunk_size is None:
        chunk_size = max(int(round(86400.0 / bundle.params.dt)), 1)
    # Snap chunk_size up to nearest multiple of output_every
    if chunk_size % output_every != 0:
        chunk_size = ((chunk_size + output_every - 1) // output_every) * output_every
    chunk_size = min(chunk_size, nt)

    # Reusable GOTM hydrodynamic state buffers (chunk_size+1 rows — one per chunk)
    hydro_T = np.zeros((chunk_size + 1, nlev + 1), dtype=np.float64)
    hydro_S = np.zeros((chunk_size + 1, nlev + 1), dtype=np.float64)
    hydro_rho = np.zeros((chunk_size + 1, nlev + 1), dtype=np.float64)
    hydro_h = np.zeros((chunk_size + 1, nlev + 1), dtype=np.float64)
    hydro_nuh = np.zeros((chunk_size + 1, nlev + 1), dtype=np.float64)
    hydro_rad = np.zeros((chunk_size + 1, nlev + 1), dtype=np.float64)
    hydro_taub = np.zeros((chunk_size + 1,), dtype=np.float64)

    # Initialise FABM engine once
    assert run.fabm_config is not None
    assert run.fabm_config.config_path is not None
    h_initial = np.ascontiguousarray(
        bundle.state.h[1 : nlev + 1], dtype=np.float64
    )  # noqa: E203
    engine = FABMEngine(run.fabm_config.config_path)
    engine.initialize(nlev=nlev, h_col=h_initial, skip_start=True)
    run.fabm_engine = engine

    cc: np.ndarray | None = None
    fabm_out_index = 0
    step_cursor = 0

    while step_cursor < nt:
        this_chunk = min(chunk_size, nt - step_cursor)

        # Build chunk params with nt=this_chunk
        chunk_params = dc_replace(bundle.params, nt=this_chunk)

        # Output slot base: step_start // output_every
        out_slot_base = step_cursor // output_every
        is_first_physics_chunk = step_cursor == 0

        # For chunks k>0: slot out_slot_base already has correct data from chunk k-1.
        # Read accumulated values to carry forward, then skip the IC write.
        if is_first_physics_chunk:
            init_int_precip = 0.0
            init_int_evap = 0.0
            init_int_swr = 0.0
            init_int_heat = 0.0
            init_int_total = 0.0
        else:
            prev_slot = out_slot_base
            init_int_precip = float(bundle.output.int_precip[prev_slot])
            init_int_evap = float(bundle.output.int_evap[prev_slot])
            init_int_swr = float(bundle.output.int_swr[prev_slot])
            init_int_heat = float(bundle.output.int_heat[prev_slot])
            init_int_total = float(bundle.output.int_total[prev_slot])

        # Resize hydrodynamic state buffers if last chunk is smaller than chunk_size
        if this_chunk < chunk_size:
            hydro_T = np.zeros((this_chunk + 1, nlev + 1), dtype=np.float64)
            hydro_S = np.zeros((this_chunk + 1, nlev + 1), dtype=np.float64)
            hydro_rho = np.zeros((this_chunk + 1, nlev + 1), dtype=np.float64)
            hydro_h = np.zeros((this_chunk + 1, nlev + 1), dtype=np.float64)
            hydro_nuh = np.zeros((this_chunk + 1, nlev + 1), dtype=np.float64)
            hydro_rad = np.zeros((this_chunk + 1, nlev + 1), dtype=np.float64)
            hydro_taub = np.zeros((this_chunk + 1,), dtype=np.float64)

        # Physics chunk
        written = run_compiled_time_loop(
            chunk_params,
            bundle.state,
            bundle.work,
            bundle.forcing,
            bundle.output,
            step_offset=step_cursor,
            out_slot_base=out_slot_base,
            write_ic=1 if is_first_physics_chunk else 0,
            init_int_precip=init_int_precip,
            init_int_evap=init_int_evap,
            init_int_swr=init_int_swr,
            init_int_heat=init_int_heat,
            init_int_total=init_int_total,
            hydro_store=1,
            hydro_T=hydro_T,
            hydro_S=hydro_S,
            hydro_rho=hydro_rho,
            hydro_h=hydro_h,
            hydro_nuh=hydro_nuh,
            hydro_rad=hydro_rad,
            hydro_taub=hydro_taub,
        )
        if written < 0:
            msg = f"compiled GOTM runtime failed with status {written}"
            raise RuntimeError(msg)

        if output:
            # FABM chunk
            cc, fabm_out_index = run_fabm_chunk(
                engine=engine,
                chunk_params=chunk_params,
                output=bundle.output,
                hydro_T=hydro_T,
                hydro_S=hydro_S,
                hydro_rho=hydro_rho,
                hydro_h=hydro_h,
                hydro_nuh=hydro_nuh,
                hydro_rad=hydro_rad,
                hydro_taub=hydro_taub,
                cc_in=cc,
                out_index_base=fabm_out_index,
                forcing_u10=bundle.forcing.u10[
                    step_cursor : step_cursor + this_chunk + 1
                ],
                forcing_v10=bundle.forcing.v10[
                    step_cursor : step_cursor + this_chunk + 1
                ],
                forcing_yearday=bundle.forcing.yearday[
                    step_cursor : step_cursor + this_chunk + 1
                ],
                forcing_secondsofday=bundle.forcing.secondsofday[
                    step_cursor : step_cursor + this_chunk + 1
                ],
                is_first_chunk=(step_cursor == 0),
            )

        step_cursor += this_chunk

    if output:
        _record_scalar_diagnostics(engine, bundle.output, fabm_out_index)

    _copy_runtime_state_to_run(run, bundle)
    run.time.update_time(nt)
    return bundle


def _copy_runtime_state_to_run(run: GotmRun, bundle: RuntimeBundle) -> None:
    state = bundle.state
    for name, array in state.iter_profile_arrays():
        if hasattr(run.meanflow, name):
            target = getattr(run.meanflow, name)
        elif hasattr(run.turbulence, name):
            target = getattr(run.turbulence, name)
        elif hasattr(run.density, name):
            target = getattr(run.density, name)
        else:
            continue
        if isinstance(target, np.ndarray):
            np.copyto(target, array)

    run.meanflow.z0b = float(state.z0b[0])
    run.meanflow.z0s = float(state.z0s[0])
    run.meanflow.za = float(state.za[0])
    run.meanflow.u_taub = float(state.u_taub[0])
    run.meanflow.u_taubo = float(state.u_taubo[0])
    run.meanflow.u_taus = float(state.u_taus[0])
    run.meanflow.taub = float(state.taub[0])
    if run.ice_state is not None:
        for name in (
            "Hice",
            "Hsnow",
            "Hfrazil",
            "T1",
            "T2",
            "Tice_surface",
            "fdd",
            "ice_cover",
            "Tf",
            "albedo_ice",
            "transmissivity",
            "ocean_ice_flux",
            "ocean_ice_heat_flux",
            "ocean_ice_salt_flux",
            "surface_ice_energy",
            "bottom_ice_energy",
            "melt_rate",
            "T_melt",
            "S_melt",
        ):
            getattr(run.ice_state, name)[0] = getattr(state, name)[0]
        run.meanflow.Hice = float(run.ice_state.Hice[0])


def finalize_gotm(run: GotmRun) -> None:
    """Release runtime state and close open input resources."""

    if not run.initialized:
        return
    close_input()
    clean_airsea(run.airsea)
    clean_stokes_drift(run.stokes_drift)
    end_seagrass(run.seagrass)
    clean_turbulence(run.turbulence)
    clean_observations(run.observations)
    clean_diagnostics(run.diagnostics)
    clean_density(run.density)
    clean_meanflow(run.meanflow)
    run.registry.finalize()
    run.initialized = False
    run.snapshot_times.clear()
    run.snapshots.clear()
    fm.clear()
