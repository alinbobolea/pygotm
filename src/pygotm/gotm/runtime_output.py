"""Dense output buffers for compiled single-column GOTM integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pygotm.gotm.runtime_state import FloatArray

IntArray = NDArray[np.int64]

__all__ = [
    "EXTRA_SCALAR_OUTPUT_NAMES",
    "EXTRA_Z_PROFILE_OUTPUT_NAMES",
    "FABM_PROMOTABLE_EXTRA_OUTPUT_NAMES",
    "IntArray",
    "OutputAttrs",
    "RuntimeOutput",
    "allocate_runtime_output",
]


OutputAttrs = dict[str, str]

EXTRA_SCALAR_OUTPUT_NAMES = (
    "Hfrazil",
    "Hice",
    "T1",
    "T2",
    "Tf",
    "Tice_surface",
    "bottom_ice_energy",
    "ocean_ice_flux",
    "ocean_ice_heat_flux",
    "ocean_ice_salt_flux",
    "surface_albedo",
    "surface_drag_coefficient_in_air",
    "surface_ice_energy",
)

EXTRA_Z_PROFILE_OUTPUT_NAMES = (
    "attenuation_coefficient_of_photosynthetic_radiative_flux",
    "eps_obs",
)

FABM_PROMOTABLE_EXTRA_OUTPUT_NAMES = (
    "attenuation_coefficient_of_photosynthetic_radiative_flux",
    "surface_albedo",
    "surface_drag_coefficient_in_air",
)


def _nout(nt: int, output_every: int, enabled: bool, force_final: bool) -> int:
    if not enabled:
        return 0
    if nt < 0:
        msg = f"nt must be non-negative, got {nt}"
        raise ValueError(msg)
    if output_every < 1:
        msg = f"output_every must be at least 1, got {output_every}"
        raise ValueError(msg)

    periodic = nt // output_every
    final_extra = 1 if force_final and nt % output_every else 0
    return 1 + periodic + final_extra


def _output_profile(nout: int, nlev: int) -> FloatArray:
    return np.zeros((nout, nlev + 1), dtype=np.float64)


def _output_scalar(nout: int) -> FloatArray:
    return np.zeros(nout, dtype=np.float64)


def _validate_output_name(name: str) -> None:
    if not name:
        msg = "output variable names must be non-empty"
        raise ValueError(msg)
    if "/" in name:
        msg = f"output variable {name!r} must use normalized '_' separators"
        raise ValueError(msg)


def _validate_scalar_outputs(
    mapping: dict[str, FloatArray],
    *,
    nout: int,
    allowed_names: tuple[str, ...] | None = None,
) -> None:
    allowed = None if allowed_names is None else set(allowed_names)
    for name, array in mapping.items():
        _validate_output_name(name)
        if allowed is not None and name not in allowed:
            msg = f"unknown scalar output {name!r}"
            raise ValueError(msg)
        if array.dtype != np.float64:
            msg = f"{name} must have dtype float64, got {array.dtype}"
            raise TypeError(msg)
        if array.shape != (nout,):
            msg = f"{name} must have shape ({nout},), got {array.shape}"
            raise ValueError(msg)
        if not array.flags.c_contiguous:
            msg = f"{name} must be C-contiguous"
            raise ValueError(msg)


def _validate_profile_outputs(
    mapping: dict[str, FloatArray],
    *,
    shape: tuple[int, int],
    allowed_names: tuple[str, ...] | None = None,
) -> None:
    allowed = None if allowed_names is None else set(allowed_names)
    for name, array in mapping.items():
        _validate_output_name(name)
        if allowed is not None and name not in allowed:
            msg = f"unknown z-profile output {name!r}"
            raise ValueError(msg)
        if array.dtype != np.float64:
            msg = f"{name} must have dtype float64, got {array.dtype}"
            raise TypeError(msg)
        if array.shape != shape:
            msg = f"{name} must have shape {shape}, got {array.shape}"
            raise ValueError(msg)
        if not array.flags.c_contiguous:
            msg = f"{name} must be C-contiguous"
            raise ValueError(msg)


def _validate_output_disjointness(
    extra_scalars: dict[str, FloatArray],
    extra_z_profiles: dict[str, FloatArray],
    fabm_scalars: dict[str, FloatArray],
    fabm_z_profiles: dict[str, FloatArray],
) -> None:
    extra_names = set(extra_scalars) | set(extra_z_profiles)
    fabm_names = set(fabm_scalars) | set(fabm_z_profiles)
    duplicate_extra = set(extra_scalars) & set(extra_z_profiles)
    duplicate_fabm = set(fabm_scalars) & set(fabm_z_profiles)
    collisions = extra_names & fabm_names
    if duplicate_extra:
        name = sorted(duplicate_extra)[0]
        msg = f"extra output {name!r} declared as both scalar and z-profile"
        raise ValueError(msg)
    if duplicate_fabm:
        name = sorted(duplicate_fabm)[0]
        msg = f"FABM output {name!r} declared as both scalar and z-profile"
        raise ValueError(msg)
    if collisions:
        name = sorted(collisions)[0]
        msg = f"FABM output {name!r} collides with a core optional output"
        raise ValueError(msg)


@dataclass(slots=True)
class RuntimeOutput:
    """Preallocated dense output buffers filled inside the compiled loop."""

    enabled: bool
    output_every: int
    force_final: bool
    nout: int
    output_step: IntArray

    time: FloatArray
    zeta: FloatArray
    u_taus: FloatArray
    u10: FloatArray
    v10: FloatArray
    airt: FloatArray
    airp: FloatArray
    hum: FloatArray
    es: FloatArray
    ea: FloatArray
    qs: FloatArray
    qa: FloatArray
    rhoa: FloatArray
    cloud: FloatArray
    albedo: FloatArray
    precip: FloatArray
    evap: FloatArray
    int_precip: FloatArray
    int_evap: FloatArray
    int_swr: FloatArray
    int_heat: FloatArray
    int_total: FloatArray
    I_0: FloatArray
    qh: FloatArray
    qe: FloatArray
    ql: FloatArray
    heat: FloatArray
    tx: FloatArray
    ty: FloatArray
    sst: FloatArray
    sst_obs: FloatArray
    sss: FloatArray
    mld_surf: FloatArray
    u_taub: FloatArray
    taub: FloatArray
    mld_bott: FloatArray
    us0: FloatArray
    vs0: FloatArray
    ds: FloatArray
    Ekin: FloatArray
    Epot: FloatArray
    Eturb: FloatArray
    extra_scalars: dict[str, FloatArray]
    fabm_scalars: dict[str, FloatArray]

    rho_p: FloatArray
    rho: FloatArray
    u: FloatArray
    v: FloatArray
    T: FloatArray
    S: FloatArray
    Tp: FloatArray
    Ti: FloatArray
    Sp: FloatArray
    Tobs: FloatArray
    Sobs: FloatArray
    u_obs: FloatArray
    v_obs: FloatArray
    idpdx: FloatArray
    idpdy: FloatArray
    tke: FloatArray
    eps: FloatArray
    num: FloatArray
    nuh: FloatArray
    h: FloatArray
    xP: FloatArray
    fric: FloatArray
    drag: FloatArray
    avh: FloatArray
    bioshade: FloatArray
    ga: FloatArray
    uu: FloatArray
    vv: FloatArray
    ww: FloatArray
    NN: FloatArray
    NNT: FloatArray
    NNS: FloatArray
    buoy: FloatArray
    SS: FloatArray
    P: FloatArray
    B: FloatArray
    Pb: FloatArray
    kb: FloatArray
    epsb: FloatArray
    L: FloatArray
    PSTK: FloatArray
    cmue1: FloatArray
    cmue2: FloatArray
    as_: FloatArray
    an: FloatArray
    at: FloatArray
    gamu: FloatArray
    gamv: FloatArray
    gamh: FloatArray
    gams: FloatArray
    Rig: FloatArray
    gamb: FloatArray
    gam: FloatArray
    r: FloatArray
    taux: FloatArray
    tauy: FloatArray
    rad: FloatArray
    us: FloatArray
    vs: FloatArray
    dusdz: FloatArray
    dvsdz: FloatArray
    nus: FloatArray
    nucl: FloatArray
    z: FloatArray
    zi: FloatArray
    extra_z_profiles: dict[str, FloatArray]
    fabm_z_profiles: dict[str, FloatArray]
    fabm_attrs: dict[str, OutputAttrs]

    def declare_fabm_output(
        self,
        name: str,
        nlev: int,
        *,
        z_profile: bool,
        attrs: OutputAttrs | None = None,
        replace_extra: bool = False,
    ) -> None:
        """Allocate one dynamic FABM output buffer."""

        _validate_output_name(name)
        if name in self.fabm_scalars or name in self.fabm_z_profiles:
            msg = f"FABM output variable {name!r} is already declared"
            raise ValueError(msg)
        if name in self.extra_scalars or name in self.extra_z_profiles:
            if not replace_extra:
                msg = f"FABM output {name!r} collides with a core optional output"
                raise ValueError(msg)
            if name not in FABM_PROMOTABLE_EXTRA_OUTPUT_NAMES:
                msg = f"FABM output {name!r} cannot replace a core optional output"
                raise ValueError(msg)
            if z_profile:
                if name in self.extra_scalars:
                    msg = (
                        f"FABM z-profile output {name!r} collides with a scalar "
                        "core optional output"
                    )
                    raise ValueError(msg)
                del self.extra_z_profiles[name]
            else:
                if name in self.extra_z_profiles:
                    msg = (
                        f"FABM scalar output {name!r} collides with a z-profile "
                        "core optional output"
                    )
                    raise ValueError(msg)
                del self.extra_scalars[name]
        if z_profile:
            self.fabm_z_profiles[name] = _output_profile(self.nout, nlev)
        else:
            self.fabm_scalars[name] = _output_scalar(self.nout)
        if attrs:
            self.fabm_attrs[name] = {
                key: value for key, value in attrs.items() if value
            }

    def validate(self, nlev: int) -> None:
        """Raise if output buffers do not match the runtime grid."""

        if self.output_every < 1:
            msg = f"output_every must be at least 1, got {self.output_every}"
            raise ValueError(msg)
        if self.nout != self.output_step.shape[0]:
            msg = "nout must equal output_step length"
            raise ValueError(msg)
        if self.time.shape != (self.nout,):
            msg = f"time must have shape ({self.nout},), got {self.time.shape}"
            raise ValueError(msg)
        for name in (
            "zeta",
            "u_taus",
            "u10",
            "v10",
            "airt",
            "airp",
            "hum",
            "es",
            "ea",
            "qs",
            "qa",
            "rhoa",
            "cloud",
            "albedo",
            "precip",
            "evap",
            "int_precip",
            "int_evap",
            "int_swr",
            "int_heat",
            "int_total",
            "I_0",
            "qh",
            "qe",
            "ql",
            "heat",
            "tx",
            "ty",
            "sst",
            "sst_obs",
            "sss",
            "mld_surf",
            "u_taub",
            "taub",
            "mld_bott",
            "us0",
            "vs0",
            "ds",
            "Ekin",
            "Epot",
            "Eturb",
        ):
            array = getattr(self, name)
            if array.dtype != np.float64:
                msg = f"{name} must have dtype float64, got {array.dtype}"
                raise TypeError(msg)
            if array.shape != (self.nout,):
                msg = f"{name} must have shape ({self.nout},), got {array.shape}"
                raise ValueError(msg)
            if not array.flags.c_contiguous:
                msg = f"{name} must be C-contiguous"
                raise ValueError(msg)
        _validate_scalar_outputs(
            self.extra_scalars,
            nout=self.nout,
            allowed_names=EXTRA_SCALAR_OUTPUT_NAMES,
        )
        _validate_scalar_outputs(self.fabm_scalars, nout=self.nout)
        expected_profile_shape = (self.nout, nlev + 1)
        for name in (
            "rho_p",
            "rho",
            "u",
            "v",
            "T",
            "S",
            "Tp",
            "Ti",
            "Sp",
            "Tobs",
            "Sobs",
            "u_obs",
            "v_obs",
            "idpdx",
            "idpdy",
            "tke",
            "eps",
            "num",
            "nuh",
            "h",
            "xP",
            "fric",
            "drag",
            "avh",
            "bioshade",
            "ga",
            "uu",
            "vv",
            "ww",
            "NN",
            "NNT",
            "NNS",
            "buoy",
            "SS",
            "P",
            "B",
            "Pb",
            "kb",
            "epsb",
            "L",
            "PSTK",
            "cmue1",
            "cmue2",
            "as_",
            "an",
            "at",
            "gamu",
            "gamv",
            "gamh",
            "gams",
            "Rig",
            "gamb",
            "gam",
            "r",
            "taux",
            "tauy",
            "rad",
            "us",
            "vs",
            "dusdz",
            "dvsdz",
            "nus",
            "nucl",
            "z",
            "zi",
        ):
            array = getattr(self, name)
            if array.dtype != np.float64:
                msg = f"{name} must have dtype float64, got {array.dtype}"
                raise TypeError(msg)
            if array.shape != expected_profile_shape:
                msg = (
                    f"{name} must have shape {expected_profile_shape}, "
                    f"got {array.shape}"
                )
                raise ValueError(msg)
            if not array.flags.c_contiguous:
                msg = f"{name} must be C-contiguous"
                raise ValueError(msg)
        _validate_profile_outputs(
            self.extra_z_profiles,
            shape=expected_profile_shape,
            allowed_names=EXTRA_Z_PROFILE_OUTPUT_NAMES,
        )
        _validate_profile_outputs(
            self.fabm_z_profiles,
            shape=expected_profile_shape,
        )
        _validate_output_disjointness(
            self.extra_scalars,
            self.extra_z_profiles,
            self.fabm_scalars,
            self.fabm_z_profiles,
        )
        declared_fabm_names = set(self.fabm_scalars) | set(self.fabm_z_profiles)
        unknown_attrs = set(self.fabm_attrs) - declared_fabm_names
        if unknown_attrs:
            name = sorted(unknown_attrs)[0]
            msg = f"FABM output attributes declared for unknown output {name!r}"
            raise ValueError(msg)


def allocate_runtime_output(
    nlev: int,
    nt: int,
    *,
    enabled: bool = True,
    output_every: int = 1,
    force_final: bool = True,
) -> RuntimeOutput:
    """Allocate dense output arrays for initial, periodic, and final states."""

    if nlev < 1:
        msg = f"nlev must be at least 1, got {nlev}"
        raise ValueError(msg)
    nout = _nout(nt, output_every, enabled, force_final)
    output = RuntimeOutput(
        enabled=enabled,
        output_every=output_every,
        force_final=force_final,
        nout=nout,
        output_step=np.full(nout, -1, dtype=np.int64),
        time=np.full(nout, np.nan, dtype=np.float64),
        zeta=_output_scalar(nout),
        u_taus=_output_scalar(nout),
        u10=_output_scalar(nout),
        v10=_output_scalar(nout),
        airt=_output_scalar(nout),
        airp=_output_scalar(nout),
        hum=_output_scalar(nout),
        es=_output_scalar(nout),
        ea=_output_scalar(nout),
        qs=_output_scalar(nout),
        qa=_output_scalar(nout),
        rhoa=_output_scalar(nout),
        cloud=_output_scalar(nout),
        albedo=_output_scalar(nout),
        precip=_output_scalar(nout),
        evap=_output_scalar(nout),
        int_precip=_output_scalar(nout),
        int_evap=_output_scalar(nout),
        int_swr=_output_scalar(nout),
        int_heat=_output_scalar(nout),
        int_total=_output_scalar(nout),
        I_0=_output_scalar(nout),
        qh=_output_scalar(nout),
        qe=_output_scalar(nout),
        ql=_output_scalar(nout),
        heat=_output_scalar(nout),
        tx=_output_scalar(nout),
        ty=_output_scalar(nout),
        sst=_output_scalar(nout),
        sst_obs=_output_scalar(nout),
        sss=_output_scalar(nout),
        mld_surf=_output_scalar(nout),
        u_taub=_output_scalar(nout),
        taub=_output_scalar(nout),
        mld_bott=_output_scalar(nout),
        us0=_output_scalar(nout),
        vs0=_output_scalar(nout),
        ds=_output_scalar(nout),
        Ekin=_output_scalar(nout),
        Epot=_output_scalar(nout),
        Eturb=_output_scalar(nout),
        extra_scalars={
            name: _output_scalar(nout) for name in EXTRA_SCALAR_OUTPUT_NAMES
        },
        fabm_scalars={},
        rho_p=_output_profile(nout, nlev),
        rho=_output_profile(nout, nlev),
        u=_output_profile(nout, nlev),
        v=_output_profile(nout, nlev),
        T=_output_profile(nout, nlev),
        S=_output_profile(nout, nlev),
        Tp=_output_profile(nout, nlev),
        Ti=_output_profile(nout, nlev),
        Sp=_output_profile(nout, nlev),
        Tobs=_output_profile(nout, nlev),
        Sobs=_output_profile(nout, nlev),
        u_obs=_output_profile(nout, nlev),
        v_obs=_output_profile(nout, nlev),
        idpdx=_output_profile(nout, nlev),
        idpdy=_output_profile(nout, nlev),
        tke=_output_profile(nout, nlev),
        eps=_output_profile(nout, nlev),
        num=_output_profile(nout, nlev),
        nuh=_output_profile(nout, nlev),
        h=_output_profile(nout, nlev),
        xP=_output_profile(nout, nlev),
        fric=_output_profile(nout, nlev),
        drag=_output_profile(nout, nlev),
        avh=_output_profile(nout, nlev),
        bioshade=_output_profile(nout, nlev),
        ga=_output_profile(nout, nlev),
        uu=_output_profile(nout, nlev),
        vv=_output_profile(nout, nlev),
        ww=_output_profile(nout, nlev),
        NN=_output_profile(nout, nlev),
        NNT=_output_profile(nout, nlev),
        NNS=_output_profile(nout, nlev),
        buoy=_output_profile(nout, nlev),
        SS=_output_profile(nout, nlev),
        P=_output_profile(nout, nlev),
        B=_output_profile(nout, nlev),
        Pb=_output_profile(nout, nlev),
        kb=_output_profile(nout, nlev),
        epsb=_output_profile(nout, nlev),
        L=_output_profile(nout, nlev),
        PSTK=_output_profile(nout, nlev),
        cmue1=_output_profile(nout, nlev),
        cmue2=_output_profile(nout, nlev),
        as_=_output_profile(nout, nlev),
        an=_output_profile(nout, nlev),
        at=_output_profile(nout, nlev),
        gamu=_output_profile(nout, nlev),
        gamv=_output_profile(nout, nlev),
        gamh=_output_profile(nout, nlev),
        gams=_output_profile(nout, nlev),
        Rig=_output_profile(nout, nlev),
        gamb=_output_profile(nout, nlev),
        gam=_output_profile(nout, nlev),
        r=_output_profile(nout, nlev),
        taux=_output_profile(nout, nlev),
        tauy=_output_profile(nout, nlev),
        rad=_output_profile(nout, nlev),
        us=_output_profile(nout, nlev),
        vs=_output_profile(nout, nlev),
        dusdz=_output_profile(nout, nlev),
        dvsdz=_output_profile(nout, nlev),
        nus=_output_profile(nout, nlev),
        nucl=_output_profile(nout, nlev),
        z=_output_profile(nout, nlev),
        zi=_output_profile(nout, nlev),
        extra_z_profiles={
            name: _output_profile(nout, nlev) for name in EXTRA_Z_PROFILE_OUTPUT_NAMES
        },
        fabm_z_profiles={},
        fabm_attrs={},
    )
    output.validate(nlev)
    return output
