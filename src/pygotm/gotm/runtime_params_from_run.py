"""Build compiled RuntimeParams from a high-level GotmRun."""

from __future__ import annotations

from typing import Any

from pygotm.gotm.runtime_core import (
    UnsupportedConfigurationError,
)
from pygotm.gotm.runtime_params import RuntimeParams, make_runtime_params
from pygotm.meanflow.water_balance import WATER_BALANCE_ZETA


def _surface_input_value(input_: Any | None, fallback: float = 0.0) -> float:
    if input_ is None:
        return fallback
    return float(input_.value)


def _surface_input_optional_value(input_: Any | None) -> float:
    if input_ is None:
        return float("nan")
    return float(input_.value)


def _input_method(input_: Any | None) -> int:
    if input_ is None:
        return 0
    return int(getattr(input_, "method", 0))


def _run_uses_ice_zeta(run: Any) -> bool:
    ice_params = getattr(run, "ice_params", None)
    return ice_params is not None and int(getattr(ice_params, "model", 0)) != 0


def _stokes_runtime_active(stokes: Any) -> bool:
    return any(
        getattr(stokes, name) != 0
        for name in (
            "us0_method",
            "vs0_method",
            "ds_method",
            "uwnd_method",
            "vwnd_method",
            "usprof_method",
            "vsprof_method",
            "dusdz_method",
            "dvsdz_method",
        )
    )


def _validate_run_supported_by_compiled_runtime(run: Any, *, output: bool) -> None:
    unsupported: list[str] = []

    if not bool(run.output_schedule.capture_initial):
        unsupported.append("output.capture_initial=False")
    if (
        bool(getattr(run.meanflow, "lake", False))
        and int(getattr(run.meanflow, "water_balance_method", 0)) == WATER_BALANCE_ZETA
    ):
        unsupported.append("lake water_balance_method=zeta")

    if unsupported:
        names = ", ".join(unsupported)
        msg = f"compiled GOTM runtime does not yet support: {names}"
        raise UnsupportedConfigurationError(msg)


def _make_runtime_params_from_run(run: Any, nt: int) -> RuntimeParams:
    meanflow = run.meanflow
    turbulence = run.turbulence
    density = run.density
    observations = run.observations
    seagrass = run.seagrass
    stokes = run.stokes_drift
    diagnostics = run.diagnostics

    rho0 = float(density.rho0)
    tx = _surface_input_value(run.surface_inputs.tx, float(run.airsea.tx)) / rho0
    ty = _surface_input_value(run.surface_inputs.ty, float(run.airsea.ty)) / rho0

    return make_runtime_params(
        nlev=int(run.nlev),
        nt=nt,
        dt=float(run.dt),
        cnpar=float(run.cnpar),
        latitude=float(run.latitude),
        longitude=float(run.longitude),
        depth=float(run.depth),
        lake=1 if bool(getattr(meanflow, "lake", False)) else 0,
        water_balance_method=int(getattr(meanflow, "water_balance_method", 0)),
        nstreams=int(getattr(getattr(run, "streams", None), "nstreams", 0)),
        gravity=float(meanflow.gravity),
        rho0=rho0,
        cori=float(meanflow.cori),
        avmolu=float(meanflow.avmolu),
        avmolT=float(meanflow.avmolT),
        avmolS=float(meanflow.avmolS),
        cp=float(density.cp),
        h0b=float(meanflow.h0b),
        z0s_min=float(meanflow.z0s_min),
        calc_bottom_stress=1 if meanflow.calc_bottom_stress else 0,
        charnock=1 if meanflow.charnock else 0,
        charnock_val=float(meanflow.charnock_val),
        max_it_z0b=int(meanflow.MaxItz0b),
        plume_active=(
            1
            if observations.int_press_type == 2 and observations.plume_type == 1
            else 0
        ),
        int_press_type=int(observations.int_press_type),
        plume_type=int(observations.plume_type),
        plume_slope_x=float(observations.plume_slope_x),
        plume_slope_y=float(observations.plume_slope_y),
        seagrass_active=1 if seagrass.seagrass_calc else 0,
        seagrass_alpha=float(seagrass.alpha),
        seagrass_grassind=int(seagrass.grassind),
        seagrass_grassn=int(seagrass.grassn),
        stokes_active=1 if _stokes_runtime_active(stokes) else 0,
        w_adv_active=int(observations.w_adv_input.method),
        w_adv_discr=int(observations.w_adv_discr),
        s_adv=1 if observations.s_adv else 0,
        t_adv=1 if observations.t_adv else 0,
        sprof_input_active=1 if observations.sprof_input.method != 0 else 0,
        tprof_input_active=1 if observations.tprof_input.method != 0 else 0,
        uprof_input_active=1 if observations.uprof_input.method != 0 else 0,
        vprof_input_active=1 if observations.vprof_input.method != 0 else 0,
        zeta_input_active=1 if _input_method(observations.zeta_input) != 0 else 0,
        grid_method=int(meanflow.grid_method),
        airsea_fluxes_method=int(run.airsea.fluxes_method),
        airsea_hum_method=int(run.airsea.hum_method),
        airsea_shortwave_method=int(run.airsea.shortwave_method),
        airsea_shortwave_type=int(run.airsea.shortwave_type),
        airsea_longwave_method=int(run.airsea.longwave_method),
        airsea_longwave_type=int(run.airsea.longwave_type),
        airsea_albedo_method=int(run.airsea.albedo_method),
        airsea_ssuv_method=int(run.airsea.ssuv_method),
        airsea_sst_obs_method=_input_method(run.surface_inputs.sst_obs),
        airsea_shortwave_scale_factor=float(run.airsea.shortwave_scale_factor),
        airsea_heat_scale_factor=float(run.airsea.heat_scale_factor),
        airsea_const_albedo=float(run.airsea.const_albedo),
        ice_model=(
            int(run.ice_params.model)
            if getattr(run, "ice_params", None) is not None
            else 1
        ),
        turb_method=int(turbulence.turb_method),
        tke_method=int(turbulence.tke_method),
        len_scale_method=int(turbulence.len_scale_method),
        my_b1=float(turbulence.b1),
        my_sq=float(turbulence.sq),
        my_sl=float(turbulence.sl),
        my_e1=float(turbulence.e1),
        my_e2=float(turbulence.e2),
        my_e3=float(turbulence.e3),
        my_ex=float(turbulence.ex),
        my_e6=float(turbulence.e6),
        my_length=int(turbulence.my_length),
        stab_method=int(turbulence.stab_method),
        scnd_method=int(turbulence.scnd_method),
        kb_method=int(turbulence.kb_method),
        epsb_method=int(turbulence.epsb_method),
        iw_model=int(turbulence.iw_model),
        prandtl0_fix=float(turbulence.Prandtl0_fix),
        mld_method=int(diagnostics.mld_method),
        mld_diff_k=float(diagnostics.diff_k),
        mld_ri_crit=float(diagnostics.Ri_crit),
        kappa=float(turbulence.kappa),
        cm0=float(turbulence.cm0),
        cmsf=float(turbulence.cmsf),
        cde=float(turbulence.cde),
        k_min=float(turbulence.k_min),
        eps_min=float(turbulence.eps_min),
        kb_min=float(turbulence.kb_min),
        epsb_min=float(turbulence.epsb_min),
        tx=tx,
        ty=ty,
        dzetadx=float(observations.dpdx_input.value),
        dzetady=float(observations.dpdy_input.value),
        ext_press_mode=int(observations.ext_press_mode),
        vel_relax_ramp=float(observations.vel_relax_ramp),
        k_ubc=int(turbulence.k_ubc),
        k_lbc=int(turbulence.k_lbc),
        psi_ubc=int(turbulence.psi_ubc),
        psi_lbc=int(turbulence.psi_lbc),
        ubc_type=int(turbulence.ubc_type),
        lbc_type=int(turbulence.lbc_type),
        length_lim=1 if turbulence.length_lim else 0,
        sig_k=float(turbulence.sig_k),
        sig_w=float(turbulence.sig_w),
        cw=float(turbulence.cw),
        gen_alpha=float(turbulence.gen_alpha),
        gen_l=float(turbulence.gen_l),
        galp=float(turbulence.galp),
        cc1=float(turbulence.cc1),
        ct1=float(turbulence.ct1),
        ctt=float(turbulence.ctt),
        a1=float(turbulence.a1),
        a2=float(turbulence.a2),
        a3=float(turbulence.a3),
        a5=float(turbulence.a5),
        at1=float(turbulence.at1),
        at2=float(turbulence.at2),
        at3=float(turbulence.at3),
        at5=float(turbulence.at5),
        cw1=float(turbulence.cw1),
        cw2=float(turbulence.cw2),
        cw3plus=float(turbulence.cw3plus),
        cw3minus=float(turbulence.cw3minus),
        cwx=float(turbulence.cwx),
        cw4=float(turbulence.cw4),
        ce1=float(turbulence.ce1),
        ce2=float(turbulence.ce2),
        ce3plus=float(turbulence.ce3plus),
        ce3minus=float(turbulence.ce3minus),
        cex=float(turbulence.cex),
        ce4=float(turbulence.ce4),
        sig_e=float(turbulence.sig_e),
        sig_e0=float(turbulence.sig_e0),
        sig_peps=1 if turbulence.sig_peps else 0,
        iw_alpha=float(turbulence.alpha),
        klimiw=float(turbulence.klimiw),
        rich_cr=float(turbulence.rich_cr),
        numiw=float(turbulence.numiw),
        nuhiw=float(turbulence.nuhiw),
        numshear=float(turbulence.numshear),
        light_A=float(observations.A_input.value),
        light_g1=float(observations.g1_input.value),
        light_g2=float(observations.g2_input.value),
        density_method=int(density.density_method),
        rhob=float(density._rhob),
        alpha0=float(density.alpha0),
        beta0=float(density.beta0),
        T0=float(density.T0),
        S0=float(density.S0),
    )
