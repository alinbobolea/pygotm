"""Scalar runtime parameters for compiled single-column GOTM integration."""

from __future__ import annotations

from dataclasses import dataclass

from pygotm.constants import STANDARD_GRAVITY_M_S2

__all__ = ["RuntimeParams", "make_runtime_params"]


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def _require_nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class RuntimeParams:
    """Scalar parameters extracted before crossing into Numba."""

    nlev: int
    nt: int
    dt: float
    cnpar: float
    latitude: float
    longitude: float
    depth: float

    gravity: float
    rho0: float
    cori: float
    avmolu: float
    avmolT: float
    avmolS: float
    cp: float
    h0b: float
    z0s_min: float

    calc_bottom_stress: int
    charnock: int
    charnock_val: float
    max_it_z0b: int
    plume_active: int
    int_press_type: int
    plume_type: int
    plume_slope_x: float
    plume_slope_y: float
    seagrass_active: int
    seagrass_alpha: float
    seagrass_grassind: int
    seagrass_grassn: int
    stokes_active: int
    w_adv_active: int
    w_adv_discr: int
    s_adv: int
    t_adv: int
    sprof_input_active: int
    tprof_input_active: int
    uprof_input_active: int
    vprof_input_active: int
    zeta_input_active: int
    grid_method: int
    airsea_fluxes_method: int
    airsea_hum_method: int
    airsea_shortwave_method: int
    airsea_shortwave_type: int
    airsea_longwave_method: int
    airsea_longwave_type: int
    airsea_albedo_method: int
    airsea_ssuv_method: int
    airsea_sst_obs_method: int
    airsea_shortwave_scale_factor: float
    airsea_heat_scale_factor: float
    airsea_const_albedo: float

    turb_method: int
    tke_method: int
    len_scale_method: int
    my_b1: float
    my_sq: float
    my_sl: float
    my_e1: float
    my_e2: float
    my_e3: float
    my_ex: float
    my_e6: float
    my_length: int
    stab_method: int
    scnd_method: int
    kb_method: int
    epsb_method: int
    iw_model: int
    prandtl0_fix: float
    mld_method: int
    mld_diff_k: float
    mld_ri_crit: float

    kappa: float
    cm0: float
    cmsf: float
    cde: float
    k_min: float
    eps_min: float
    kb_min: float
    epsb_min: float
    tx: float
    ty: float
    dzetadx: float
    dzetady: float
    ext_press_mode: int
    vel_relax_ramp: float

    k_ubc: int
    k_lbc: int
    psi_ubc: int
    psi_lbc: int
    ubc_type: int
    lbc_type: int
    length_lim: int

    sig_k: float
    sig_w: float
    cw: float
    gen_alpha: float
    gen_l: float
    galp: float

    cc1: float
    ct1: float
    ctt: float
    a1: float
    a2: float
    a3: float
    a5: float
    at1: float
    at2: float
    at3: float
    at5: float

    cw1: float
    cw2: float
    cw3plus: float
    cw3minus: float
    cwx: float
    cw4: float

    ce1: float
    ce2: float
    ce3plus: float
    ce3minus: float
    cex: float
    ce4: float
    sig_e: float
    sig_e0: float
    sig_peps: int
    iw_alpha: float
    klimiw: float
    rich_cr: float
    numiw: float
    nuhiw: float
    numshear: float
    light_A: float
    light_g1: float
    light_g2: float

    density_method: int
    rhob: float
    alpha0: float
    beta0: float
    T0: float
    S0: float

    def __post_init__(self) -> None:
        if self.nlev < 1:
            msg = f"nlev must be at least 1, got {self.nlev}"
            raise ValueError(msg)
        _require_nonnegative_int("nt", self.nt)
        _require_positive("dt", self.dt)
        _require_positive("depth", self.depth)
        _require_positive("gravity", self.gravity)
        _require_positive("rho0", self.rho0)
        _require_positive("kappa", self.kappa)
        _require_positive("mld_diff_k", self.mld_diff_k)
        _require_positive("mld_ri_crit", self.mld_ri_crit)
        _require_nonnegative_int("max_it_z0b", self.max_it_z0b)
        _require_nonnegative_int("airsea_fluxes_method", self.airsea_fluxes_method)


def make_runtime_params(
    *,
    nlev: int,
    nt: int,
    dt: float,
    cnpar: float = 1.0,
    latitude: float = 0.0,
    longitude: float = 0.0,
    depth: float = 1.0,
    gravity: float = STANDARD_GRAVITY_M_S2,
    rho0: float = 1027.0,
    cori: float = 0.0,
    avmolu: float = 1.3e-6,
    avmolT: float = 1.4e-7,
    avmolS: float = 1.1e-9,
    cp: float = 3991.86795711963,
    h0b: float = 0.05,
    z0s_min: float = 0.02,
    calc_bottom_stress: int = 0,
    charnock: int = 0,
    charnock_val: float = 1400.0,
    max_it_z0b: int = 10,
    plume_active: int = 0,
    int_press_type: int = 0,
    plume_type: int = 2,
    plume_slope_x: float = 0.0,
    plume_slope_y: float = 0.0,
    seagrass_active: int = 0,
    seagrass_alpha: float = 0.0,
    seagrass_grassind: int = 0,
    seagrass_grassn: int = 0,
    stokes_active: int = 0,
    w_adv_active: int = 0,
    w_adv_discr: int = 4,
    s_adv: int = 0,
    t_adv: int = 0,
    sprof_input_active: int = 0,
    tprof_input_active: int = 0,
    uprof_input_active: int = 0,
    vprof_input_active: int = 0,
    zeta_input_active: int = 0,
    grid_method: int = 0,
    airsea_fluxes_method: int = 0,
    airsea_hum_method: int = 1,
    airsea_shortwave_method: int = 1,
    airsea_shortwave_type: int = 1,
    airsea_longwave_method: int = 3,
    airsea_longwave_type: int = 1,
    airsea_albedo_method: int = 0,
    airsea_ssuv_method: int = 1,
    airsea_sst_obs_method: int = 0,
    airsea_shortwave_scale_factor: float = 1.0,
    airsea_heat_scale_factor: float = 1.0,
    airsea_const_albedo: float = 0.0,
    turb_method: int = 0,
    tke_method: int = 0,
    len_scale_method: int = 0,
    my_b1: float = 0.0,
    my_sq: float = 0.2,
    my_sl: float = 0.2,
    my_e1: float = 1.8,
    my_e2: float = 1.33,
    my_e3: float = 1.8,
    my_ex: float = 1.8,
    my_e6: float = 4.0,
    my_length: int = 1,
    stab_method: int = 0,
    scnd_method: int = 0,
    kb_method: int = 0,
    epsb_method: int = 0,
    iw_model: int = 0,
    prandtl0_fix: float = 0.74,
    mld_method: int = 2,
    mld_diff_k: float = 1.0e-5,
    mld_ri_crit: float = 0.5,
    kappa: float = 0.4,
    cm0: float = 0.5477,
    cmsf: float = 1.0,
    cde: float = 0.0,
    k_min: float = 1.0e-10,
    eps_min: float = 1.0e-12,
    kb_min: float = 1.0e-10,
    epsb_min: float = 1.0e-12,
    tx: float = 0.0,
    ty: float = 0.0,
    dzetadx: float = 0.0,
    dzetady: float = 0.0,
    ext_press_mode: int = 0,
    vel_relax_ramp: float = 1.0e15,
    k_ubc: int = 1,
    k_lbc: int = 1,
    psi_ubc: int = 1,
    psi_lbc: int = 1,
    ubc_type: int = 1,
    lbc_type: int = 1,
    length_lim: int = 0,
    sig_k: float = 1.0,
    sig_w: float = 2.0,
    cw: float = 100.0,
    gen_alpha: float = -2.0,
    gen_l: float = 0.2,
    galp: float = 0.27,
    cc1: float = 0.0,
    ct1: float = 0.0,
    ctt: float = 0.0,
    a1: float = 0.0,
    a2: float = 0.0,
    a3: float = 0.0,
    a5: float = 0.0,
    at1: float = 0.0,
    at2: float = 0.0,
    at3: float = 0.0,
    at5: float = 0.0,
    cw1: float = 0.555,
    cw2: float = 0.833,
    cw3plus: float = 0.5,
    cw3minus: float = 0.0,
    cwx: float = 0.555,
    cw4: float = 0.15,
    ce1: float = 1.44,
    ce2: float = 1.92,
    ce3plus: float = 1.5,
    ce3minus: float = 0.0,
    cex: float = 1.44,
    ce4: float = 0.0,
    sig_e: float = 1.3,
    sig_e0: float = 1.3,
    sig_peps: int = 0,
    iw_alpha: float = 0.0,
    klimiw: float = 1.0e-6,
    rich_cr: float = 0.7,
    numiw: float = 1.0e-4,
    nuhiw: float = 5.0e-5,
    numshear: float = 5.0e-3,
    light_A: float = 0.58,
    light_g1: float = 0.35,
    light_g2: float = 23.0,
    density_method: int = 1,
    rhob: float = 1027.0,
    alpha0: float = 0.0,
    beta0: float = 0.0,
    T0: float = 10.0,
    S0: float = 35.0,
) -> RuntimeParams:
    """Build RuntimeParams with explicit GOTM-compatible defaults."""

    return RuntimeParams(
        nlev=nlev,
        nt=nt,
        dt=dt,
        cnpar=cnpar,
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        gravity=gravity,
        rho0=rho0,
        cori=cori,
        avmolu=avmolu,
        avmolT=avmolT,
        avmolS=avmolS,
        cp=cp,
        h0b=h0b,
        z0s_min=z0s_min,
        calc_bottom_stress=calc_bottom_stress,
        charnock=charnock,
        charnock_val=charnock_val,
        max_it_z0b=max_it_z0b,
        plume_active=plume_active,
        int_press_type=int_press_type,
        plume_type=plume_type,
        plume_slope_x=plume_slope_x,
        plume_slope_y=plume_slope_y,
        seagrass_active=seagrass_active,
        seagrass_alpha=seagrass_alpha,
        seagrass_grassind=seagrass_grassind,
        seagrass_grassn=seagrass_grassn,
        stokes_active=stokes_active,
        w_adv_active=w_adv_active,
        w_adv_discr=w_adv_discr,
        s_adv=s_adv,
        t_adv=t_adv,
        sprof_input_active=sprof_input_active,
        tprof_input_active=tprof_input_active,
        uprof_input_active=uprof_input_active,
        vprof_input_active=vprof_input_active,
        zeta_input_active=zeta_input_active,
        grid_method=grid_method,
        airsea_fluxes_method=airsea_fluxes_method,
        airsea_hum_method=airsea_hum_method,
        airsea_shortwave_method=airsea_shortwave_method,
        airsea_shortwave_type=airsea_shortwave_type,
        airsea_longwave_method=airsea_longwave_method,
        airsea_longwave_type=airsea_longwave_type,
        airsea_albedo_method=airsea_albedo_method,
        airsea_ssuv_method=airsea_ssuv_method,
        airsea_sst_obs_method=airsea_sst_obs_method,
        airsea_shortwave_scale_factor=airsea_shortwave_scale_factor,
        airsea_heat_scale_factor=airsea_heat_scale_factor,
        airsea_const_albedo=airsea_const_albedo,
        turb_method=turb_method,
        tke_method=tke_method,
        len_scale_method=len_scale_method,
        my_b1=my_b1,
        my_sq=my_sq,
        my_sl=my_sl,
        my_e1=my_e1,
        my_e2=my_e2,
        my_e3=my_e3,
        my_ex=my_ex,
        my_e6=my_e6,
        my_length=my_length,
        stab_method=stab_method,
        scnd_method=scnd_method,
        kb_method=kb_method,
        epsb_method=epsb_method,
        iw_model=iw_model,
        prandtl0_fix=prandtl0_fix,
        mld_method=mld_method,
        mld_diff_k=mld_diff_k,
        mld_ri_crit=mld_ri_crit,
        kappa=kappa,
        cm0=cm0,
        cmsf=cmsf,
        cde=cde,
        k_min=k_min,
        eps_min=eps_min,
        kb_min=kb_min,
        epsb_min=epsb_min,
        tx=tx,
        ty=ty,
        dzetadx=dzetadx,
        dzetady=dzetady,
        ext_press_mode=ext_press_mode,
        vel_relax_ramp=vel_relax_ramp,
        k_ubc=k_ubc,
        k_lbc=k_lbc,
        psi_ubc=psi_ubc,
        psi_lbc=psi_lbc,
        ubc_type=ubc_type,
        lbc_type=lbc_type,
        length_lim=length_lim,
        sig_k=sig_k,
        sig_w=sig_w,
        cw=cw,
        gen_alpha=gen_alpha,
        gen_l=gen_l,
        galp=galp,
        cc1=cc1,
        ct1=ct1,
        ctt=ctt,
        a1=a1,
        a2=a2,
        a3=a3,
        a5=a5,
        at1=at1,
        at2=at2,
        at3=at3,
        at5=at5,
        cw1=cw1,
        cw2=cw2,
        cw3plus=cw3plus,
        cw3minus=cw3minus,
        cwx=cwx,
        cw4=cw4,
        ce1=ce1,
        ce2=ce2,
        ce3plus=ce3plus,
        ce3minus=ce3minus,
        cex=cex,
        ce4=ce4,
        sig_e=sig_e,
        sig_e0=sig_e0,
        sig_peps=sig_peps,
        iw_alpha=iw_alpha,
        klimiw=klimiw,
        rich_cr=rich_cr,
        numiw=numiw,
        nuhiw=nuhiw,
        numshear=numshear,
        light_A=light_A,
        light_g1=light_g1,
        light_g2=light_g2,
        density_method=density_method,
        rhob=rhob,
        alpha0=alpha0,
        beta0=beta0,
        T0=T0,
        S0=S0,
    )
