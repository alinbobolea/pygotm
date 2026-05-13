"""Numba-compiled central timestep loop for single-column GOTM runs."""

import math

import numba
import numpy as np

from pygotm.constants import DEG_TO_RAD, PI, RAD_TO_DEG, SOLAR_CONSTANT_W_M2
from pygotm.gotm.runtime_forcing import RuntimeForcing
from pygotm.gotm.runtime_output import RuntimeOutput
from pygotm.gotm.runtime_params import RuntimeParams
from pygotm.gotm.runtime_state import RuntimeState
from pygotm.gotm.runtime_work import RuntimeWork
from pygotm.meanflow.coriolis import step_coriolis_single
from pygotm.meanflow.friction import step_friction_single
from pygotm.meanflow.salinity import step_salinity_single
from pygotm.meanflow.shear import step_shear_single
from pygotm.meanflow.temperature import step_temperature_single
from pygotm.meanflow.uequation import step_uequation_single
from pygotm.meanflow.updategrid import step_updategrid_single
from pygotm.meanflow.vequation import step_vequation_single
from pygotm.turbulence.alpha_mnb import step_alpha_mnb_single
from pygotm.turbulence.cmue_c import step_cmue_c_single
from pygotm.turbulence.cmue_d import step_cmue_d_single
from pygotm.turbulence.cmue_ma import _step_cmue_ma as _step_cmue_ma_single
from pygotm.turbulence.cmue_sg import _step_cmue_sg as _step_cmue_sg_single
from pygotm.turbulence.dissipationeq import step_dissipationeq_single
from pygotm.turbulence.epsbalgebraic import step_epsbalgebraic_single
from pygotm.turbulence.internal_wave import step_internal_wave_single
from pygotm.turbulence.kbalgebraic import step_kbalgebraic_single
from pygotm.turbulence.lengthscaleeq import (
    _step_lengthscaleeq as _step_lengthscaleeq_single,
)
from pygotm.turbulence.omegaeq import step_omegaeq_single
from pygotm.turbulence.production import step_production_single
from pygotm.turbulence.q2over2eq import _step_q2over2eq as _step_q2over2eq_single
from pygotm.turbulence.tkeeq import step_tkeeq_single
from pygotm.turbulence.variances import step_variances_single
from pygotm.util.gsw import gsw_alpha, gsw_beta, gsw_rho, gsw_sigma0

__all__ = [
    "run_compiled_time_loop",
    "time_loop_compiled",
    "warmup_couette_step_routines",
]

_KELVIN = 273.15
_CONST06 = 0.62198
_RGAS = 287.1
_CPA = 1008.0
_CPW = 3985.0
_AIRSEA_RHO0 = 1025.0
_EMISS = 0.97
_BOLZ = 5.670374419e-8
_LONG = 1.0e15
_GSW_CP0 = 3991.86795711963
_GSW_SFAC = 0.0248826675584615
_GSW_UPS = 35.16504 / 35.0
_GSW_T0 = 273.15

# Fairall bulk-flux constants (Fairall et al. 1996)
_FAIRALL_G = 9.81  # gravitational acceleration [m s⁻²]
_FAIRALL_KAPPA = 0.41  # von Karman constant
_FAIRALL_FDG = 1.0  # Fairall LKB roughness Reynolds to von Karman
_FAIRALL_BETA = 1.2  # gustiness parameter
_FAIRALL_ZABL = 600.0  # atmospheric boundary layer height [m]
_FAIRALL_R3 = 1.0 / 3.0
_FAIRALL_ZT = 2.0  # temperature measurement height [m]
_FAIRALL_ZQ = 2.0  # humidity measurement height [m]
_FAIRALL_ZW = 10.0  # wind measurement height [m]
_FAIRALL_WGUST = 0.0  # gustiness wind speed [m s⁻¹]
_FAIRALL_ITERMAX = 20
# Liu et al. (1979) look-up table: rt = A[:,0]*Rr**B[:,0], rq = A[:,1]*Rr**B[:,1]
_FAIRALL_LIU_A0 = np.array(
    [0.177, 1.376, 1.026, 1.625, 4.661, 34.904, 1667.190, 588000.0]
)
_FAIRALL_LIU_A1 = np.array(
    [0.292, 1.808, 1.393, 1.956, 4.994, 30.709, 1448.680, 298000.0]
)
_FAIRALL_LIU_B0 = np.array([0.0, 0.929, -0.599, -1.018, -1.475, -2.067, -2.907, -3.935])
_FAIRALL_LIU_B1 = np.array([0.0, 0.826, -0.528, -0.870, -1.297, -1.845, -2.682, -3.616])
_FAIRALL_LIU_RR = np.array([0.0, 0.11, 0.825, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])

_CLOUD_CORRECTION_FACTOR = (
    0.497202,
    0.501885,
    0.506568,
    0.511250,
    0.515933,
    0.520616,
    0.525299,
    0.529982,
    0.534665,
    0.539348,
    0.544031,
    0.548714,
    0.553397,
    0.558080,
    0.562763,
    0.567446,
    0.572129,
    0.576812,
    0.581495,
    0.586178,
    0.590861,
    0.595544,
    0.600227,
    0.604910,
    0.609593,
    0.614276,
    0.618959,
    0.623641,
    0.628324,
    0.633007,
    0.637690,
    0.642373,
    0.647056,
    0.651739,
    0.656422,
    0.661105,
    0.665788,
    0.670471,
    0.675154,
    0.679837,
    0.684520,
    0.689203,
    0.693886,
    0.698569,
    0.703252,
    0.707935,
    0.712618,
    0.717301,
    0.721984,
    0.726667,
    0.731350,
    0.736032,
    0.740715,
    0.745398,
    0.750081,
    0.754764,
    0.759447,
    0.764130,
    0.768813,
    0.773496,
    0.778179,
    0.782862,
    0.787545,
    0.792228,
    0.796911,
    0.801594,
    0.806277,
    0.810960,
    0.815643,
    0.820326,
    0.825009,
    0.829692,
    0.834375,
    0.839058,
    0.843741,
    0.848423,
    0.853106,
    0.857789,
    0.862472,
    0.867155,
    0.871838,
    0.876521,
    0.881204,
    0.885887,
    0.890570,
    0.895253,
    0.899936,
    0.904619,
    0.909302,
    0.913985,
)
_PAYNE_ALBEDO = (
    0.719,
    0.656,
    0.603,
    0.480,
    0.385,
    0.300,
    0.250,
    0.193,
    0.164,
    0.131,
    0.103,
    0.084,
    0.071,
    0.061,
    0.054,
    0.043,
    0.039,
    0.036,
    0.034,
    0.034,
)
_PAYNE_ZA = (
    90.0,
    88.0,
    86.0,
    84.0,
    82.0,
    80.0,
    78.0,
    76.0,
    74.0,
    70.0,
    66.0,
    62.0,
    58.0,
    54.0,
    50.0,
    40.0,
    30.0,
    20.0,
    10.0,
    0.0,
)
_PAYNE_DZA = (
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    4.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
)


@numba.njit(cache=True, fastmath=False)
def _compute_mld_single(
    nlev: int,
    mld_method: int,
    diff_k: float,
    ri_crit: float,
    turb_method: int,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    tke: np.ndarray,
) -> tuple[float, float]:
    mld_surf = 0.0
    mld_bott = 0.0

    if mld_method == 1:
        if turb_method != 100:
            for i in range(nlev, 0, -1):
                if tke[i] < diff_k:
                    break
                mld_surf += h[i]
            for i in range(1, nlev + 1):
                if tke[i] < diff_k:
                    break
                mld_bott += h[i]
    elif mld_method == 2:
        mld_surf = h[nlev]
        for i in range(nlev - 1, 0, -1):
            if NN[i] / (SS[i] + 1.0e-10) > ri_crit:
                break
            mld_surf += h[i]
    elif mld_method == 3:
        index = 1
        max_nn = NN[1]
        for i in range(2, nlev + 1):
            if NN[i] > max_nn:
                max_nn = NN[i]
                index = i
        for i in range(index, nlev + 1):
            mld_surf += h[i]

    return mld_surf, mld_bott


@numba.njit(cache=True, fastmath=False)
def _write_output_slot(
    slot: int,
    step: int,
    time_value: float,
    nlev: int,
    rho0_value: float,
    zeta_value: float,
    u10_value: float,
    v10_value: float,
    airt_value: float,
    airp_value: float,
    hum_value: float,
    es_value: float,
    ea_value: float,
    qs_value: float,
    qa_value: float,
    rhoa_value: float,
    cloud_value: float,
    albedo_value: float,
    precip_value: float,
    evap_value: float,
    int_precip_value: float,
    int_evap_value: float,
    int_swr_value: float,
    int_heat_value: float,
    int_total_value: float,
    i0_value: float,
    qh_value: float,
    qe_value: float,
    ql_value: float,
    heat_value: float,
    tx_surface_value: float,
    ty_surface_value: float,
    sst_value: float,
    sst_obs_value: float,
    sss_value: float,
    mld_surf_value: float,
    mld_bott_value: float,
    us0_value: float,
    vs0_value: float,
    ds_value: float,
    ekin_value: float,
    epot_value: float,
    eturb_value: float,
    u_taus: np.ndarray,
    u_taub: np.ndarray,
    taub: np.ndarray,
    rho_p: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    Tp: np.ndarray,
    Ti: np.ndarray,
    Sp: np.ndarray,
    Tobs: np.ndarray,
    Sobs: np.ndarray,
    u_obs: np.ndarray,
    v_obs: np.ndarray,
    idpdx: np.ndarray,
    idpdy: np.ndarray,
    tke: np.ndarray,
    eps: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    h: np.ndarray,
    xP: np.ndarray,
    fric: np.ndarray,
    drag: np.ndarray,
    avh: np.ndarray,
    bioshade: np.ndarray,
    ga: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    ww: np.ndarray,
    NN: np.ndarray,
    NNT: np.ndarray,
    NNS: np.ndarray,
    buoy: np.ndarray,
    SS: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Pb: np.ndarray,
    kb: np.ndarray,
    epsb: np.ndarray,
    L: np.ndarray,
    PSTK: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    gamh: np.ndarray,
    gams: np.ndarray,
    rad: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    dusdz: np.ndarray,
    dvsdz: np.ndarray,
    nus: np.ndarray,
    nucl: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    output_step: np.ndarray,
    output_time: np.ndarray,
    output_zeta: np.ndarray,
    output_u_taus: np.ndarray,
    output_u10: np.ndarray,
    output_v10: np.ndarray,
    output_airt: np.ndarray,
    output_airp: np.ndarray,
    output_hum: np.ndarray,
    output_es: np.ndarray,
    output_ea: np.ndarray,
    output_qs: np.ndarray,
    output_qa: np.ndarray,
    output_rhoa: np.ndarray,
    output_cloud: np.ndarray,
    output_albedo: np.ndarray,
    output_precip: np.ndarray,
    output_evap: np.ndarray,
    output_int_precip: np.ndarray,
    output_int_evap: np.ndarray,
    output_int_swr: np.ndarray,
    output_int_heat: np.ndarray,
    output_int_total: np.ndarray,
    output_I_0: np.ndarray,
    output_qh: np.ndarray,
    output_qe: np.ndarray,
    output_ql: np.ndarray,
    output_heat: np.ndarray,
    output_tx: np.ndarray,
    output_ty: np.ndarray,
    output_sst: np.ndarray,
    output_sst_obs: np.ndarray,
    output_sss: np.ndarray,
    output_mld_surf: np.ndarray,
    output_u_taub: np.ndarray,
    output_taub: np.ndarray,
    output_mld_bott: np.ndarray,
    output_us0: np.ndarray,
    output_vs0: np.ndarray,
    output_ds: np.ndarray,
    output_Ekin: np.ndarray,
    output_Epot: np.ndarray,
    output_Eturb: np.ndarray,
    output_rho_p: np.ndarray,
    output_rho: np.ndarray,
    output_u: np.ndarray,
    output_v: np.ndarray,
    output_T: np.ndarray,
    output_S: np.ndarray,
    output_Tp: np.ndarray,
    output_Ti: np.ndarray,
    output_Sp: np.ndarray,
    output_Tobs: np.ndarray,
    output_Sobs: np.ndarray,
    output_u_obs: np.ndarray,
    output_v_obs: np.ndarray,
    output_idpdx: np.ndarray,
    output_idpdy: np.ndarray,
    output_tke: np.ndarray,
    output_eps: np.ndarray,
    output_num: np.ndarray,
    output_nuh: np.ndarray,
    output_h: np.ndarray,
    output_xP: np.ndarray,
    output_fric: np.ndarray,
    output_drag: np.ndarray,
    output_avh: np.ndarray,
    output_bioshade: np.ndarray,
    output_ga: np.ndarray,
    output_uu: np.ndarray,
    output_vv: np.ndarray,
    output_ww: np.ndarray,
    output_NN: np.ndarray,
    output_NNT: np.ndarray,
    output_NNS: np.ndarray,
    output_buoy: np.ndarray,
    output_SS: np.ndarray,
    output_P: np.ndarray,
    output_B: np.ndarray,
    output_Pb: np.ndarray,
    output_kb: np.ndarray,
    output_epsb: np.ndarray,
    output_L: np.ndarray,
    output_PSTK: np.ndarray,
    output_cmue1: np.ndarray,
    output_cmue2: np.ndarray,
    output_as: np.ndarray,
    output_an: np.ndarray,
    output_at: np.ndarray,
    output_gamu: np.ndarray,
    output_gamv: np.ndarray,
    output_gamh: np.ndarray,
    output_gams: np.ndarray,
    output_Rig: np.ndarray,
    output_gamb: np.ndarray,
    output_gam: np.ndarray,
    output_r: np.ndarray,
    output_taux: np.ndarray,
    output_tauy: np.ndarray,
    output_rad: np.ndarray,
    output_us: np.ndarray,
    output_vs: np.ndarray,
    output_dusdz: np.ndarray,
    output_dvsdz: np.ndarray,
    output_nus: np.ndarray,
    output_nucl: np.ndarray,
    output_z: np.ndarray,
    output_zi: np.ndarray,
) -> None:
    output_step[slot] = step
    output_time[slot] = time_value
    output_zeta[slot] = zeta_value
    output_u_taus[slot] = u_taus[0]
    output_u10[slot] = u10_value
    output_v10[slot] = v10_value
    output_airt[slot] = airt_value
    output_airp[slot] = airp_value
    output_hum[slot] = hum_value
    output_es[slot] = es_value
    output_ea[slot] = ea_value
    output_qs[slot] = qs_value
    output_qa[slot] = qa_value
    output_rhoa[slot] = rhoa_value
    output_cloud[slot] = cloud_value
    output_albedo[slot] = albedo_value
    output_precip[slot] = precip_value
    output_evap[slot] = evap_value
    output_int_precip[slot] = int_precip_value
    output_int_evap[slot] = int_evap_value
    output_int_swr[slot] = int_swr_value
    output_int_heat[slot] = int_heat_value
    output_int_total[slot] = int_total_value
    output_I_0[slot] = i0_value
    output_qh[slot] = qh_value
    output_qe[slot] = qe_value
    output_ql[slot] = ql_value
    output_heat[slot] = heat_value
    output_tx[slot] = tx_surface_value
    output_ty[slot] = ty_surface_value
    output_sst[slot] = sst_value
    output_sst_obs[slot] = 0.0 if sst_obs_value != sst_obs_value else sst_obs_value
    output_sss[slot] = 0.0 if sss_value != sss_value else sss_value
    output_mld_surf[slot] = mld_surf_value
    output_u_taub[slot] = u_taub[0]
    output_taub[slot] = taub[0]
    output_mld_bott[slot] = mld_bott_value
    output_us0[slot] = us0_value
    output_vs0[slot] = vs0_value
    output_ds[slot] = ds_value
    output_Ekin[slot] = ekin_value
    output_Epot[slot] = epot_value
    output_Eturb[slot] = eturb_value
    if step != 0:
        ekin = 0.0
        epot = 0.0
        eturb = 0.0
        zloc = 0.0
        for i in range(1, nlev + 1):
            zloc -= 0.5 * h[i]
            ekin += 0.5 * h[i] * (u[i] * u[i] + v[i] * v[i])
            eturb += h[i] * (tke[i] + tke[i - 1])
            epot += h[i] * buoy[i] * zloc
            zloc -= 0.5 * h[i]
        output_Ekin[slot] = ekin * rho0_value
        output_Epot[slot] = epot * rho0_value
        output_Eturb[slot] = eturb * rho0_value
    for k in range(nlev + 1):
        output_rho_p[slot, k] = rho_p[k]
        output_rho[slot, k] = rho[k]
        output_u[slot, k] = u[k]
        output_v[slot, k] = v[k]
        output_T[slot, k] = T[k]
        output_S[slot, k] = S[k]
        output_Tp[slot, k] = Tp[k]
        output_Ti[slot, k] = Ti[k]
        output_Sp[slot, k] = Sp[k]
        output_Tobs[slot, k] = Tobs[k]
        output_Sobs[slot, k] = Sobs[k]
        output_u_obs[slot, k] = u_obs[k]
        output_v_obs[slot, k] = v_obs[k]
        output_idpdx[slot, k] = idpdx[k]
        output_idpdy[slot, k] = idpdy[k]
        output_tke[slot, k] = tke[k]
        output_eps[slot, k] = eps[k]
        output_num[slot, k] = num[k]
        output_nuh[slot, k] = nuh[k]
        output_h[slot, k] = h[k]
        output_xP[slot, k] = xP[k]
        output_fric[slot, k] = fric[k]
        output_drag[slot, k] = drag[k]
        output_avh[slot, k] = avh[k]
        output_bioshade[slot, k] = bioshade[k]
        output_ga[slot, k] = ga[k]
        output_uu[slot, k] = uu[k]
        output_vv[slot, k] = vv[k]
        output_ww[slot, k] = ww[k]
        output_NN[slot, k] = NN[k]
        output_NNT[slot, k] = NNT[k]
        output_NNS[slot, k] = NNS[k]
        output_buoy[slot, k] = buoy[k]
        output_SS[slot, k] = SS[k]
        output_P[slot, k] = P[k]
        output_B[slot, k] = B[k]
        output_Pb[slot, k] = Pb[k]
        output_kb[slot, k] = kb[k]
        output_epsb[slot, k] = epsb[k]
        output_L[slot, k] = L[k]
        output_PSTK[slot, k] = PSTK[k]
        output_cmue1[slot, k] = cmue1[k]
        output_cmue2[slot, k] = cmue2[k]
        output_as[slot, k] = as_[k]
        output_an[slot, k] = an[k]
        output_at[slot, k] = at[k]
        output_gamu[slot, k] = 0.0
        output_gamv[slot, k] = 0.0
        output_gamh[slot, k] = gamh[k]
        output_gams[slot, k] = gams[k]
        output_Rig[slot, k] = 0.0 if step == 0 else NN[k] / (SS[k] + 1.0e-10)
        output_gamb[slot, k] = 0.0
        output_gam[slot, k] = 0.0
        output_r[slot, k] = 0.0
        if step == 0:
            output_taux[slot, k] = 0.0
            output_tauy[slot, k] = 0.0
        elif k == 0:
            speed = math.sqrt(u[1] * u[1] + v[1] * v[1])
            output_taux[slot, k] = -drag[1] * u[1] * speed
            output_tauy[slot, k] = -drag[1] * v[1] * speed
        elif k == nlev:
            output_taux[slot, k] = -tx_surface_value
            output_tauy[slot, k] = -ty_surface_value
        else:
            spacing = 0.5 * (h[k + 1] + h[k])
            output_taux[slot, k] = (
                -num[k] * (u[k + 1] - u[k]) / spacing - nucl[k] * dusdz[k]
            )
            output_tauy[slot, k] = (
                -num[k] * (v[k + 1] - v[k]) / spacing - nucl[k] * dvsdz[k]
            )
        output_rad[slot, k] = rad[k]
        output_us[slot, k] = us[k]
        output_vs[slot, k] = vs[k]
        output_dusdz[slot, k] = dusdz[k]
        output_dvsdz[slot, k] = dvsdz[k]
        output_nus[slot, k] = nus[k]
        output_nucl[slot, k] = nucl[k]
        output_z[slot, k] = z[k]
        output_zi[slot, k] = zi[k]


@numba.njit(cache=True, fastmath=False)
def time_loop_compiled(
    nlev: int,
    nt: int,
    dt: float,
    cnpar: float,
    output_every: int,
    output_enabled: int,
    force_final_output: int,
    gravity: float,
    rho0: float,
    density_method: int,
    rhob: float,
    alpha0: float,
    beta0: float,
    T0: float,
    S0: float,
    avmolu: float,
    avmolT: float,
    avmolS: float,
    cp: float,
    cori: float,
    latitude: float,
    longitude: float,
    depth: float,
    h0b: float,
    z0s_min: float,
    charnock: int,
    charnock_val: float,
    calc_bottom_stress: int,
    max_it_z0b: int,
    plume_active: int,
    int_press_type: int,
    plume_type: int,
    plume_slope_x: float,
    plume_slope_y: float,
    seagrass_active: int,
    seagrass_alpha: float,
    seagrass_grassind: int,
    seagrass_grassn: int,
    w_adv_active: int,
    w_adv_discr: int,
    s_adv: int,
    t_adv: int,
    sprof_input_active: int,
    tprof_input_active: int,
    uprof_input_active: int,
    vprof_input_active: int,
    zeta_input_active: int,
    grid_method: int,
    ext_press_mode: int,
    vel_relax_ramp: float,
    airsea_fluxes_method: int,
    airsea_hum_method: int,
    airsea_shortwave_method: int,
    airsea_shortwave_type: int,
    airsea_longwave_method: int,
    airsea_longwave_type: int,
    airsea_albedo_method: int,
    airsea_ssuv_method: int,
    airsea_sst_obs_method: int,
    airsea_shortwave_scale_factor: float,
    airsea_heat_scale_factor: float,
    airsea_const_albedo: float,
    light_A: float,
    light_g1: float,
    light_g2: float,
    len_scale_method: int,
    scnd_method: int,
    tke_method: int,
    turb_method: int,
    stab_method: int,
    prandtl0_fix: float,
    mld_method: int,
    mld_diff_k: float,
    mld_ri_crit: float,
    my_b1: float,
    my_sq: float,
    my_sl: float,
    my_e1: float,
    my_e2: float,
    my_e3: float,
    my_ex: float,
    my_e6: float,
    my_length: int,
    kappa: float,
    cm0: float,
    cmsf: float,
    cde: float,
    k_min: float,
    eps_min: float,
    kb_min: float,
    epsb_min: float,
    k_ubc: int,
    k_lbc: int,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    length_lim: int,
    sig_k: float,
    sig_w: float,
    sig_e: float,
    sig_e0: float,
    sig_peps: int,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    galp: float,
    cc1: float,
    ct1: float,
    ctt: float,
    a1: float,
    a2: float,
    a3: float,
    a5: float,
    at1: float,
    at2: float,
    at3: float,
    at5: float,
    cw1: float,
    cw2: float,
    cw3plus: float,
    cw3minus: float,
    cwx: float,
    cw4: float,
    ce1: float,
    ce2: float,
    ce3plus: float,
    ce3minus: float,
    cex: float,
    ce4: float,
    iw_model: int,
    iw_alpha: float,
    klimiw: float,
    rich_cr: float,
    numiw: float,
    nuhiw: float,
    numshear: float,
    h: np.ndarray,
    ho: np.ndarray,
    u: np.ndarray,
    uo: np.ndarray,
    v: np.ndarray,
    vo: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    Tp: np.ndarray,
    Ti: np.ndarray,
    Sp: np.ndarray,
    Tobs: np.ndarray,
    Sobs: np.ndarray,
    NN: np.ndarray,
    NNT: np.ndarray,
    NNS: np.ndarray,
    SS: np.ndarray,
    SSU: np.ndarray,
    SSV: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
    xP: np.ndarray,
    fric: np.ndarray,
    drag: np.ndarray,
    avh: np.ndarray,
    bioshade: np.ndarray,
    ga: np.ndarray,
    rad: np.ndarray,
    buoy: np.ndarray,
    alpha_density: np.ndarray,
    beta_density: np.ndarray,
    rho_p: np.ndarray,
    rho: np.ndarray,
    tke: np.ndarray,
    tkeo: np.ndarray,
    eps: np.ndarray,
    omega: np.ndarray,
    L: np.ndarray,
    kb: np.ndarray,
    epsb: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Pb: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nus: np.ndarray,
    nucl: np.ndarray,
    gamh: np.ndarray,
    gams: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    cmue3: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    av: np.ndarray,
    aw: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    ww: np.ndarray,
    sq_var: np.ndarray,
    sl_var: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    z0b: np.ndarray,
    z0s: np.ndarray,
    za: np.ndarray,
    u_taub: np.ndarray,
    u_taubo: np.ndarray,
    u_taus: np.ndarray,
    taub: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
    work_avh: np.ndarray,
    q_sour: np.ndarray,
    l_sour: np.ndarray,
    sig_eff: np.ndarray,
    adv_cu: np.ndarray,
    idpdx: np.ndarray,
    idpdy: np.ndarray,
    dusdz: np.ndarray,
    dvsdz: np.ndarray,
    relax_tau: np.ndarray,
    relax_tau_eff: np.ndarray,
    s_relax_tau: np.ndarray,
    t_relax_tau: np.ndarray,
    uprof: np.ndarray,
    vprof: np.ndarray,
    q2l: np.ndarray,
    seagrass_z: np.ndarray,
    seagrass_exc: np.ndarray,
    seagrass_vfric: np.ndarray,
    seagrass_xx: np.ndarray,
    seagrass_yy: np.ndarray,
    seagrass_xxP: np.ndarray,
    seagrass_excur: np.ndarray,
    seagrass_grassfric: np.ndarray,
    stokes_active: int,
    forcing_yearday: np.ndarray,
    forcing_secondsofday: np.ndarray,
    forcing_dpdx: np.ndarray,
    forcing_dpdy: np.ndarray,
    forcing_h_press: np.ndarray,
    forcing_tx: np.ndarray,
    forcing_ty: np.ndarray,
    forcing_heat: np.ndarray,
    forcing_swr: np.ndarray,
    forcing_airp: np.ndarray,
    forcing_airt: np.ndarray,
    forcing_hum: np.ndarray,
    forcing_cloud: np.ndarray,
    forcing_u10: np.ndarray,
    forcing_v10: np.ndarray,
    forcing_precip: np.ndarray,
    forcing_longwave: np.ndarray,
    forcing_sst_obs: np.ndarray,
    forcing_sss_obs: np.ndarray,
    forcing_Tobs: np.ndarray,
    forcing_Sobs: np.ndarray,
    forcing_uprof: np.ndarray,
    forcing_vprof: np.ndarray,
    forcing_dtdx: np.ndarray,
    forcing_dtdy: np.ndarray,
    forcing_dsdx: np.ndarray,
    forcing_dsdy: np.ndarray,
    forcing_w_adv: np.ndarray,
    forcing_w_height: np.ndarray,
    forcing_zeta: np.ndarray,
    forcing_us0: np.ndarray,
    forcing_vs0: np.ndarray,
    forcing_ds: np.ndarray,
    forcing_us: np.ndarray,
    forcing_vs: np.ndarray,
    forcing_dusdz: np.ndarray,
    forcing_dvsdz: np.ndarray,
    output_step: np.ndarray,
    output_time: np.ndarray,
    output_zeta: np.ndarray,
    output_u_taus: np.ndarray,
    output_u10: np.ndarray,
    output_v10: np.ndarray,
    output_airt: np.ndarray,
    output_airp: np.ndarray,
    output_hum: np.ndarray,
    output_es: np.ndarray,
    output_ea: np.ndarray,
    output_qs: np.ndarray,
    output_qa: np.ndarray,
    output_rhoa: np.ndarray,
    output_cloud: np.ndarray,
    output_albedo: np.ndarray,
    output_precip: np.ndarray,
    output_evap: np.ndarray,
    output_int_precip: np.ndarray,
    output_int_evap: np.ndarray,
    output_int_swr: np.ndarray,
    output_int_heat: np.ndarray,
    output_int_total: np.ndarray,
    output_I_0: np.ndarray,
    output_qh: np.ndarray,
    output_qe: np.ndarray,
    output_ql: np.ndarray,
    output_heat: np.ndarray,
    output_tx: np.ndarray,
    output_ty: np.ndarray,
    output_sst: np.ndarray,
    output_sst_obs: np.ndarray,
    output_sss: np.ndarray,
    output_mld_surf: np.ndarray,
    output_u_taub: np.ndarray,
    output_taub: np.ndarray,
    output_mld_bott: np.ndarray,
    output_us0: np.ndarray,
    output_vs0: np.ndarray,
    output_ds: np.ndarray,
    output_Ekin: np.ndarray,
    output_Epot: np.ndarray,
    output_Eturb: np.ndarray,
    output_rho_p: np.ndarray,
    output_rho: np.ndarray,
    output_u: np.ndarray,
    output_v: np.ndarray,
    output_T: np.ndarray,
    output_S: np.ndarray,
    output_Tp: np.ndarray,
    output_Ti: np.ndarray,
    output_Sp: np.ndarray,
    output_Tobs: np.ndarray,
    output_Sobs: np.ndarray,
    output_u_obs: np.ndarray,
    output_v_obs: np.ndarray,
    output_idpdx: np.ndarray,
    output_idpdy: np.ndarray,
    output_tke: np.ndarray,
    output_eps: np.ndarray,
    output_num: np.ndarray,
    output_nuh: np.ndarray,
    output_h: np.ndarray,
    output_xP: np.ndarray,
    output_fric: np.ndarray,
    output_drag: np.ndarray,
    output_avh: np.ndarray,
    output_bioshade: np.ndarray,
    output_ga: np.ndarray,
    output_uu: np.ndarray,
    output_vv: np.ndarray,
    output_ww: np.ndarray,
    output_NN: np.ndarray,
    output_NNT: np.ndarray,
    output_NNS: np.ndarray,
    output_buoy: np.ndarray,
    output_SS: np.ndarray,
    output_P: np.ndarray,
    output_B: np.ndarray,
    output_Pb: np.ndarray,
    output_kb: np.ndarray,
    output_epsb: np.ndarray,
    output_L: np.ndarray,
    output_PSTK: np.ndarray,
    output_cmue1: np.ndarray,
    output_cmue2: np.ndarray,
    output_as: np.ndarray,
    output_an: np.ndarray,
    output_at: np.ndarray,
    output_gamu: np.ndarray,
    output_gamv: np.ndarray,
    output_gamh: np.ndarray,
    output_gams: np.ndarray,
    output_Rig: np.ndarray,
    output_gamb: np.ndarray,
    output_gam: np.ndarray,
    output_r: np.ndarray,
    output_taux: np.ndarray,
    output_tauy: np.ndarray,
    output_rad: np.ndarray,
    output_us: np.ndarray,
    output_vs: np.ndarray,
    output_dusdz: np.ndarray,
    output_dvsdz: np.ndarray,
    output_nus: np.ndarray,
    output_nucl: np.ndarray,
    output_z: np.ndarray,
    output_zi: np.ndarray,
    out_slot_base: int,
    write_ic: int,
    init_int_precip: float,
    init_int_evap: float,
    init_int_swr: float,
    init_int_heat: float,
    init_int_total: float,
    hydro_store: int,
    hydro_T: np.ndarray,
    hydro_S: np.ndarray,
    hydro_rho: np.ndarray,
    hydro_h: np.ndarray,
    hydro_nuh: np.ndarray,
    hydro_rad: np.ndarray,
    hydro_taub: np.ndarray,
) -> int:
    """Run profile-forced cases through the compiled timestep loop."""

    if nlev < 1 or nt < 0 or dt <= 0.0 or output_every < 1:
        return -2

    for k in range(nlev + 1):
        idpdx[k] = 0.0
        idpdy[k] = 0.0

    if turb_method == 2:  # first_order — init stability functions before step-0 output
        for k in range(nlev + 1):
            cmue3[k] = 0.0
        if stab_method == 1:  # Constant
            for k in range(nlev + 1):
                cmue1[k] = cm0
                cmue2[k] = cm0 / prandtl0_fix
        elif stab_method == 2:  # Munk_Anderson
            _step_cmue_ma_single(nlev, cm0, prandtl0_fix, as_, an, cmue1, cmue2)
            cmue1[0] = cmue1[1]
            cmue1[nlev] = cmue1[nlev - 1]
            cmue2[0] = cmue2[1]
            cmue2[nlev] = cmue2[nlev - 1]
        else:  # Schumann_Gerz
            _step_cmue_sg_single(nlev, cm0, prandtl0_fix, as_, an, cmue1, cmue2)
            cmue1[0] = cmue1[1]
            cmue1[nlev] = cmue1[nlev - 1]
            cmue2[0] = cmue2[1]
            cmue2[nlev] = cmue2[nlev - 1]

    int_precip = init_int_precip
    int_evap = init_int_evap
    int_swr = init_int_swr
    int_heat = init_int_heat
    int_total = init_int_total

    out_index = 0
    if output_enabled != 0:
        if out_slot_base + out_index >= output_step.shape[0]:
            return -1
        initial_evap = 0.0
        initial_tx_surface = forcing_tx[0]
        initial_ty_surface = forcing_ty[0]
        initial_heat = forcing_heat[0]
        initial_shortwave = forcing_swr[0]
        initial_albedo = 0.0
        initial_es = 0.0
        initial_ea = 0.0
        initial_qs = 0.0
        initial_qa = 0.0
        initial_rhoa = 0.0
        initial_qh = 0.0
        initial_qe = 0.0
        initial_ql = 0.0
        initial_sst = _output_sst_value(
            0.0,
            forcing_sst_obs[0],
            airsea_sst_obs_method,
        )
        if airsea_fluxes_method == 1:
            (
                initial_evap,
                initial_tx_surface,
                initial_ty_surface,
                initial_heat,
                initial_shortwave,
                initial_albedo,
                initial_es,
                initial_ea,
                initial_qs,
                initial_qa,
                initial_rhoa,
                initial_qh,
                initial_qe,
                initial_ql,
            ) = _airsea_kondo_compiled(
                forcing_yearday[0],
                forcing_secondsofday[0],
                latitude,
                longitude,
                airsea_hum_method,
                airsea_shortwave_method,
                airsea_shortwave_type,
                airsea_longwave_method,
                airsea_longwave_type,
                airsea_albedo_method,
                airsea_shortwave_scale_factor,
                airsea_heat_scale_factor,
                airsea_const_albedo,
                0.0,
                forcing_airp[0],
                forcing_airt[0],
                forcing_hum[0],
                forcing_cloud[0],
                forcing_u10[0],
                forcing_v10[0],
                forcing_precip[0],
                forcing_longwave[0],
                forcing_swr[0],
            )
        elif airsea_fluxes_method == 2:
            (
                initial_evap,
                initial_tx_surface,
                initial_ty_surface,
                initial_heat,
                initial_shortwave,
                initial_albedo,
                initial_es,
                initial_ea,
                initial_qs,
                initial_qa,
                initial_rhoa,
                initial_qh,
                initial_qe,
                initial_ql,
            ) = _airsea_fairall_compiled(
                forcing_yearday[0],
                forcing_secondsofday[0],
                latitude,
                longitude,
                airsea_hum_method,
                airsea_shortwave_method,
                airsea_shortwave_type,
                airsea_longwave_method,
                airsea_longwave_type,
                airsea_albedo_method,
                airsea_shortwave_scale_factor,
                airsea_heat_scale_factor,
                airsea_const_albedo,
                0.0,
                forcing_airp[0],
                forcing_airt[0],
                forcing_hum[0],
                forcing_cloud[0],
                forcing_u10[0],
                forcing_v10[0],
                forcing_precip[0],
                forcing_longwave[0],
                forcing_swr[0],
            )
        if write_ic != 0:
            _write_output_slot(
                out_slot_base + out_index,
                0,
                0.0,
                nlev,
                rho0,
                forcing_zeta[0],
                forcing_u10[0],
                forcing_v10[0],
                forcing_airt[0],
                forcing_airp[0],
                forcing_hum[0],
                initial_es,
                initial_ea,
                initial_qs,
                initial_qa,
                initial_rhoa,
                forcing_cloud[0],
                initial_albedo,
                forcing_precip[0],
                initial_evap,
                int_precip,
                int_evap,
                int_swr,
                int_heat,
                int_total,
                initial_shortwave,
                initial_qh,
                initial_qe,
                initial_ql,
                initial_heat,
                initial_tx_surface,
                initial_ty_surface,
                initial_sst,
                forcing_sst_obs[0],
                forcing_sss_obs[0],
                0.0,
                0.0,
                forcing_us0[0],
                forcing_vs0[0],
                forcing_ds[0],
                0.0,
                0.0,
                0.0,
                u_taus,
                u_taub,
                taub,
                rho_p,
                rho,
                u,
                v,
                T,
                S,
                Tp,
                Ti,
                Sp,
                Tobs,
                Sobs,
                uprof,
                vprof,
                idpdx,
                idpdy,
                tke,
                eps,
                num,
                nuh,
                h,
                xP,
                fric,
                drag,
                avh,
                bioshade,
                ga,
                uu,
                vv,
                ww,
                NN,
                NNT,
                NNS,
                buoy,
                SS,
                P,
                B,
                Pb,
                kb,
                epsb,
                L,
                PSTK,
                cmue1,
                cmue2,
                as_,
                an,
                at,
                gamh,
                gams,
                rad,
                forcing_us[0],
                forcing_vs[0],
                dusdz,
                dvsdz,
                nus,
                nucl,
                z,
                zi,
                output_step,
                output_time,
                output_zeta,
                output_u_taus,
                output_u10,
                output_v10,
                output_airt,
                output_airp,
                output_hum,
                output_es,
                output_ea,
                output_qs,
                output_qa,
                output_rhoa,
                output_cloud,
                output_albedo,
                output_precip,
                output_evap,
                output_int_precip,
                output_int_evap,
                output_int_swr,
                output_int_heat,
                output_int_total,
                output_I_0,
                output_qh,
                output_qe,
                output_ql,
                output_heat,
                output_tx,
                output_ty,
                output_sst,
                output_sst_obs,
                output_sss,
                output_mld_surf,
                output_u_taub,
                output_taub,
                output_mld_bott,
                output_us0,
                output_vs0,
                output_ds,
                output_Ekin,
                output_Epot,
                output_Eturb,
                output_rho_p,
                output_rho,
                output_u,
                output_v,
                output_T,
                output_S,
                output_Tp,
                output_Ti,
                output_Sp,
                output_Tobs,
                output_Sobs,
                output_u_obs,
                output_v_obs,
                output_idpdx,
                output_idpdy,
                output_tke,
                output_eps,
                output_num,
                output_nuh,
                output_h,
                output_xP,
                output_fric,
                output_drag,
                output_avh,
                output_bioshade,
                output_ga,
                output_uu,
                output_vv,
                output_ww,
                output_NN,
                output_NNT,
                output_NNS,
                output_buoy,
                output_SS,
                output_P,
                output_B,
                output_Pb,
                output_kb,
                output_epsb,
                output_L,
                output_PSTK,
                output_cmue1,
                output_cmue2,
                output_as,
                output_an,
                output_at,
                output_gamu,
                output_gamv,
                output_gamh,
                output_gams,
                output_Rig,
                output_gamb,
                output_gam,
                output_r,
                output_taux,
                output_tauy,
                output_rad,
                output_us,
                output_vs,
                output_dusdz,
                output_dvsdz,
                output_nus,
                output_nucl,
                output_z,
                output_zi,
            )
        out_index += 1

    if hydro_store != 0:
        for k in range(nlev + 1):
            hydro_T[0, k] = T[k]
            hydro_S[0, k] = S[k]
            hydro_rho[0, k] = rho[k]
            hydro_h[0, k] = h[k]
            hydro_nuh[0, k] = nuh[k]
            hydro_rad[0, k] = rad[k]
        hydro_taub[0] = taub[0]

    cosomega = math.cos(cori * dt)
    sinomega = math.sin(cori * dt)
    w_adv_for_equations = 1 if w_adv_active != 0 else 0

    for step in range(1, nt + 1):
        if uprof_input_active != 0:
            for k in range(nlev + 1):
                uprof[k] = forcing_uprof[step, k]
        if vprof_input_active != 0:
            for k in range(nlev + 1):
                vprof[k] = forcing_vprof[step, k]

        if zeta_input_active != 0:
            step_updategrid_single(
                nlev,
                depth,
                forcing_zeta[step],
                grid_method,
                ga,
                h,
                ho,
                z,
                zi,
            )

        if stokes_active != 0:
            for k in range(nlev + 1):
                dusdz[k] = forcing_dusdz[step, k]
                dvsdz[k] = forcing_dvsdz[step, k]

        if w_adv_active != 0:
            _step_wequation_single(
                nlev,
                w_adv_active,
                forcing_w_adv[step],
                forcing_w_height[step],
                zi,
                w,
            )

        if cori != 0.0:
            step_coriolis_single(nlev, cosomega, sinomega, u, v, uprof, vprof)

        evap = 0.0
        albedo = 0.0
        es = 0.0
        ea = 0.0
        qs = 0.0
        qa = 0.0
        rhoa = 0.0
        qh = 0.0
        qe = 0.0
        ql = 0.0
        shortwave = forcing_swr[step]
        sst_value = _output_sst_value(
            0.0,
            forcing_sst_obs[step],
            airsea_sst_obs_method,
        )
        sss_value = forcing_sss_obs[step]
        if airsea_fluxes_method == 1:
            wind_u = forcing_u10[step]
            wind_v = forcing_v10[step]
            if airsea_ssuv_method != 0:
                wind_u -= u[nlev]
                wind_v -= v[nlev]
            model_sst = _gsw_t_from_ct_surface_compiled(S[nlev], T[nlev])
            sst_value = _output_sst_value(
                model_sst,
                forcing_sst_obs[step],
                airsea_sst_obs_method,
            )
            (
                evap,
                tx_surface,
                ty_surface,
                heat,
                shortwave,
                albedo,
                es,
                ea,
                qs,
                qa,
                rhoa,
                qh,
                qe,
                ql,
            ) = _airsea_kondo_compiled(
                forcing_yearday[step],
                forcing_secondsofday[step],
                latitude,
                longitude,
                airsea_hum_method,
                airsea_shortwave_method,
                airsea_shortwave_type,
                airsea_longwave_method,
                airsea_longwave_type,
                airsea_albedo_method,
                airsea_shortwave_scale_factor,
                airsea_heat_scale_factor,
                airsea_const_albedo,
                model_sst,
                forcing_airp[step],
                forcing_airt[step],
                forcing_hum[step],
                forcing_cloud[step],
                wind_u,
                wind_v,
                forcing_precip[step],
                forcing_longwave[step],
                forcing_swr[step],
            )
            tx_value = tx_surface / rho0
            ty_value = ty_surface / rho0
            swf = forcing_precip[step] + evap
            shf = -heat
            current_i0 = _net_shortwave_compiled(
                airsea_shortwave_method,
                airsea_shortwave_type,
                shortwave,
                albedo,
            )
        elif airsea_fluxes_method == 2:
            wind_u = forcing_u10[step]
            wind_v = forcing_v10[step]
            if airsea_ssuv_method != 0:
                wind_u -= u[nlev]
                wind_v -= v[nlev]
            model_sst = _gsw_t_from_ct_surface_compiled(S[nlev], T[nlev])
            sst_value = _output_sst_value(
                model_sst,
                forcing_sst_obs[step],
                airsea_sst_obs_method,
            )
            (
                evap,
                tx_surface,
                ty_surface,
                heat,
                shortwave,
                albedo,
                es,
                ea,
                qs,
                qa,
                rhoa,
                qh,
                qe,
                ql,
            ) = _airsea_fairall_compiled(
                forcing_yearday[step],
                forcing_secondsofday[step],
                latitude,
                longitude,
                airsea_hum_method,
                airsea_shortwave_method,
                airsea_shortwave_type,
                airsea_longwave_method,
                airsea_longwave_type,
                airsea_albedo_method,
                airsea_shortwave_scale_factor,
                airsea_heat_scale_factor,
                airsea_const_albedo,
                model_sst,
                forcing_airp[step],
                forcing_airt[step],
                forcing_hum[step],
                forcing_cloud[step],
                wind_u,
                wind_v,
                forcing_precip[step],
                forcing_longwave[step],
                forcing_swr[step],
            )
            tx_value = tx_surface / rho0
            ty_value = ty_surface / rho0
            swf = forcing_precip[step] + evap
            shf = -heat
            current_i0 = _net_shortwave_compiled(
                airsea_shortwave_method,
                airsea_shortwave_type,
                shortwave,
                albedo,
            )
        else:
            tx_surface = forcing_tx[step]
            ty_surface = forcing_ty[step]
            tx_value = forcing_tx[step] / rho0
            ty_value = forcing_ty[step] / rho0
            heat = forcing_heat[step]
            current_i0 = forcing_swr[step]
            swf = forcing_precip[step]
            shf = -heat

        tx[0] = tx_value
        ty[0] = ty_value
        ssf = S[nlev] * swf
        int_precip += forcing_precip[step] * dt
        int_evap += evap * dt
        int_swr += current_i0 * dt
        int_heat += heat * dt
        int_total += (heat + current_i0) * dt

        relax_factor = 1.0
        elapsed = step * dt
        if vel_relax_ramp < _LONG and elapsed < vel_relax_ramp:
            relax_factor = vel_relax_ramp / (vel_relax_ramp - elapsed)
        for k in range(nlev + 1):
            relax_tau_eff[k] = relax_tau[k] * relax_factor

        if seagrass_active != 0:
            _step_seagrass_single(
                nlev,
                dt,
                seagrass_alpha,
                seagrass_grassind,
                seagrass_grassn,
                seagrass_z,
                seagrass_exc,
                seagrass_vfric,
                u,
                v,
                h,
                drag,
                xP,
                seagrass_xx,
                seagrass_yy,
                seagrass_xxP,
                seagrass_excur,
                seagrass_grassfric,
                q_sour,
            )

        step_uequation_single(
            nlev,
            dt,
            cnpar,
            avmolu,
            gravity,
            ext_press_mode,
            w_adv_for_equations,
            w_adv_discr,
            seagrass_active,
            plume_active,
            tx_value,
            forcing_dpdx[step],
            u,
            uo,
            v,
            h,
            w,
            drag,
            num,
            nucl,
            dusdz,
            idpdx,
            uprof,
            relax_tau_eff,
            work_avh,
            q_sour,
            l_sour,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
            adv_cu,
        )
        step_vequation_single(
            nlev,
            dt,
            cnpar,
            avmolu,
            gravity,
            ext_press_mode,
            w_adv_for_equations,
            w_adv_discr,
            plume_active,
            ty_value,
            forcing_dpdy[step],
            v,
            vo,
            u,
            h,
            w,
            drag,
            num,
            nucl,
            dvsdz,
            idpdy,
            vprof,
            relax_tau_eff,
            work_avh,
            q_sour,
            l_sour,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
            adv_cu,
        )
        _external_pressure_single(
            nlev,
            ext_press_mode,
            forcing_dpdx[step],
            forcing_dpdy[step],
            forcing_h_press[step],
            h,
            u,
            v,
            q_sour,
        )
        if int_press_type == 1:
            _internal_pressure_gradients_single(
                nlev,
                gravity,
                rho0,
                density_method,
                rhob,
                alpha0,
                beta0,
                T0,
                S0,
                h,
                T,
                S,
                forcing_dsdx[step],
                forcing_dsdy[step],
                forcing_dtdx[step],
                forcing_dtdy[step],
                q_sour,
                l_sour,
                idpdx,
                idpdy,
            )
        elif int_press_type == 2:
            _internal_pressure_plume_single(
                nlev,
                plume_type,
                plume_slope_x,
                plume_slope_y,
                buoy,
                idpdx,
                idpdy,
            )
        for k in range(nlev + 1):
            avh[k] = work_avh[k]
        step_friction_single(
            nlev,
            kappa,
            avmolu,
            rho0,
            gravity,
            h0b,
            z0s_min,
            charnock,
            charnock_val,
            calc_bottom_stress,
            max_it_z0b,
            plume_active,
            1 if step == 1 else 0,
            h,
            u,
            v,
            drag,
            z0b,
            z0s,
            za,
            u_taub,
            u_taubo,
            u_taus,
            taub,
            tx,
            ty,
        )
        if sprof_input_active != 0:
            _copy_profile(nlev, forcing_Sobs[step], Sobs)
            step_salinity_single(
                nlev,
                dt,
                cnpar,
                avmolS,
                w_adv_for_equations,
                w_adv_discr,
                s_adv,
                S,
                h,
                w,
                u,
                v,
                nus,
                gams,
                Sobs,
                s_relax_tau,
                -ssf,
                forcing_dsdx[step],
                forcing_dsdy[step],
                work_avh,
                q_sour,
                l_sour,
                au,
                bu,
                cu,
                du,
                ru,
                qu,
                adv_cu,
            )
            for k in range(nlev + 1):
                avh[k] = work_avh[k]
        if tprof_input_active != 0:
            _copy_profile(nlev, forcing_Tobs[step], Tobs)
            step_temperature_single(
                nlev,
                dt,
                cnpar,
                avmolT,
                rho0,
                cp,
                light_A,
                light_g1,
                light_g2,
                w_adv_for_equations,
                w_adv_discr,
                t_adv,
                T,
                S,
                h,
                w,
                u,
                v,
                nuh,
                gamh,
                bioshade,
                rad,
                Tobs,
                t_relax_tau,
                current_i0,
                -shf / (rho0 * cp),
                forcing_dtdx[step],
                forcing_dtdy[step],
                work_avh,
                q_sour,
                l_sour,
                au,
                bu,
                cu,
                du,
                ru,
                qu,
                adv_cu,
            )
            for k in range(nlev + 1):
                avh[k] = work_avh[k]

        _copy_profile(nlev, forcing_Sobs[step], Sobs)
        _copy_profile(nlev, forcing_Tobs[step], Tobs)

        step_shear_single(
            nlev,
            cnpar,
            h,
            u,
            v,
            uo,
            vo,
            dusdz,
            dvsdz,
            SS,
            SSU,
            SSV,
            SSCSTK,
            SSSTK,
        )
        _update_density_single(
            nlev,
            density_method,
            gravity,
            rho0,
            alpha0,
            beta0,
            T0,
            S0,
            rhob,
            T,
            S,
            z,
            zi,
            alpha_density,
            beta_density,
            rho_p,
            rho,
            buoy,
        )
        _stratification_from_alpha_beta_single(
            nlev,
            gravity,
            h,
            T,
            S,
            alpha_density,
            beta_density,
            NN,
            NNT,
            NNS,
        )
        if turb_method == 3:  # second_order
            _run_second_order_turbulence_single(
                nlev,
                dt,
                depth,
                tke_method,
                len_scale_method,
                scnd_method,
                iw_model,
                iw_alpha,
                sig_k,
                sig_w,
                sig_e,
                sig_e0,
                sig_peps,
                k_min,
                eps_min,
                kb_min,
                epsb_min,
                k_ubc,
                k_lbc,
                psi_ubc,
                psi_lbc,
                ubc_type,
                lbc_type,
                cm0,
                cmsf,
                cde,
                kappa,
                cw,
                gen_alpha,
                gen_l,
                galp,
                length_lim,
                my_b1,
                my_sq,
                my_sl,
                my_e1,
                my_e2,
                my_e3,
                my_ex,
                my_e6,
                my_length,
                cc1,
                ct1,
                ctt,
                a1,
                a2,
                a3,
                a5,
                at1,
                at2,
                at3,
                at5,
                cw1,
                cw2,
                cw3plus,
                cw3minus,
                cwx,
                cw4,
                ce1,
                ce2,
                ce3plus,
                ce3minus,
                cex,
                ce4,
                klimiw,
                rich_cr,
                numiw,
                nuhiw,
                numshear,
                h,
                NN,
                SS,
                SSU,
                SSV,
                SSCSTK,
                SSSTK,
                xP,
                tke,
                tkeo,
                eps,
                omega,
                L,
                kb,
                epsb,
                P,
                B,
                Pb,
                Px,
                PSTK,
                num,
                nuh,
                nus,
                nucl,
                cmue1,
                cmue2,
                cmue3,
                as_,
                an,
                at,
                av,
                aw,
                uu,
                vv,
                ww,
                sq_var,
                sl_var,
                u_taus[0],
                u_taub[0],
                z0s[0],
                z0b[0],
                work_avh,
                sig_eff,
                q_sour,
                l_sour,
                q2l,
                au,
                bu,
                cu,
                du,
                ru,
                qu,
            )
        else:  # first_order (turb_method == 2)
            step_turbulence_first_order_single(
                nlev,
                dt,
                depth,
                stab_method,
                len_scale_method,
                iw_model,
                iw_alpha,
                sig_k,
                sig_w,
                sig_e,
                sig_e0,
                sig_peps,
                k_min,
                eps_min,
                k_ubc,
                k_lbc,
                psi_ubc,
                psi_lbc,
                ubc_type,
                lbc_type,
                cm0,
                cmsf,
                cde,
                kappa,
                cw,
                gen_alpha,
                gen_l,
                galp,
                length_lim,
                prandtl0_fix,
                cc1,
                ct1,
                a2,
                a3,
                a5,
                cw1,
                cw2,
                cw3plus,
                cw3minus,
                cwx,
                cw4,
                ce1,
                ce2,
                ce3plus,
                ce3minus,
                cex,
                ce4,
                klimiw,
                rich_cr,
                numiw,
                nuhiw,
                numshear,
                h,
                NN,
                SS,
                SSU,
                SSV,
                xP,
                SSCSTK,
                SSSTK,
                tke,
                tkeo,
                eps,
                omega,
                L,
                P,
                B,
                Pb,
                Px,
                PSTK,
                num,
                nuh,
                nus,
                nucl,
                cmue1,
                cmue2,
                cmue3,
                as_,
                an,
                at,
                av,
                aw,
                uu,
                vv,
                ww,
                u_taus[0],
                u_taub[0],
                z0s[0],
                z0b[0],
                work_avh,
                sig_eff,
                q_sour,
                l_sour,
                au,
                bu,
                cu,
                du,
                ru,
                qu,
            )

        if hydro_store != 0:
            for k in range(nlev + 1):
                hydro_T[step, k] = T[k]
                hydro_S[step, k] = S[k]
                hydro_rho[step, k] = rho[k]
                hydro_h[step, k] = h[k]
                hydro_nuh[step, k] = nuh[k]
                hydro_rad[step, k] = rad[k]
            hydro_taub[step] = taub[0]

        if output_enabled != 0 and (
            step % output_every == 0 or (force_final_output != 0 and step == nt)
        ):
            if out_slot_base + out_index >= output_step.shape[0]:
                return -1
            mld_surf_value, mld_bott_value = _compute_mld_single(
                nlev,
                mld_method,
                mld_diff_k,
                mld_ri_crit,
                turb_method,
                h,
                NN,
                SS,
                tke,
            )
            _write_output_slot(
                out_slot_base + out_index,
                step,
                step * dt,
                nlev,
                rho0,
                forcing_zeta[step],
                forcing_u10[step],
                forcing_v10[step],
                forcing_airt[step],
                forcing_airp[step],
                forcing_hum[step],
                es,
                ea,
                qs,
                qa,
                rhoa,
                forcing_cloud[step],
                albedo,
                forcing_precip[step],
                evap,
                int_precip,
                int_evap,
                int_swr,
                int_heat,
                int_total,
                current_i0,
                qh,
                qe,
                ql,
                heat,
                tx_value,
                ty_value,
                sst_value,
                forcing_sst_obs[step],
                sss_value,
                mld_surf_value,
                mld_bott_value,
                forcing_us0[step],
                forcing_vs0[step],
                forcing_ds[step],
                0.0,
                0.0,
                0.0,
                u_taus,
                u_taub,
                taub,
                rho_p,
                rho,
                u,
                v,
                T,
                S,
                Tp,
                Ti,
                Sp,
                Tobs,
                Sobs,
                uprof,
                vprof,
                idpdx,
                idpdy,
                tke,
                eps,
                num,
                nuh,
                h,
                xP,
                fric,
                drag,
                avh,
                bioshade,
                ga,
                uu,
                vv,
                ww,
                NN,
                NNT,
                NNS,
                buoy,
                SS,
                P,
                B,
                Pb,
                kb,
                epsb,
                L,
                PSTK,
                cmue1,
                cmue2,
                as_,
                an,
                at,
                gamh,
                gams,
                rad,
                forcing_us[step],
                forcing_vs[step],
                dusdz,
                dvsdz,
                nus,
                nucl,
                z,
                zi,
                output_step,
                output_time,
                output_zeta,
                output_u_taus,
                output_u10,
                output_v10,
                output_airt,
                output_airp,
                output_hum,
                output_es,
                output_ea,
                output_qs,
                output_qa,
                output_rhoa,
                output_cloud,
                output_albedo,
                output_precip,
                output_evap,
                output_int_precip,
                output_int_evap,
                output_int_swr,
                output_int_heat,
                output_int_total,
                output_I_0,
                output_qh,
                output_qe,
                output_ql,
                output_heat,
                output_tx,
                output_ty,
                output_sst,
                output_sst_obs,
                output_sss,
                output_mld_surf,
                output_u_taub,
                output_taub,
                output_mld_bott,
                output_us0,
                output_vs0,
                output_ds,
                output_Ekin,
                output_Epot,
                output_Eturb,
                output_rho_p,
                output_rho,
                output_u,
                output_v,
                output_T,
                output_S,
                output_Tp,
                output_Ti,
                output_Sp,
                output_Tobs,
                output_Sobs,
                output_u_obs,
                output_v_obs,
                output_idpdx,
                output_idpdy,
                output_tke,
                output_eps,
                output_num,
                output_nuh,
                output_h,
                output_xP,
                output_fric,
                output_drag,
                output_avh,
                output_bioshade,
                output_ga,
                output_uu,
                output_vv,
                output_ww,
                output_NN,
                output_NNT,
                output_NNS,
                output_buoy,
                output_SS,
                output_P,
                output_B,
                output_Pb,
                output_kb,
                output_epsb,
                output_L,
                output_PSTK,
                output_cmue1,
                output_cmue2,
                output_as,
                output_an,
                output_at,
                output_gamu,
                output_gamv,
                output_gamh,
                output_gams,
                output_Rig,
                output_gamb,
                output_gam,
                output_r,
                output_taux,
                output_tauy,
                output_rad,
                output_us,
                output_vs,
                output_dusdz,
                output_dvsdz,
                output_nus,
                output_nucl,
                output_z,
                output_zi,
            )
            out_index += 1

    return out_index


def _simple_freezing_temperature(salinity: float) -> float:
    """Return GOTM's simple ice freezing temperature in degrees Celsius."""

    return -0.0575 * salinity


def _populate_simple_ice_reference_scalars(
    params: RuntimeParams,
    output: RuntimeOutput,
    written: int,
) -> None:
    tf = output.reference_scalars.get("Tf")
    if tf is None or written <= 0:
        return

    top = params.nlev
    count = min(written, output.nout)
    for slot in range(count):
        salinity = (
            output.Sp[slot, top] if output.Sp[slot, top] != 0.0 else output.S[slot, top]
        )
        tf[slot] = _simple_freezing_temperature(float(salinity))


def run_compiled_time_loop(
    params: RuntimeParams,
    state: RuntimeState,
    work: RuntimeWork,
    forcing: RuntimeForcing,
    output: RuntimeOutput,
    step_offset: int = 0,
    out_slot_base: int = 0,
    write_ic: int = 1,
    init_int_precip: float = 0.0,
    init_int_evap: float = 0.0,
    init_int_swr: float = 0.0,
    init_int_heat: float = 0.0,
    init_int_total: float = 0.0,
    hydro_store: int = 0,
    hydro_T: np.ndarray | None = None,
    hydro_S: np.ndarray | None = None,
    hydro_rho: np.ndarray | None = None,
    hydro_h: np.ndarray | None = None,
    hydro_nuh: np.ndarray | None = None,
    hydro_rad: np.ndarray | None = None,
    hydro_taub: np.ndarray | None = None,
) -> int:
    """Validate runtime containers and cross into the compiled unified loop.

    *step_offset* is the global step index where this call starts — used to
    slice forcing arrays when running one chunk of a larger simulation.
    *out_slot_base* is the output-array slot offset for this call.
    Both default to 0 for full-run (non-chunked) execution.
    """

    state.validate()
    work.validate()
    forcing.validate()
    output.validate(params.nlev)

    nlev = params.nlev
    _dummy = np.zeros((1, nlev + 1), dtype=np.float64)
    # Build contiguous forcing slices for this chunk (no-op views when step_offset=0
    # and params.nt == forcing.nt, but always produces the correct window).
    _s = step_offset
    _e = step_offset + params.nt + 1
    _f_yearday = np.ascontiguousarray(forcing.yearday[_s:_e])
    _f_secondsofday = np.ascontiguousarray(forcing.secondsofday[_s:_e])
    _f_dpdx = np.ascontiguousarray(forcing.dpdx[_s:_e])
    _f_dpdy = np.ascontiguousarray(forcing.dpdy[_s:_e])
    _f_h_press = np.ascontiguousarray(forcing.h_press[_s:_e])
    _f_tx = np.ascontiguousarray(forcing.tx[_s:_e])
    _f_ty = np.ascontiguousarray(forcing.ty[_s:_e])
    _f_heat = np.ascontiguousarray(forcing.heat[_s:_e])
    _f_swr = np.ascontiguousarray(forcing.swr[_s:_e])
    _f_airp = np.ascontiguousarray(forcing.airp[_s:_e])
    _f_airt = np.ascontiguousarray(forcing.airt[_s:_e])
    _f_hum = np.ascontiguousarray(forcing.hum[_s:_e])
    _f_cloud = np.ascontiguousarray(forcing.cloud[_s:_e])
    _f_u10 = np.ascontiguousarray(forcing.u10[_s:_e])
    _f_v10 = np.ascontiguousarray(forcing.v10[_s:_e])
    _f_precip = np.ascontiguousarray(forcing.precip[_s:_e])
    _f_longwave = np.ascontiguousarray(forcing.longwave[_s:_e])
    _f_sst_obs = np.ascontiguousarray(forcing.sst_obs[_s:_e])
    _f_sss_obs = np.ascontiguousarray(forcing.sss_obs[_s:_e])
    _f_w_adv = np.ascontiguousarray(forcing.w_adv[_s:_e])
    _f_w_height = np.ascontiguousarray(forcing.w_height[_s:_e])
    _f_zeta = np.ascontiguousarray(forcing.zeta[_s:_e])
    _f_us0 = np.ascontiguousarray(forcing.us0[_s:_e])
    _f_vs0 = np.ascontiguousarray(forcing.vs0[_s:_e])
    _f_ds = np.ascontiguousarray(forcing.ds[_s:_e])
    _f_Tobs = np.ascontiguousarray(forcing.Tobs[_s:_e, :])
    _f_Sobs = np.ascontiguousarray(forcing.Sobs[_s:_e, :])
    _f_uprof = np.ascontiguousarray(forcing.uprof[_s:_e, :])
    _f_vprof = np.ascontiguousarray(forcing.vprof[_s:_e, :])
    _f_dtdx = np.ascontiguousarray(forcing.dtdx[_s:_e, :])
    _f_dtdy = np.ascontiguousarray(forcing.dtdy[_s:_e, :])
    _f_dsdx = np.ascontiguousarray(forcing.dsdx[_s:_e, :])
    _f_dsdy = np.ascontiguousarray(forcing.dsdy[_s:_e, :])
    _f_us = np.ascontiguousarray(forcing.us[_s:_e, :])
    _f_vs = np.ascontiguousarray(forcing.vs[_s:_e, :])
    _f_dusdz = np.ascontiguousarray(forcing.dusdz[_s:_e, :])
    _f_dvsdz = np.ascontiguousarray(forcing.dvsdz[_s:_e, :])
    _hydro_store = hydro_store if hydro_store != 0 else 0
    _hydro_T = hydro_T if hydro_T is not None else _dummy
    _hydro_S = hydro_S if hydro_S is not None else _dummy
    _hydro_rho = hydro_rho if hydro_rho is not None else _dummy
    _hydro_h = hydro_h if hydro_h is not None else _dummy
    _hydro_nuh = hydro_nuh if hydro_nuh is not None else _dummy
    _hydro_rad = hydro_rad if hydro_rad is not None else _dummy
    _dummy_scalar = np.zeros(1, dtype=np.float64)
    _hydro_taub = hydro_taub if hydro_taub is not None else _dummy_scalar
    del nlev

    written = int(
        time_loop_compiled(
            params.nlev,
            params.nt,
            params.dt,
            params.cnpar,
            output.output_every,
            1 if output.enabled else 0,
            1 if output.force_final else 0,
            params.gravity,
            params.rho0,
            params.density_method,
            params.rhob,
            params.alpha0,
            params.beta0,
            params.T0,
            params.S0,
            params.avmolu,
            params.avmolT,
            params.avmolS,
            params.cp,
            params.cori,
            params.latitude,
            params.longitude,
            params.depth,
            params.h0b,
            params.z0s_min,
            params.charnock,
            params.charnock_val,
            params.calc_bottom_stress,
            params.max_it_z0b,
            params.plume_active,
            params.int_press_type,
            params.plume_type,
            params.plume_slope_x,
            params.plume_slope_y,
            params.seagrass_active,
            params.seagrass_alpha,
            params.seagrass_grassind,
            params.seagrass_grassn,
            params.w_adv_active,
            params.w_adv_discr,
            params.s_adv,
            params.t_adv,
            params.sprof_input_active,
            params.tprof_input_active,
            params.uprof_input_active,
            params.vprof_input_active,
            params.zeta_input_active,
            params.grid_method,
            params.ext_press_mode,
            params.vel_relax_ramp,
            params.airsea_fluxes_method,
            params.airsea_hum_method,
            params.airsea_shortwave_method,
            params.airsea_shortwave_type,
            params.airsea_longwave_method,
            params.airsea_longwave_type,
            params.airsea_albedo_method,
            params.airsea_ssuv_method,
            params.airsea_sst_obs_method,
            params.airsea_shortwave_scale_factor,
            params.airsea_heat_scale_factor,
            params.airsea_const_albedo,
            params.light_A,
            params.light_g1,
            params.light_g2,
            params.len_scale_method,
            params.scnd_method,
            params.tke_method,
            params.turb_method,
            params.stab_method,
            params.prandtl0_fix,
            params.mld_method,
            params.mld_diff_k,
            params.mld_ri_crit,
            params.my_b1,
            params.my_sq,
            params.my_sl,
            params.my_e1,
            params.my_e2,
            params.my_e3,
            params.my_ex,
            params.my_e6,
            params.my_length,
            params.kappa,
            params.cm0,
            params.cmsf,
            params.cde,
            params.k_min,
            params.eps_min,
            params.kb_min,
            params.epsb_min,
            params.k_ubc,
            params.k_lbc,
            params.psi_ubc,
            params.psi_lbc,
            params.ubc_type,
            params.lbc_type,
            params.length_lim,
            params.sig_k,
            params.sig_w,
            params.sig_e,
            params.sig_e0,
            params.sig_peps,
            params.cw,
            params.gen_alpha,
            params.gen_l,
            params.galp,
            params.cc1,
            params.ct1,
            params.ctt,
            params.a1,
            params.a2,
            params.a3,
            params.a5,
            params.at1,
            params.at2,
            params.at3,
            params.at5,
            params.cw1,
            params.cw2,
            params.cw3plus,
            params.cw3minus,
            params.cwx,
            params.cw4,
            params.ce1,
            params.ce2,
            params.ce3plus,
            params.ce3minus,
            params.cex,
            params.ce4,
            params.iw_model,
            params.iw_alpha,
            params.klimiw,
            params.rich_cr,
            params.numiw,
            params.nuhiw,
            params.numshear,
            state.h,
            state.ho,
            state.u,
            state.uo,
            state.v,
            state.vo,
            state.w,
            state.T,
            state.S,
            state.Tp,
            state.Ti,
            state.Sp,
            state.Tobs,
            state.Sobs,
            state.NN,
            state.NNT,
            state.NNS,
            state.SS,
            state.SSU,
            state.SSV,
            state.SSCSTK,
            state.SSSTK,
            state.xP,
            state.fric,
            state.drag,
            state.avh,
            state.bioshade,
            state.ga,
            state.rad,
            state.buoy,
            state.alpha,
            state.beta,
            state.rho_p,
            state.rho,
            state.tke,
            state.tkeo,
            state.eps,
            state.omega,
            state.L,
            state.kb,
            state.epsb,
            state.P,
            state.B,
            state.Pb,
            state.Px,
            state.PSTK,
            state.num,
            state.nuh,
            state.nus,
            state.nucl,
            state.gamh,
            state.gams,
            state.cmue1,
            state.cmue2,
            state.cmue3,
            state.as_,
            state.an,
            state.at,
            state.av,
            state.aw,
            state.uu,
            state.vv,
            state.ww,
            state.sq_var,
            state.sl_var,
            state.z,
            state.zi,
            state.z0b,
            state.z0s,
            state.za,
            state.u_taub,
            state.u_taubo,
            state.u_taus,
            state.taub,
            state.tx,
            state.ty,
            work.au,
            work.bu,
            work.cu,
            work.du,
            work.ru,
            work.qu,
            work.avh,
            work.q_sour,
            work.l_sour,
            work.sig_eff,
            work.adv_cu,
            work.idpdx,
            work.idpdy,
            work.dusdz,
            work.dvsdz,
            work.vel_relax_tau,
            work.vel_relax_tau_eff,
            work.s_relax_tau,
            work.t_relax_tau,
            work.uprof,
            work.vprof,
            work.q2l,
            work.seagrass_z,
            work.seagrass_exc,
            work.seagrass_vfric,
            work.seagrass_xx,
            work.seagrass_yy,
            work.seagrass_xxP,
            work.seagrass_excur,
            work.seagrass_grassfric,
            params.stokes_active,
            _f_yearday,
            _f_secondsofday,
            _f_dpdx,
            _f_dpdy,
            _f_h_press,
            _f_tx,
            _f_ty,
            _f_heat,
            _f_swr,
            _f_airp,
            _f_airt,
            _f_hum,
            _f_cloud,
            _f_u10,
            _f_v10,
            _f_precip,
            _f_longwave,
            _f_sst_obs,
            _f_sss_obs,
            _f_Tobs,
            _f_Sobs,
            _f_uprof,
            _f_vprof,
            _f_dtdx,
            _f_dtdy,
            _f_dsdx,
            _f_dsdy,
            _f_w_adv,
            _f_w_height,
            _f_zeta,
            _f_us0,
            _f_vs0,
            _f_ds,
            _f_us,
            _f_vs,
            _f_dusdz,
            _f_dvsdz,
            output.output_step,
            output.time,
            output.zeta,
            output.u_taus,
            output.u10,
            output.v10,
            output.airt,
            output.airp,
            output.hum,
            output.es,
            output.ea,
            output.qs,
            output.qa,
            output.rhoa,
            output.cloud,
            output.albedo,
            output.precip,
            output.evap,
            output.int_precip,
            output.int_evap,
            output.int_swr,
            output.int_heat,
            output.int_total,
            output.I_0,
            output.qh,
            output.qe,
            output.ql,
            output.heat,
            output.tx,
            output.ty,
            output.sst,
            output.sst_obs,
            output.sss,
            output.mld_surf,
            output.u_taub,
            output.taub,
            output.mld_bott,
            output.us0,
            output.vs0,
            output.ds,
            output.Ekin,
            output.Epot,
            output.Eturb,
            output.rho_p,
            output.rho,
            output.u,
            output.v,
            output.T,
            output.S,
            output.Tp,
            output.Ti,
            output.Sp,
            output.Tobs,
            output.Sobs,
            output.u_obs,
            output.v_obs,
            output.idpdx,
            output.idpdy,
            output.tke,
            output.eps,
            output.num,
            output.nuh,
            output.h,
            output.xP,
            output.fric,
            output.drag,
            output.avh,
            output.bioshade,
            output.ga,
            output.uu,
            output.vv,
            output.ww,
            output.NN,
            output.NNT,
            output.NNS,
            output.buoy,
            output.SS,
            output.P,
            output.B,
            output.Pb,
            output.kb,
            output.epsb,
            output.L,
            output.PSTK,
            output.cmue1,
            output.cmue2,
            output.as_,
            output.an,
            output.at,
            output.gamu,
            output.gamv,
            output.gamh,
            output.gams,
            output.Rig,
            output.gamb,
            output.gam,
            output.r,
            output.taux,
            output.tauy,
            output.rad,
            output.us,
            output.vs,
            output.dusdz,
            output.dvsdz,
            output.nus,
            output.nucl,
            output.z,
            output.zi,
            out_slot_base,
            write_ic,
            init_int_precip,
            init_int_evap,
            init_int_swr,
            init_int_heat,
            init_int_total,
            _hydro_store,
            _hydro_T,
            _hydro_S,
            _hydro_rho,
            _hydro_h,
            _hydro_nuh,
            _hydro_rad,
            _hydro_taub,
        )
    )
    if written > 0:
        _populate_simple_ice_reference_scalars(params, output, written)
    return written


@numba.njit(cache=True, fastmath=False)
def _kolpran_single(
    nlev: int,
    tke: np.ndarray,
    length_scale: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nus: np.ndarray,
    nucl: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    cmue3: np.ndarray,
) -> None:
    for i in range(nlev + 1):
        x = math.sqrt(tke[i]) * length_scale[i]
        num[i] = cmue1[i] * x
        nuh[i] = cmue2[i] * x
        nus[i] = cmue2[i] * x
        nucl[i] = cmue3[i] * x


@numba.njit(cache=True, fastmath=False)
def _copy_profile(nlev: int, source: np.ndarray, target: np.ndarray) -> None:
    for k in range(nlev + 1):
        target[k] = source[k]


@numba.njit(cache=True, fastmath=False)
def _step_wequation_single(
    nlev: int,
    w_adv_method: int,
    w_adv: float,
    w_height: float,
    zi: np.ndarray,
    w: np.ndarray,
) -> float:
    if w_adv_method != 1 and w_adv_method != 2:
        return w_height

    z_top = zi[nlev]
    z_bottom = zi[0]
    column_depth = z_top - z_bottom

    z_crit = z_top - 0.01 * column_depth
    if w_height > z_crit:
        w_height = z_crit
    z_crit = z_bottom + 0.01 * column_depth
    if w_height < z_crit:
        w_height = z_crit

    for i in range(1, nlev):
        if zi[i] > w_height:
            w[i] = (z_top - zi[i]) / (z_top - w_height) * w_adv
        else:
            w[i] = (z_bottom - zi[i]) / (z_bottom - w_height) * w_adv
    w[0] = 0.0
    w[nlev] = 0.0
    return w_height


@numba.njit(cache=True, fastmath=False)
def _density_value_compiled(
    density_method: int,
    rhob: float,
    alpha0: float,
    beta0: float,
    T0: float,
    S0: float,
    S_value: float,
    T_value: float,
    pressure: float,
) -> float:
    if density_method == 1:
        return float(gsw_rho(S_value, T_value, pressure))
    return rhob * (1.0 - alpha0 * (T_value - T0) + beta0 * (S_value - S0))


@numba.njit(cache=True, fastmath=False)
def _internal_pressure_gradients_single(
    nlev: int,
    gravity: float,
    rho0: float,
    density_method: int,
    rhob: float,
    alpha0: float,
    beta0: float,
    T0: float,
    S0: float,
    h: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    dsdx: np.ndarray,
    dsdy: np.ndarray,
    dtdx: np.ndarray,
    dtdy: np.ndarray,
    dxB: np.ndarray,
    dyB: np.ndarray,
    idpdx: np.ndarray,
    idpdy: np.ndarray,
) -> None:
    for i in range(nlev + 1):
        idpdx[i] = 0.0
        idpdy[i] = 0.0
        dxB[i] = 0.0
        dyB[i] = 0.0

    dx = 10.0
    dy = 10.0
    z = 0.0
    for i in range(nlev, 0, -1):
        z += 0.5 * h[i]

        dSS = dx * dsdx[i]
        dTT = dx * dtdx[i]
        Bl = (
            -gravity
            * (
                _density_value_compiled(
                    density_method, rhob, alpha0, beta0, T0, S0, S[i], T[i], z
                )
                - rho0
            )
            / rho0
        )
        Br = (
            -gravity
            * (
                _density_value_compiled(
                    density_method,
                    rhob,
                    alpha0,
                    beta0,
                    T0,
                    S0,
                    S[i] + dSS,
                    T[i] + dTT,
                    z,
                )
                - rho0
            )
            / rho0
        )
        dxB[i] = (Br - Bl) / dx

        dSS = dy * dsdy[i]
        dTT = dy * dtdy[i]
        Bl = (
            -gravity
            * (
                _density_value_compiled(
                    density_method, rhob, alpha0, beta0, T0, S0, S[i], T[i], z
                )
                - rho0
            )
            / rho0
        )
        Br = (
            -gravity
            * (
                _density_value_compiled(
                    density_method,
                    rhob,
                    alpha0,
                    beta0,
                    T0,
                    S0,
                    S[i] + dSS,
                    T[i] + dTT,
                    z,
                )
                - rho0
            )
            / rho0
        )
        dyB[i] = (Br - Bl) / dy

        z += 0.5 * h[i]

    acc = 0.5 * h[nlev] * dxB[nlev]
    idpdx[nlev] = acc
    for i in range(nlev - 1, 0, -1):
        acc += 0.5 * h[i + 1] * dxB[i + 1] + 0.5 * h[i] * dxB[i]
        idpdx[i] = acc

    acc = 0.5 * h[nlev] * dyB[nlev]
    idpdy[nlev] = acc
    for i in range(nlev - 1, 0, -1):
        acc += 0.5 * h[i + 1] * dyB[i + 1] + 0.5 * h[i] * dyB[i]
        idpdy[i] = acc


@numba.njit(cache=True, fastmath=False)
def _internal_pressure_plume_single(
    nlev: int,
    plume_type: int,
    plume_slope_x: float,
    plume_slope_y: float,
    buoy: np.ndarray,
    idpdx: np.ndarray,
    idpdy: np.ndarray,
) -> None:
    if plume_type == 1:
        for k in range(1, nlev + 1):
            idpdx[k] = plume_slope_x * (buoy[k] - buoy[1])
            idpdy[k] = plume_slope_y * (buoy[k] - buoy[1])
    elif plume_type == 2:
        for k in range(1, nlev + 1):
            idpdx[k] = -plume_slope_x * (buoy[nlev] - buoy[k])
            idpdy[k] = -plume_slope_y * (buoy[nlev] - buoy[k])


@numba.njit(cache=True, fastmath=False)
def _external_pressure_single(
    nlev: int,
    method: int,
    dpdx: float,
    dpdy: float,
    h_press: float,
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z_work: np.ndarray,
) -> None:
    if method == 1:
        z_current = 0.5 * h[1]
        z_next = z_current
        i = 1
        while True:
            if i < nlev:
                z_next = z_current + 0.5 * (h[i] + h[i + 1])
            else:
                z_next = z_current + h[nlev]
            if not (z_next < h_press and i < nlev):
                break
            i += 1
            z_current = z_next

        if i >= nlev:
            uint = u[nlev]
            vint = v[nlev]
        else:
            dz = z_next - z_current
            rat = 0.0
            if dz > 0.0:
                rat = (h_press - z_current) / dz
            if rat < 0.0:
                rat = 0.0
            elif rat > 1.0:
                rat = 1.0

            uint = rat * u[i + 1] + (1.0 - rat) * u[i]
            vint = rat * v[i + 1] + (1.0 - rat) * v[i]
        shift_u = dpdx - uint
        shift_v = dpdy - vint
        for k in range(1, nlev + 1):
            u[k] += shift_u
            v[k] += shift_v
    elif method == 2:
        hint = 0.0
        uint = 0.0
        vint = 0.0
        for k in range(1, nlev + 1):
            hint += h[k]
            uint += h[k] * u[k]
            vint += h[k] * v[k]
        uint /= hint
        vint /= hint
        shift_u = dpdx - uint
        shift_v = dpdy - vint
        for k in range(1, nlev + 1):
            u[k] += shift_u
            v[k] += shift_v


@numba.njit(cache=True, fastmath=False)
def _step_seagrass_single(
    nlev: int,
    dt: float,
    alpha: float,
    grassind: int,
    grassn: int,
    grassz: np.ndarray,
    exc: np.ndarray,
    vfric: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    drag: np.ndarray,
    xP: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xxP: np.ndarray,
    excur: np.ndarray,
    grassfric: np.ndarray,
    z_work: np.ndarray,
) -> None:
    if grassind < 1 or grassn < 1:
        return

    z_work[1] = 0.5 * h[1]
    for i in range(2, nlev + 1):
        z_work[i] = z_work[i - 1] + 0.5 * (h[i - 1] + h[i])

    for i in range(nlev + 1):
        excur[i] = 0.0
        grassfric[i] = 0.0
    for i in range(1, nlev + 1):
        if z_work[i] >= grassz[grassn]:
            excur[i] = exc[grassn]
            grassfric[i] = vfric[grassn]
        elif z_work[i] <= grassz[1]:
            excur[i] = exc[1]
            grassfric[i] = vfric[1]
        else:
            ii = 1
            while grassz[ii] <= z_work[i]:
                ii += 1
            rat = (z_work[i] - grassz[ii - 1]) / (grassz[ii] - grassz[ii - 1])
            excur[i] = (1.0 - rat) * exc[ii - 1] + rat * exc[ii]
            grassfric[i] = (1.0 - rat) * vfric[ii - 1] + rat * vfric[ii]

    upper = grassind
    if upper > nlev:
        upper = nlev
    for i in range(1, upper + 1):
        xx[i] += dt * u[i]
        yy[i] += dt * v[i]
        dist = math.sqrt(xx[i] * xx[i] + yy[i] * yy[i])
        if dist > excur[i] and dist > 0.0:
            xx[i] = excur[i] / dist * xx[i]
            yy[i] = excur[i] / dist * yy[i]
            drag[i] += grassfric[i]
            speed = math.sqrt(u[i] * u[i] + v[i] * v[i])
            xxP[i] = alpha * grassfric[i] * speed * speed * speed
        else:
            xxP[i] = 0.0

    for i in range(1, nlev):
        xP[i] = 0.5 * (xxP[i] + xxP[i + 1])


@numba.njit(cache=True, fastmath=False)
def _fortran_nint(value: float) -> int:
    return int(math.floor(value + 0.5))


@numba.njit(cache=True, fastmath=False)
def _saturation_vapor_pressure(temp_c: float) -> float:
    pressure_mb = 6.107799961 + temp_c * (
        4.436518521e-1
        + temp_c
        * (
            1.428945805e-2
            + temp_c
            * (
                2.650648471e-4
                + temp_c
                * (
                    3.031240396e-6
                    + temp_c * (2.034080948e-8 + temp_c * 6.136820929e-11)
                )
            )
        )
    )
    return pressure_mb * 100.0


@numba.njit(cache=True, fastmath=False)
def _specific_humidity(airp: float, vapour_pressure: float) -> float:
    return _CONST06 * vapour_pressure / (airp - 0.377 * vapour_pressure)


@numba.njit(cache=True, fastmath=False)
def _humidity_compiled(
    hum_method: int,
    hum: float,
    airp: float,
    tw: float,
    ta: float,
) -> tuple[float, float, float, float, float]:
    es = 0.98 * _saturation_vapor_pressure(tw)
    qs = _specific_humidity(airp, es)
    ea = 0.0
    qa = 0.0
    if hum_method == 1:
        ea = 0.01 * hum * _saturation_vapor_pressure(ta)
        qa = _specific_humidity(airp, ea)
    elif hum_method == 2:
        twet = hum if hum < 100.0 else hum - _KELVIN
        ea = _saturation_vapor_pressure(twet)
        ea = ea - 6.6e-4 * (1.0 + 1.15e-3 * twet) * airp * (ta - twet)
        qa = _specific_humidity(airp, ea)
    elif hum_method == 3:
        dew = hum if hum < 100.0 else hum - _KELVIN
        ea = _saturation_vapor_pressure(dew)
        qa = _specific_humidity(airp, ea)
    else:
        qa = hum
        ea = qa * airp / (_CONST06 + 0.378 * qa)
    rhoa = airp / (_RGAS * (ta + _KELVIN) * (1.0 + _CONST06 * qa))
    return es, ea, qs, qa, rhoa


@numba.njit(cache=True, fastmath=False)
def _solar_zenith_angle_compiled(
    yday: int,
    hour: float,
    dlon: float,
    dlat: float,
) -> float:
    rlon = DEG_TO_RAD * dlon
    rlat = DEG_TO_RAD * dlat
    th0 = 2.0 * PI * yday / 365.25
    th02 = 2.0 * th0
    th03 = 3.0 * th0
    sundec = (
        0.006918
        - 0.399912 * math.cos(th0)
        + 0.070257 * math.sin(th0)
        - 0.006758 * math.cos(th02)
        + 0.000907 * math.sin(th02)
        - 0.002697 * math.cos(th03)
        + 0.001480 * math.sin(th03)
    )
    thsun = (hour - 12.0) * 15.0 * DEG_TO_RAD + rlon
    coszen = math.sin(rlat) * math.sin(sundec) + math.cos(rlat) * math.cos(
        sundec
    ) * math.cos(thsun)
    if coszen < 0.0:
        coszen = 0.0
    return RAD_TO_DEG * math.acos(coszen)


@numba.njit(cache=True, fastmath=False)
def _shortwave_radiation_compiled(
    zenith_angle: float,
    yday: int,
    dlat: float,
    cloud: float,
) -> float:
    coszen = math.cos(DEG_TO_RAD * zenith_angle)
    if coszen <= 0.0:
        coszen = 0.0
        qatten = 0.0
    else:
        qatten = 0.7 ** (1.0 / coszen)
    qzer = coszen * SOLAR_CONSTANT_W_M2
    qdir = qzer * qatten
    qdiff = ((1.0 - 0.09) * qzer - qdir) * 0.5
    qtot = qdir + qdiff
    rlat = DEG_TO_RAD * dlat
    eqnx = (yday - 81.0) / 365.0 * 2.0 * PI
    sunbet = math.sin(rlat) * math.sin(23.439 * DEG_TO_RAD * math.sin(eqnx)) + (
        math.cos(rlat) * math.cos(23.439 * DEG_TO_RAD * math.sin(eqnx))
    )
    sunbet = math.asin(sunbet) * RAD_TO_DEG
    qshort = qtot * (1.0 - 0.62 * cloud + 0.0019 * sunbet)
    if qshort > qtot:
        qshort = qtot
    return qshort


@numba.njit(cache=True, fastmath=False)
def _albedo_payne_compiled(zen: float) -> float:
    if zen >= 74.0:
        jab = int(0.5 * (90.0 - zen) + 1.0)
    elif zen >= 50.0:
        jab = int(0.23 * (74.0 - zen) + 9.0)
    else:
        jab = int(0.10 * (50.0 - zen) + 15.0)
    if jab < 1:
        jab = 1
    if jab > 20:
        jab = 20
    idx = jab - 1
    if jab == 20:
        return _PAYNE_ALBEDO[idx]
    dzen = (_PAYNE_ZA[idx] - zen) / _PAYNE_DZA[idx]
    return _PAYNE_ALBEDO[idx] + dzen * (_PAYNE_ALBEDO[idx + 1] - _PAYNE_ALBEDO[idx])


@numba.njit(cache=True, fastmath=False)
def _longwave_clark_compiled(
    dlat: float,
    tw_k: float,
    ta_k: float,
    cloud: float,
    ea: float,
) -> float:
    idx = _fortran_nint(abs(dlat))
    if idx < 0:
        idx = 0
    if idx >= len(_CLOUD_CORRECTION_FACTOR):
        idx = len(_CLOUD_CORRECTION_FACTOR) - 1
    ccf = _CLOUD_CORRECTION_FACTOR[idx]
    x1 = (1.0 - ccf * cloud * cloud) * (tw_k**4)
    x2 = 0.39 - 0.05 * math.sqrt(ea * 0.01)
    x3 = 4.0 * (tw_k**3) * (tw_k - ta_k)
    return -_EMISS * _BOLZ * (x1 * x2 + x3)


@numba.njit(cache=True, fastmath=False)
def _longwave_compiled(
    method: int,
    longwave_type: int,
    prescribed_longwave: float,
    dlat: float,
    tw_k: float,
    ta_k: float,
    cloud: float,
    ea: float,
    qa: float,
) -> float:
    if method == 0:
        return prescribed_longwave
    if method == 2:
        if longwave_type == 1:
            return prescribed_longwave
        return prescribed_longwave - _BOLZ * _EMISS * (tw_k**4)

    idx = _fortran_nint(abs(dlat))
    if idx < 0:
        idx = 0
    if idx >= len(_CLOUD_CORRECTION_FACTOR):
        idx = len(_CLOUD_CORRECTION_FACTOR) - 1
    ccf = _CLOUD_CORRECTION_FACTOR[idx]

    if method == 3:
        x1 = (1.0 - ccf * cloud * cloud) * (tw_k**4)
        x2 = 0.39 - 0.05 * math.sqrt(ea * 0.01)
        x3 = 4.0 * (tw_k**3) * (tw_k - ta_k)
        return -_EMISS * _BOLZ * (x1 * x2 + x3)
    if method == 4:
        x1 = (1.0 - ccf * cloud * cloud) * (tw_k**4)
        x2 = 0.39 - 0.056 * math.sqrt(1000.0 * qa)
        x3 = 4.0 * (tw_k**3) * (tw_k - ta_k)
        return -_EMISS * _BOLZ * (x1 * x2 + x3)
    if method == 5:
        x1 = (1.0 + 0.1762 * cloud * cloud) * (ta_k**4)
        x2 = 0.653 + 0.00535 * (ea * 0.01)
        x3 = _EMISS * (tw_k**4)
        return -_BOLZ * (-x1 * x2 + x3)
    if method == 6:
        x1 = (1.0 - 0.6823 * cloud * cloud) * (ta_k**4)
        x2 = 0.39 - 0.05 * math.sqrt(0.01 * ea)
        x3 = 4.0 * ta_k**3 * (tw_k - ta_k)
        return -_EMISS * _BOLZ * (x1 * x2 + x3)
    if method == 7:
        x1 = _EMISS * tw_k**4
        x2 = (10.77 * cloud + 2.34) * cloud - 18.44
        x3 = 0.955 * (ta_k + x2) ** 4
        return -_BOLZ * (x1 - x3)
    if method == 8:
        ea_for = ea
        if ea_for < 10.0:
            ea_for = 10.0
        x1 = _EMISS * tw_k**4
        x2 = 34.07 + 4157.0 / math.log(2.1718e10 / ea_for)
        x2 = (10.77 * cloud + 2.34) * cloud - 18.44 + 0.84 * (x2 - ta_k + 4.01)
        x3 = 0.955 * (ta_k + x2) ** 4
        return -_BOLZ * (x1 - x3)
    return 0.0


@numba.njit(cache=True, fastmath=False)
def _net_shortwave_compiled(
    shortwave_method: int,
    shortwave_type: int,
    shortwave: float,
    albedo: float,
) -> float:
    if shortwave_method == 3 or shortwave_type == 2:
        return shortwave * (1.0 - albedo)
    return shortwave


@numba.njit(cache=True, fastmath=False)
def _output_sst_value(
    model_sst: float,
    observed_sst: float,
    sst_obs_method: int,
) -> float:
    if sst_obs_method == 2 and observed_sst == observed_sst:
        return observed_sst
    return model_sst


@numba.njit(cache=True, fastmath=False)
def _gsw_ct_from_pt_compiled(sa: float, pt: float) -> float:
    x2 = _GSW_SFAC * sa
    x = math.sqrt(x2)
    y = pt * 0.025
    pot_enthalpy = (
        61.01362420681071
        + y
        * (
            168776.46138048015
            + y
            * (
                -2735.2785605119625
                + y
                * (
                    2574.2164453821433
                    + y
                    * (
                        -1536.6644434977543
                        + y
                        * (
                            545.7340497931629
                            + (-50.91091728474331 - 18.30489878927802 * y) * y
                        )
                    )
                )
            )
        )
        + x2
        * (
            268.5520265845071
            + y
            * (
                -12019.028203559312
                + y
                * (
                    3734.858026725145
                    + y
                    * (
                        -2046.7671145057618
                        + y
                        * (
                            465.28655623826234
                            + (-0.6370820302376359 - 10.650848542359153 * y) * y
                        )
                    )
                )
            )
            + x
            * (
                937.2099110620707
                + y
                * (
                    588.1802812170108
                    + y
                    * (
                        248.39476522971285
                        + (-3.871557904936333 - 2.6268019854268356 * y) * y
                    )
                )
                + x
                * (
                    -1687.914374187449
                    + x
                    * (
                        246.9598888781377
                        + x * (123.59576582457964 - 48.5891069025409 * x)
                    )
                    + y
                    * (
                        936.3206544460336
                        + y
                        * (
                            -942.7827304544439
                            + y
                            * (
                                369.4389437509002
                                + (-33.83664947895248 - 9.987880382780322 * y) * y
                            )
                        )
                    )
                )
            )
        )
    )
    return pot_enthalpy / _GSW_CP0


@numba.njit(cache=True, fastmath=False)
def _gsw_gibbs_pt0_pt0_compiled(sa: float, pt0: float) -> float:
    x2 = _GSW_SFAC * sa
    x = math.sqrt(x2)
    y = pt0 * 0.025
    g03 = -24715.571866078 + y * (
        4420.4472249096725
        + y
        * (
            -1778.231237203896
            + y
            * (1160.5182516851419 + y * (-569.531539542516 + y * 128.13429152494615))
        )
    )
    g08 = x2 * (
        1760.062705994408
        + x
        * (
            -86.1329351956084
            + x
            * (
                -137.1145018408982
                + y
                * (
                    296.20061691375236
                    + y * (-205.67709290374563 + 49.9394019139016 * y)
                )
            )
        )
        + y * (-60.136422517125 + y * 10.50720794170734)
    ) + y * (
        -1351.605895580406
        + y * (1097.1125373015109 + y * (-433.20648175062206 + 63.905091254154904 * y))
    )
    return (g03 + g08) * 0.000625


@numba.njit(cache=True, fastmath=False)
def _gsw_t_from_ct_surface_compiled(sa: float, ct: float) -> float:
    s1 = sa / _GSW_UPS
    a0 = -1.446013646344788e-2
    a1 = -3.305308995852924e-3
    a2 = 1.062415929128982e-4
    a3 = 9.477566673794488e-1
    a4 = 2.166591947736613e-3
    a5 = 3.828842955039902e-3
    b0 = 1.0
    b1 = 6.506097115635800e-4
    b2 = 3.830289486850898e-3
    b3 = 1.247811760368034e-6

    a5ct = a5 * ct
    b3ct = b3 * ct
    ct_factor = a3 + a4 * s1 + a5ct
    pt_num = a0 + s1 * (a1 + a2 * s1) + ct * ct_factor
    pt_recden = 1.0 / (b0 + b1 * s1 + ct * (b2 + b3ct))
    pt = pt_num * pt_recden
    dpt_dct = (ct_factor + a5ct - (b2 + b3ct + b3ct) * pt) * pt_recden

    ct_diff = _gsw_ct_from_pt_compiled(sa, pt) - ct
    pt_old = pt
    pt = pt_old - ct_diff * dpt_dct
    ptm = 0.5 * (pt + pt_old)
    dpt_dct = -_GSW_CP0 / ((ptm + _GSW_T0) * _gsw_gibbs_pt0_pt0_compiled(sa, ptm))
    pt = pt_old - ct_diff * dpt_dct
    ct_diff = _gsw_ct_from_pt_compiled(sa, pt) - ct
    pt_old = pt
    return float(pt_old - ct_diff * dpt_dct)


@numba.njit(cache=True, fastmath=False)
def _kondo_compiled(
    sst: float,
    airt: float,
    u10: float,
    v10: float,
    precip: float,
    qs: float,
    qa: float,
    rhoa: float,
) -> tuple[float, float, float, float, float]:
    evap = 0.0
    w = math.sqrt(u10 * u10 + v10 * v10)
    latent_heat = (2.5 - 0.00234 * sst) * 1.0e6
    s0 = 0.25 * (sst - airt) / ((w + 1.0e-10) ** 2)
    s = s0 * abs(s0) / (abs(s0) + 0.01)
    eps = 1.0e-12
    if w < 2.2:
        x = math.log(w + eps)
        cdd = 1.08 * math.exp(-0.15 * x) * 1.0e-3
        chd = 1.185 * math.exp(-0.157 * x) * 1.0e-3
        ced = 1.23 * math.exp(-0.16 * x) * 1.0e-3
    elif w < 5.0:
        x = w + eps
        cdd = (0.771 + 0.0858 * x) * 1.0e-3
        chd = (0.927 + 0.0546 * x) * 1.0e-3
        ced = (0.969 + 0.0521 * x) * 1.0e-3
    elif w < 8.0:
        x = w + eps
        cdd = (0.867 + 0.0667 * x) * 1.0e-3
        chd = (1.15 + 0.01 * x) * 1.0e-3
        ced = (1.18 + 0.01 * x) * 1.0e-3
    elif w < 25.0:
        x = w + eps
        cdd = (1.2 + 0.025 * x) * 1.0e-3
        chd = (1.17 + 0.0075 * x - 0.00045 * (w - 8.0) ** 2) * 1.0e-3
        ced = (1.196 + 0.008 * x - 0.0004 * (w - 8.0) ** 2) * 1.0e-3
    else:
        x = w + eps
        cdd = 0.073 * x * 1.0e-3
        chd = (1.652 - 0.017 * x) * 1.0e-3
        ced = (1.68 - 0.016 * x) * 1.0e-3
    if s < 0.0:
        if s > -3.3:
            x = 0.1 + 0.03 * s + 0.9 * math.exp(4.8 * s)
        else:
            x = 0.0
        cdd = x * cdd
        chd = x * chd
        ced = x * ced
    else:
        root = math.sqrt(s)
        cdd = cdd * (1.0 + 0.47 * root)
        chd = chd * (1.0 + 0.63 * root)
        ced = ced * (1.0 + 0.63 * root)
    qh = -chd * _CPA * rhoa * w * (sst - airt)
    qe = -ced * latent_heat * rhoa * w * (qs - qa)
    tmp = cdd * rhoa * w
    taux = tmp * u10
    tauy = tmp * v10
    return evap, taux, tauy, qe, qh


@numba.njit(cache=True, fastmath=False)
def _airsea_kondo_compiled(
    yday: int,
    secondsofday: float,
    latitude: float,
    longitude: float,
    hum_method: int,
    shortwave_method: int,
    shortwave_type: int,
    longwave_method: int,
    longwave_type: int,
    albedo_method: int,
    shortwave_scale_factor: float,
    heat_scale_factor: float,
    const_albedo: float,
    sst: float,
    airp: float,
    airt: float,
    hum: float,
    cloud: float,
    u10: float,
    v10: float,
    precip: float,
    prescribed_longwave: float,
    prescribed_shortwave: float,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    tw = sst if sst < 100.0 else sst - _KELVIN
    tw_k = sst + _KELVIN if sst < 100.0 else sst
    ta = airt if airt < 100.0 else airt - _KELVIN
    ta_k = airt + _KELVIN if airt < 100.0 else airt
    es, ea, qs, qa, rhoa = _humidity_compiled(hum_method, hum, airp, tw, ta)
    ql = _longwave_compiled(
        longwave_method,
        longwave_type,
        prescribed_longwave,
        latitude,
        tw_k,
        ta_k,
        cloud,
        ea,
        qa,
    )
    evap, taux, tauy, qe, qh = _kondo_compiled(tw, ta, u10, v10, precip, qs, qa, rhoa)
    shortwave = prescribed_shortwave
    zenith = 0.0
    if shortwave_method == 3 or shortwave_type == 2:
        zenith = _solar_zenith_angle_compiled(
            yday, secondsofday / 3600.0, longitude, latitude
        )
    if shortwave_method == 3:
        shortwave = (
            _shortwave_radiation_compiled(zenith, yday, latitude, cloud)
            * shortwave_scale_factor
        )
    if shortwave_method == 3 or shortwave_type == 2:
        if albedo_method == 0:
            albedo = const_albedo
        else:
            albedo = _albedo_payne_compiled(zenith)
    else:
        albedo = 0.0
    heat = (ql + qe + qh) * heat_scale_factor
    return (
        evap,
        taux,
        tauy,
        heat,
        shortwave,
        albedo,
        es,
        ea,
        qs,
        qa,
        rhoa,
        qh,
        qe,
        ql,
    )


@numba.njit(cache=True, fastmath=False)
def _psi_fairall(iflag: int, zol: float) -> float:
    """Stability function psi for Fairall/Liu method (Liu et al. 1979)."""
    psi_value = 0.0
    if zol < 0.0:
        chik = (1.0 - 16.0 * zol) ** 0.25
        if iflag == 1:
            psik = (
                2.0 * math.log(0.5 * (1.0 + chik))
                + math.log(0.5 * (1.0 + chik * chik))
                - 2.0 * math.atan(chik)
                + 0.5 * math.pi
            )
        else:
            psik = 2.0 * math.log(0.5 * (1.0 + chik * chik))
        sqr3 = 1.7320508
        chic = (1.0 - 12.87 * zol) ** _FAIRALL_R3
        psic = (
            1.5 * math.log(_FAIRALL_R3 * (1.0 + chic + chic * chic))
            - sqr3 * math.atan((1.0 + 2.0 * chic) / sqr3)
            + math.pi / sqr3
        )
        fw = 1.0 / (1.0 + zol * zol)
        psi_value = fw * psik + (1.0 - fw) * psic
    elif zol > 0.0:
        psi_value = -4.7 * zol
    return psi_value


@numba.njit(cache=True, fastmath=False)
def _airsea_fairall_compiled(
    yday: int,
    secondsofday: float,
    latitude: float,
    longitude: float,
    hum_method: int,
    shortwave_method: int,
    shortwave_type: int,
    longwave_method: int,
    longwave_type: int,
    albedo_method: int,
    shortwave_scale_factor: float,
    heat_scale_factor: float,
    const_albedo: float,
    sst: float,
    airp: float,
    airt: float,
    hum: float,
    cloud: float,
    u10: float,
    v10: float,
    precip: float,
    prescribed_longwave: float,
    prescribed_shortwave: float,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Fairall COARE bulk flux kernel — Fairall et al. (1996), Liu et al. (1979)."""
    tw = sst if sst < 100.0 else sst - _KELVIN
    tw_k = sst + _KELVIN if sst < 100.0 else sst
    ta = airt if airt < 100.0 else airt - _KELVIN
    ta_k = airt + _KELVIN if airt < 100.0 else airt
    es, ea, qs, qa, rhoa = _humidity_compiled(hum_method, hum, airp, tw, ta)
    ql = _longwave_compiled(
        longwave_method,
        longwave_type,
        prescribed_longwave,
        latitude,
        tw_k,
        ta_k,
        cloud,
        ea,
        qa,
    )

    evap = 0.0
    w = math.sqrt(u10 * u10 + v10 * v10)
    qe = 0.0
    qh = 0.0
    taux = 0.0
    tauy = 0.0
    delw = math.sqrt(w * w + _FAIRALL_WGUST * _FAIRALL_WGUST)

    if delw != 0.0:
        vis_air = 1.326e-5 * (1.0 + ta * (6.542e-3 + ta * (8.301e-6 - 4.84e-9 * ta)))
        latent_heat = (2.501 - 0.00237 * tw) * 1.0e6
        ier = 0
        delq = qa - qs
        delt = ta - tw
        zow = 0.0005
        wstar = 0.04 * delw
        tstar = 0.04 * delt
        qstar = 0.04 * delq
        tvstar = tstar * (1.0 + 0.61 * qa) + 0.61 * ta_k * qstar
        ri = (
            _FAIRALL_G
            * _FAIRALL_ZW
            * (delt + 0.61 * ta_k * delq)
            / (ta_k * delw * delw)
        )
        wgus = 0.0

        if ri <= 0.25:
            for _ in range(_FAIRALL_ITERMAX):
                if ier >= 0:
                    ol = (
                        _FAIRALL_G
                        * _FAIRALL_KAPPA
                        * tvstar
                        / (ta_k * (1.0 + 0.61 * qa) * wstar * wstar)
                    )
                    zwol = _FAIRALL_ZW * ol
                    ztol = _FAIRALL_ZT * ol
                    zqol = _FAIRALL_ZQ * ol

                    wpsi = _psi_fairall(1, zwol)
                    tpsi = _psi_fairall(2, ztol)
                    qpsi = _psi_fairall(2, zqol)

                    zow = 0.011 * wstar * wstar / _FAIRALL_G + 0.11 * vis_air / wstar
                    wstar = delw * _FAIRALL_KAPPA / (math.log(_FAIRALL_ZW / zow) - wpsi)

                    rr = zow * wstar / vis_air
                    if 0.0 <= rr < 1000.0:
                        rt = 0.0
                        rq = 0.0
                        for k in range(8):
                            if _FAIRALL_LIU_RR[k] <= rr < _FAIRALL_LIU_RR[k + 1]:
                                rt = _FAIRALL_LIU_A0[k] * rr ** _FAIRALL_LIU_B0[k]
                                rq = _FAIRALL_LIU_A1[k] * rr ** _FAIRALL_LIU_B1[k]

                        cff = vis_air / wstar
                        zot = rt * cff
                        zoq = rq * cff
                        cff = _FAIRALL_KAPPA * _FAIRALL_FDG
                        tstar = delt * cff / (math.log(_FAIRALL_ZT / zot) - tpsi)
                        qstar = delq * cff / (math.log(_FAIRALL_ZQ / zoq) - qpsi)

                        tvstar = tstar * (1.0 + 0.61 * qa) + 0.61 * ta_k * qstar
                        bf = -_FAIRALL_G / ta_k * wstar * tvstar
                        if bf > 0.0:
                            wgus = _FAIRALL_BETA * (bf * _FAIRALL_ZABL) ** _FAIRALL_R3
                        else:
                            wgus = 0.0
                        delw = math.sqrt(w * w + wgus * wgus)
                    else:
                        ier = -2

            if ier >= 0:
                wspeed = math.sqrt(w * w + wgus * wgus)
                cd = wstar * wstar / (wspeed * wspeed)
                qh = _CPA * rhoa * wstar * tstar
                qe = latent_heat * rhoa * wstar * qstar
                upvel = -1.61 * wstar * qstar - (1.0 + 1.61 * qa) * wstar * tstar / ta_k
                qe = qe - rhoa * latent_heat * upvel * qa
                cff = rhoa * cd * wspeed
                taux = cff * u10
                tauy = cff * v10

    shortwave = prescribed_shortwave
    zenith = 0.0
    if shortwave_method == 3 or shortwave_type == 2:
        zenith = _solar_zenith_angle_compiled(
            yday, secondsofday / 3600.0, longitude, latitude
        )
    if shortwave_method == 3:
        shortwave = (
            _shortwave_radiation_compiled(zenith, yday, latitude, cloud)
            * shortwave_scale_factor
        )
    if shortwave_method == 3 or shortwave_type == 2:
        if albedo_method == 0:
            albedo = const_albedo
        else:
            albedo = _albedo_payne_compiled(zenith)
    else:
        albedo = 0.0
    heat = (ql + qe + qh) * heat_scale_factor
    return (
        evap,
        taux,
        tauy,
        heat,
        shortwave,
        albedo,
        es,
        ea,
        qs,
        qa,
        rhoa,
        qh,
        qe,
        ql,
    )


@numba.njit(cache=True, fastmath=False)
def _update_density_single(
    nlev: int,
    density_method: int,
    gravity: float,
    rho0: float,
    alpha0: float,
    beta0: float,
    T0: float,
    S0: float,
    rhob: float,
    T: np.ndarray,
    S: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    rho_p: np.ndarray,
    rho: np.ndarray,
    buoy: np.ndarray,
) -> None:
    """Mirror of Fortran do_density: update rho_p, alpha, beta, buoy each timestep."""
    if density_method == 1:
        # Full TEOS-10
        for k in range(1, nlev + 1):
            rho[k] = gsw_rho(S[k], T[k], -z[k])
            rho_p[k] = gsw_sigma0(S[k], T[k]) + 1000.0
        # Interior interfaces: average of adjacent cell-centre values
        for k in range(1, nlev):
            Si_k = 0.5 * (S[k] + S[k + 1])
            Ti_k = 0.5 * (T[k] + T[k + 1])
            pi_k = -zi[k]
            alpha[k] = gsw_alpha(Si_k, Ti_k, pi_k)
            beta[k] = gsw_beta(Si_k, Ti_k, pi_k)
        # Boundary interfaces: use boundary cell values
        alpha[0] = gsw_alpha(S[1], T[1], -zi[0])
        beta[0] = gsw_beta(S[1], T[1], -zi[0])
        alpha[nlev] = gsw_alpha(S[nlev], T[nlev], -zi[nlev])
        beta[nlev] = gsw_beta(S[nlev], T[nlev], -zi[nlev])
    else:
        # Linear EOS (methods 2 and 3): no pressure dependency is applied.
        for k in range(1, nlev + 1):
            rho_p[k] = rhob * (1.0 - alpha0 * (T[k] - T0) + beta0 * (S[k] - S0))
            rho[k] = rho_p[k]
    # Buoyancy from potential density anomaly
    for k in range(1, nlev + 1):
        buoy[k] = -gravity * (rho_p[k] - rho0) / rho0


@numba.njit(cache=True, fastmath=False)
def _stratification_from_alpha_beta_single(
    nlev: int,
    gravity: float,
    h: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    NN: np.ndarray,
    NNT: np.ndarray,
    NNS: np.ndarray,
) -> None:
    for i in range(1, nlev):
        idz = 2.0 / (h[i] + h[i + 1])
        dT = T[i + 1] - T[i]
        dS = S[i + 1] - S[i]
        NNT[i] = alpha[i] * gravity * dT * idz
        NNS[i] = -beta[i] * gravity * dS * idz
        NN[i] = NNT[i] + NNS[i]
    NNT[0] = 0.0
    NNT[nlev] = 0.0
    NNS[0] = 0.0
    NNS[nlev] = 0.0
    NN[0] = 0.0
    NN[nlev] = 0.0


@numba.njit(cache=True, fastmath=False)
def step_turbulence_first_order_single(
    nlev: int,
    dt: float,
    depth: float,
    stab_method: int,
    len_scale_method: int,
    iw_model: int,
    iw_alpha: float,
    sig_k: float,
    sig_w: float,
    sig_e: float,
    sig_e0: float,
    sig_peps: int,
    k_min: float,
    eps_min: float,
    k_ubc: int,
    k_lbc: int,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    cm0: float,
    cmsf: float,
    cde: float,
    kappa: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    galp: float,
    length_lim: int,
    prandtl0_fix: float,
    cc1: float,
    ct1: float,
    a2: float,
    a3: float,
    a5: float,
    cw1: float,
    cw2: float,
    cw3plus: float,
    cw3minus: float,
    cwx: float,
    cw4: float,
    ce1: float,
    ce2: float,
    ce3plus: float,
    ce3minus: float,
    cex: float,
    ce4: float,
    klimiw: float,
    rich_cr: float,
    numiw: float,
    nuhiw: float,
    numshear: float,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    SSU: np.ndarray,
    SSV: np.ndarray,
    xP: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
    tke: np.ndarray,
    tkeo: np.ndarray,
    eps: np.ndarray,
    omega: np.ndarray,
    L: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Pb: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nus: np.ndarray,
    nucl: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    cmue3: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    av: np.ndarray,
    aw: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    ww: np.ndarray,
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    work_avh: np.ndarray,
    sig_eff: np.ndarray,
    q_sour: np.ndarray,
    l_sour: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    """First-order turbulence closure (Fairall et al. stability functions).

    Equivalent to GOTM turb_method=first_order. Uses algebraic stability
    functions (cmue1, cmue2) rather than second-order moment equations.
    No kb/epsb algebraic steps; no second alpha-MNB call.
    """
    step_production_single(
        nlev,
        iw_model,
        iw_alpha,
        1,
        1,
        1,
        NN,
        SS,
        xP,
        SSCSTK,
        SSSTK,
        num,
        nuh,
        nucl,
        P,
        B,
        Pb,
        Px,
        PSTK,
    )
    step_alpha_mnb_single(
        nlev,
        0,
        0,
        tke,
        eps,
        tke,  # kb placeholder — first_order doesn't track kb
        NN,
        SS,
        SSCSTK,  # unused (has_sscstk=0)
        SSSTK,  # unused (has_ssstk=0)
        as_,  # correct output array
        an,  # correct output array
        at,
        av,
        aw,
    )
    # First-order stability functions
    for k in range(nlev + 1):
        cmue3[k] = 0.0
    if stab_method == 1:  # Constant
        for k in range(nlev + 1):
            cmue1[k] = cm0
            cmue2[k] = cm0 / prandtl0_fix
    elif stab_method == 2:  # Munk_Anderson
        _step_cmue_ma_single(nlev, cm0, prandtl0_fix, as_, an, cmue1, cmue2)
        cmue1[0] = cmue1[1]
        cmue1[nlev] = cmue1[nlev - 1]
        cmue2[0] = cmue2[1]
        cmue2[nlev] = cmue2[nlev - 1]
    else:  # Schumann_Gerz (stab_method == 3)
        _step_cmue_sg_single(nlev, cm0, prandtl0_fix, as_, an, cmue1, cmue2)
        cmue1[0] = cmue1[1]
        cmue1[nlev] = cmue1[nlev - 1]
        cmue2[0] = cmue2[1]
        cmue2[nlev] = cmue2[nlev - 1]
    step_tkeeq_single(
        nlev,
        dt,
        sig_k,
        k_min,
        k_ubc,
        k_lbc,
        ubc_type,
        lbc_type,
        cm0,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
        tke,
        tkeo,
        h,
        P,
        B,
        Px,
        PSTK,
        num,
        eps,
        work_avh,
        l_sour,
        q_sour,
        u_taus,
        u_taub,
        z0s,
        z0b,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )
    if len_scale_method == 8:  # diss_eq
        step_dissipationeq_single(
            nlev,
            dt,
            ce1,
            ce2,
            ce3plus,
            ce3minus,
            cex,
            ce4,
            cm0,
            cde,
            kappa,
            galp,
            sig_k,
            sig_e,
            sig_e0,
            sig_peps,
            length_lim,
            eps_min,
            psi_ubc,
            psi_lbc,
            ubc_type,
            lbc_type,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
            tke,
            tkeo,
            eps,
            L,
            h,
            NN,
            SS,
            P,
            B,
            Px,
            PSTK,
            num,
            work_avh,
            sig_eff,
            l_sour,
            q_sour,
            u_taus,
            u_taub,
            z0s,
            z0b,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )
    else:  # omega_eq
        step_omegaeq_single(
            nlev,
            dt,
            cw1,
            cw2,
            cw3plus,
            cw3minus,
            cwx,
            cw4,
            sig_w,
            cm0,
            kappa,
            cde,
            galp,
            length_lim,
            eps_min,
            psi_ubc,
            psi_lbc,
            ubc_type,
            lbc_type,
            sig_k,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
            tke,
            tkeo,
            eps,
            L,
            h,
            NN,
            SS,
            P,
            B,
            Px,
            PSTK,
            num,
            u_taus,
            u_taub,
            z0s,
            z0b,
            omega,
            work_avh,
            l_sour,
            q_sour,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )
    _kolpran_single(nlev, tke, L, num, nuh, nus, nucl, cmue1, cmue2, cmue3)
    if iw_model == 2:
        step_internal_wave_single(
            nlev,
            iw_model,
            klimiw,
            rich_cr,
            numiw,
            nuhiw,
            numshear,
            tke,
            num,
            nuh,
            NN,
            SS,
        )
    step_variances_single(
        nlev,
        cc1,
        ct1,
        a2,
        a3,
        a5,
        tke,
        eps,
        P,
        B,
        Px,
        num,
        SSU,
        SSV,
        uu,
        vv,
        ww,
    )


@numba.njit(cache=True, fastmath=False)
def _run_second_order_turbulence_single(
    nlev: int,
    dt: float,
    depth: float,
    tke_method: int,
    len_scale_method: int,
    scnd_method: int,
    iw_model: int,
    iw_alpha: float,
    sig_k: float,
    sig_w: float,
    sig_e: float,
    sig_e0: float,
    sig_peps: int,
    k_min: float,
    eps_min: float,
    kb_min: float,
    epsb_min: float,
    k_ubc: int,
    k_lbc: int,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    cm0: float,
    cmsf: float,
    cde: float,
    kappa: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    galp: float,
    length_lim: int,
    my_b1: float,
    my_sq: float,
    my_sl: float,
    my_e1: float,
    my_e2: float,
    my_e3: float,
    my_ex: float,
    my_e6: float,
    my_length: int,
    cc1: float,
    ct1: float,
    ctt: float,
    a1: float,
    a2: float,
    a3: float,
    a5: float,
    at1: float,
    at2: float,
    at3: float,
    at5: float,
    cw1: float,
    cw2: float,
    cw3plus: float,
    cw3minus: float,
    cwx: float,
    cw4: float,
    ce1: float,
    ce2: float,
    ce3plus: float,
    ce3minus: float,
    cex: float,
    ce4: float,
    klimiw: float,
    rich_cr: float,
    numiw: float,
    nuhiw: float,
    numshear: float,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    SSU: np.ndarray,
    SSV: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
    xP: np.ndarray,
    tke: np.ndarray,
    tkeo: np.ndarray,
    eps: np.ndarray,
    omega: np.ndarray,
    L: np.ndarray,
    kb: np.ndarray,
    epsb: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Pb: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nus: np.ndarray,
    nucl: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    cmue3: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    av: np.ndarray,
    aw: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    ww: np.ndarray,
    sq_var: np.ndarray,
    sl_var: np.ndarray,
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    work_avh: np.ndarray,
    sig_eff: np.ndarray,
    q_sour: np.ndarray,
    l_sour: np.ndarray,
    q2l: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    step_production_single(
        nlev,
        iw_model,
        iw_alpha,
        1,
        1,
        1,
        NN,
        SS,
        xP,
        SSCSTK,
        SSSTK,
        num,
        nuh,
        nucl,
        P,
        B,
        Pb,
        Px,
        PSTK,
    )
    step_alpha_mnb_single(
        nlev,
        0,
        0,
        tke,
        eps,
        kb,
        NN,
        SS,
        SSCSTK,
        SSSTK,
        as_,
        an,
        at,
        av,
        aw,
    )
    if scnd_method == 1:
        step_cmue_d_single(
            nlev,
            cm0,
            cc1,
            ct1,
            a1,
            a2,
            a3,
            a5,
            at1,
            at2,
            at3,
            at5,
            as_,
            an,
            cmue1,
            cmue2,
        )
        for k in range(nlev + 1):
            cmue3[k] = 0.0
    else:
        step_cmue_c_single(
            nlev,
            cm0,
            cc1,
            ct1,
            a1,
            a2,
            a3,
            a5,
            at1,
            at2,
            at3,
            at5,
            as_,
            an,
            cmue1,
            cmue2,
        )
        for k in range(nlev + 1):
            cmue3[k] = 0.0
    if tke_method == 3:  # tke_MY
        _step_q2over2eq_single(
            nlev,
            dt,
            k_min,
            my_b1,
            k_ubc,
            k_lbc,
            ubc_type,
            lbc_type,
            my_sq,
            cw,
            gen_alpha,
            gen_l,
            tke,
            tkeo,
            h,
            P,
            B,
            Px,
            PSTK,
            eps,
            L,
            sq_var,
            work_avh,
            l_sour,
            q_sour,
            u_taus,
            u_taub,
            z0s,
            z0b,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )
    else:  # tke_keps
        step_tkeeq_single(
            nlev,
            dt,
            sig_k,
            k_min,
            k_ubc,
            k_lbc,
            ubc_type,
            lbc_type,
            cm0,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
            tke,
            tkeo,
            h,
            P,
            B,
            Px,
            PSTK,
            num,
            eps,
            work_avh,
            l_sour,
            q_sour,
            u_taus,
            u_taub,
            z0s,
            z0b,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )
    step_kbalgebraic_single(nlev, ctt, kb_min, tke, eps, kb, Pb)
    if len_scale_method == 8:
        step_dissipationeq_single(
            nlev,
            dt,
            ce1,
            ce2,
            ce3plus,
            ce3minus,
            cex,
            ce4,
            cm0,
            cde,
            kappa,
            galp,
            sig_k,
            sig_e,
            sig_e0,
            sig_peps,
            length_lim,
            eps_min,
            psi_ubc,
            psi_lbc,
            ubc_type,
            lbc_type,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
            tke,
            tkeo,
            eps,
            L,
            h,
            NN,
            SS,
            P,
            B,
            Px,
            PSTK,
            num,
            work_avh,
            sig_eff,
            l_sour,
            q_sour,
            u_taus,
            u_taub,
            z0s,
            z0b,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )
    elif len_scale_method == 9:  # length_eq (Mellor-Yamada q2l)
        _step_lengthscaleeq_single(
            nlev,
            dt,
            k_min,
            eps_min,
            kappa,
            my_e1,
            my_e2,
            my_e3,
            my_ex,
            my_e6,
            my_b1,
            cde,
            my_length,
            galp,
            length_lim,
            psi_ubc,
            psi_lbc,
            ubc_type,
            lbc_type,
            my_sl,
            my_sq,
            cw,
            gen_alpha,
            gen_l,
            tke,
            tkeo,
            eps,
            L,
            h,
            NN,
            SS,
            P,
            B,
            Px,
            PSTK,
            sl_var,
            depth,
            u_taus,
            u_taub,
            z0s,
            z0b,
            q2l,
            work_avh,
            l_sour,
            q_sour,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )
    else:  # omega_eq
        step_omegaeq_single(
            nlev,
            dt,
            cw1,
            cw2,
            cw3plus,
            cw3minus,
            cwx,
            cw4,
            sig_w,
            cm0,
            kappa,
            cde,
            galp,
            length_lim,
            eps_min,
            psi_ubc,
            psi_lbc,
            ubc_type,
            lbc_type,
            sig_k,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
            tke,
            tkeo,
            eps,
            L,
            h,
            NN,
            SS,
            P,
            B,
            Px,
            PSTK,
            num,
            u_taus,
            u_taub,
            z0s,
            z0b,
            omega,
            work_avh,
            l_sour,
            q_sour,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )
    step_epsbalgebraic_single(nlev, ctt, epsb_min, tke, eps, kb, epsb)
    step_alpha_mnb_single(
        nlev,
        0,
        0,
        tke,
        eps,
        kb,
        NN,
        SS,
        SSCSTK,
        SSSTK,
        as_,
        an,
        at,
        av,
        aw,
    )
    _kolpran_single(nlev, tke, L, num, nuh, nus, nucl, cmue1, cmue2, cmue3)
    if iw_model == 2:
        step_internal_wave_single(
            nlev,
            iw_model,
            klimiw,
            rich_cr,
            numiw,
            nuhiw,
            numshear,
            tke,
            num,
            nuh,
            NN,
            SS,
        )
    step_variances_single(
        nlev,
        cc1,
        ct1,
        a2,
        a3,
        a5,
        tke,
        eps,
        P,
        B,
        Px,
        num,
        SSU,
        SSV,
        uu,
        vv,
        ww,
    )


def warmup_couette_step_routines(
    params: RuntimeParams,
    state: RuntimeState,
    work: RuntimeWork,
) -> None:
    """Directly call major Couette step routines once to populate signatures."""

    state.tx[0] = params.tx
    state.ty[0] = params.ty
    step_uequation_single(
        params.nlev,
        params.dt,
        params.cnpar,
        params.avmolu,
        params.gravity,
        0,
        0,
        0,
        0,
        0,
        params.tx,
        params.dzetadx,
        state.u,
        state.uo,
        state.v,
        state.h,
        state.w,
        state.drag,
        state.num,
        state.nucl,
        work.dusdz,
        work.idpdx,
        work.uprof,
        work.vel_relax_tau,
        work.avh,
        work.q_sour,
        work.l_sour,
        work.au,
        work.bu,
        work.cu,
        work.du,
        work.ru,
        work.qu,
        work.adv_cu,
    )
    step_vequation_single(
        params.nlev,
        params.dt,
        params.cnpar,
        params.avmolu,
        params.gravity,
        0,
        0,
        0,
        0,
        params.ty,
        params.dzetady,
        state.v,
        state.vo,
        state.u,
        state.h,
        state.w,
        state.drag,
        state.num,
        state.nucl,
        work.dvsdz,
        work.idpdy,
        work.vprof,
        work.vel_relax_tau,
        work.avh,
        work.q_sour,
        work.l_sour,
        work.au,
        work.bu,
        work.cu,
        work.du,
        work.ru,
        work.qu,
        work.adv_cu,
    )
    step_friction_single(
        params.nlev,
        params.kappa,
        params.avmolu,
        params.rho0,
        params.gravity,
        params.h0b,
        params.z0s_min,
        params.charnock,
        params.charnock_val,
        params.calc_bottom_stress,
        params.max_it_z0b,
        0,
        1,
        state.h,
        state.u,
        state.v,
        state.drag,
        state.z0b,
        state.z0s,
        state.za,
        state.u_taub,
        state.u_taubo,
        state.u_taus,
        state.taub,
        state.tx,
        state.ty,
    )
    step_shear_single(
        params.nlev,
        params.cnpar,
        state.h,
        state.u,
        state.v,
        state.uo,
        state.vo,
        work.dusdz,
        work.dvsdz,
        state.SS,
        state.SSU,
        state.SSV,
        state.SSCSTK,
        state.SSSTK,
    )
    step_production_single(
        params.nlev,
        params.iw_model,
        0.0,
        1,
        1,
        1,
        state.NN,
        state.SS,
        state.xP,
        state.SSCSTK,
        state.SSSTK,
        state.num,
        state.nuh,
        state.nucl,
        state.P,
        state.B,
        state.Pb,
        state.Px,
        state.PSTK,
    )
    step_alpha_mnb_single(
        params.nlev,
        0,
        0,
        state.tke,
        state.eps,
        state.kb,
        state.NN,
        state.SS,
        state.SSCSTK,
        state.SSSTK,
        state.as_,
        state.an,
        state.at,
        state.av,
        state.aw,
    )
    step_cmue_c_single(
        params.nlev,
        params.cm0,
        params.cc1,
        params.ct1,
        params.a1,
        params.a2,
        params.a3,
        params.a5,
        params.at1,
        params.at2,
        params.at3,
        params.at5,
        state.as_,
        state.an,
        state.cmue1,
        state.cmue2,
    )
    state.cmue3.fill(0.0)
    step_tkeeq_single(
        params.nlev,
        params.dt,
        params.sig_k,
        params.k_min,
        params.k_ubc,
        params.k_lbc,
        params.ubc_type,
        params.lbc_type,
        params.cm0,
        params.cmsf,
        params.cw,
        params.gen_alpha,
        params.gen_l,
        state.tke,
        state.tkeo,
        state.h,
        state.P,
        state.B,
        state.Px,
        state.PSTK,
        state.num,
        state.eps,
        work.avh,
        work.l_sour,
        work.q_sour,
        state.u_taus[0],
        state.u_taub[0],
        state.z0s[0],
        state.z0b[0],
        work.au,
        work.bu,
        work.cu,
        work.du,
        work.ru,
        work.qu,
    )
    step_kbalgebraic_single(
        params.nlev,
        params.ctt,
        params.kb_min,
        state.tke,
        state.eps,
        state.kb,
        state.Pb,
    )
    step_omegaeq_single(
        params.nlev,
        params.dt,
        params.cw1,
        params.cw2,
        params.cw3plus,
        params.cw3minus,
        params.cwx,
        params.cw4,
        params.sig_w,
        params.cm0,
        params.kappa,
        params.cde,
        params.galp,
        params.length_lim,
        params.eps_min,
        params.psi_ubc,
        params.psi_lbc,
        params.ubc_type,
        params.lbc_type,
        params.sig_k,
        params.cmsf,
        params.cw,
        params.gen_alpha,
        params.gen_l,
        state.tke,
        state.tkeo,
        state.eps,
        state.L,
        state.h,
        state.NN,
        state.SS,
        state.P,
        state.B,
        state.Px,
        state.PSTK,
        state.num,
        state.u_taus[0],
        state.u_taub[0],
        state.z0s[0],
        state.z0b[0],
        state.omega,
        work.avh,
        work.l_sour,
        work.q_sour,
        work.au,
        work.bu,
        work.cu,
        work.du,
        work.ru,
        work.qu,
    )
    step_epsbalgebraic_single(
        params.nlev,
        params.ctt,
        params.epsb_min,
        state.tke,
        state.eps,
        state.kb,
        state.epsb,
    )
    _kolpran_single(
        params.nlev,
        state.tke,
        state.L,
        state.num,
        state.nuh,
        state.nus,
        state.nucl,
        state.cmue1,
        state.cmue2,
        state.cmue3,
    )
