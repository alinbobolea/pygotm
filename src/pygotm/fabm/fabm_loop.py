"""FABM time loop driven by stored GOTM hydrodynamic state."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pygotm.util.adv_center import CONSERVATIVE, FLUX, P2_PDM, adv_center
from pygotm.util.diff_center import NEUMANN, diff_center

if TYPE_CHECKING:
    from pygotm.fabm.engine import FABMEngine
    from pygotm.gotm.runtime_output import RuntimeOutput
    from pygotm.gotm.runtime_params import RuntimeParams

__all__ = ["run_fabm_chunk", "run_fabm_loop"]


def run_fabm_chunk(
    engine: FABMEngine,
    chunk_params: RuntimeParams,
    output: RuntimeOutput,
    hydro_T: np.ndarray,
    hydro_S: np.ndarray,
    hydro_rho: np.ndarray,
    hydro_h: np.ndarray,
    hydro_nuh: np.ndarray,
    hydro_rad: np.ndarray,
    cc_in: np.ndarray | None,
    out_index_base: int,
    forcing_u10: np.ndarray | None = None,
    forcing_v10: np.ndarray | None = None,
    forcing_yearday: np.ndarray | None = None,
    forcing_secondsofday: np.ndarray | None = None,
    hydro_taub: np.ndarray | None = None,
    is_first_chunk: bool = True,
) -> tuple[np.ndarray, int]:
    """Run pyfabm for one physics chunk and return updated (cc, out_index).

    Parameters
    ----------
    cc_in : None on first chunk — engine.start() sets IC; np.ndarray on
            subsequent chunks (shape n_vars × nlev) carrying FABM state.
    out_index_base : output slot counter before this chunk starts.
    is_first_chunk : True on the first chunk — triggers engine.start() and
                     records the IC diagnostic slot.

    Returns
    -------
    cc : updated FABM state at end of chunk (shape n_vars × nlev)
    out_index : updated output slot counter after this chunk
    """

    model = engine.model
    nlev = chunk_params.nlev
    dt = chunk_params.dt
    cnpar = chunk_params.cnpar
    nt = chunk_params.nt
    output_every = output.output_every
    force_final = output.force_final
    light_A = chunk_params.light_A
    light_g1 = chunk_params.light_g1
    light_g2 = chunk_params.light_g2

    sv_names = engine.state_variable_names()
    n_vars = len(sv_names)

    sv_to_ref: list[np.ndarray | None] = []
    for name in sv_names:
        norm_name = name.replace("/", "_")
        arr = output.reference_z_profiles.get(norm_name)
        sv_to_ref.append(arr)

    _au = np.zeros(nlev + 1, dtype=np.float64)
    _bu = np.zeros(nlev + 1, dtype=np.float64)
    _cu = np.zeros(nlev + 1, dtype=np.float64)
    _du = np.zeros(nlev + 1, dtype=np.float64)
    _ru = np.zeros(nlev + 1, dtype=np.float64)
    _qu = np.zeros(nlev + 1, dtype=np.float64)
    _l_sour = np.zeros(nlev + 1, dtype=np.float64)
    _q_sour = np.zeros(nlev + 1, dtype=np.float64)
    _tau_r = np.full(nlev + 1, 1.0e15, dtype=np.float64)
    _y_obs = np.zeros(nlev + 1, dtype=np.float64)
    _y = np.zeros(nlev + 1, dtype=np.float64)
    _ws = np.zeros(nlev + 1, dtype=np.float64)
    _adv_cu = np.zeros(nlev + 1, dtype=np.float64)

    out_index = out_index_base

    if is_first_chunk:
        _set_environment(
            engine,
            model,
            nlev,
            hydro_T[0],
            hydro_S[0],
            hydro_rho[0],
            hydro_h[0],
            hydro_rad[0],
            u10=float(forcing_u10[0]) if forcing_u10 is not None else None,
            v10=float(forcing_v10[0]) if forcing_v10 is not None else None,
            yearday=_fabm_day_of_year(
                float(forcing_yearday[0]) if forcing_yearday is not None else None,
                (
                    float(forcing_secondsofday[0])
                    if forcing_secondsofday is not None
                    else None
                ),
            ),
            taub=float(hydro_taub[0]) if hydro_taub is not None else None,
            light_A=light_A,
            light_g1=light_g1,
            light_g2=light_g2,
        )
        engine.start()
        cc = np.zeros((n_vars, nlev), dtype=np.float64)
        _read_model_state_into(model, cc)
        _set_model_state(model, cc)
        engine.get_rates(surface=False, bottom=False)
        _update_light_from_diagnostics(
            engine, nlev, hydro_h[0], hydro_rad[0], light_A, light_g2
        )
        engine.get_rates(surface=False, bottom=False)
        _record_fabm_output(engine, cc, sv_to_ref, output, out_index, nlev)
        out_index += 1
    else:
        assert cc_in is not None, "cc_in must be provided on non-first chunks"
        cc = cc_in.copy()

    for step in range(1, nt + 1):
        h_step = hydro_h[step]
        nuh_step = hydro_nuh[step]

        _set_environment(
            engine,
            model,
            nlev,
            hydro_T[step],
            hydro_S[step],
            hydro_rho[step],
            h_step,
            hydro_rad[step],
            u10=float(forcing_u10[step]) if forcing_u10 is not None else None,
            v10=float(forcing_v10[step]) if forcing_v10 is not None else None,
            yearday=_fabm_day_of_year(
                float(forcing_yearday[step]) if forcing_yearday is not None else None,
                (
                    float(forcing_secondsofday[step])
                    if forcing_secondsofday is not None
                    else None
                ),
            ),
            taub=float(hydro_taub[step]) if hydro_taub is not None else None,
            light_A=light_A,
            light_g1=light_g1,
            light_g2=light_g2,
        )

        # Transport order matches Fortran gotm_fabm.F90:
        # 1. Sinking/rising advection (conservative, zero-flux BCs)
        # 2. Turbulent diffusion
        # 3. Compute rates on post-transport state
        # 4. Apply source rates
        #
        # get_vertical_movement() works before getRates() because FABM sinking
        # velocities are constant parameters for BSEM (and typically constant
        # for other models too).  Calling it before the expensive getRates()
        # keeps the order consistent with gotm_fabm.F90 lines 1161-1199.
        vert_move = engine.get_vertical_movement()
        if (
            vert_move is not None
            and vert_move.ndim == 2
            and vert_move.shape[0] == n_vars
            and vert_move.shape[1] == nlev
        ):
            _apply_sinking(
                vert_move, h_step, cc, nlev, dt, n_vars, _y, _ws, _adv_cu
            )

        # Turbulent diffusion
        for var in range(n_vars):
            _y[1 : nlev + 1] = cc[var, :]
            diff_center(
                nlev,
                dt,
                cnpar,
                0,
                h_step,
                NEUMANN,
                NEUMANN,
                0.0,
                0.0,
                nuh_step,
                _l_sour,
                _q_sour,
                _tau_r,
                _y_obs,
                _y,
                _au,
                _bu,
                _cu,
                _du,
                _ru,
                _qu,
            )
            cc[var, :] = _y[1 : nlev + 1]

        # Source/sink rates computed on post-transport state (matches Fortran)
        _set_model_state(model, cc)
        engine.get_rates(surface=False, bottom=False)
        _update_light_from_diagnostics(
            engine, nlev, h_step, hydro_rad[step], light_A, light_g2
        )
        bulk_rates = engine.get_rates(surface=False, bottom=False).copy()
        surf_rates = engine.get_rates(surface=True, bottom=False).copy()
        bot_rates = engine.get_rates(surface=False, bottom=True)

        cc += dt * bulk_rates
        cc[:, -1] += dt * (surf_rates[:, -1] - bulk_rates[:, -1])
        cc[:, 0] += dt * (bot_rates[:, 0] - bulk_rates[:, 0])
        _set_model_state(model, cc)

        is_output = (step % output_every == 0) or (
            force_final and step == nt and nt % output_every != 0
        )
        if is_output:
            engine.get_rates(surface=False, bottom=False)
            _update_light_from_diagnostics(
                engine, nlev, h_step, hydro_rad[step], light_A, light_g2
            )
            engine.get_rates(surface=False, bottom=False)
            _record_fabm_output(engine, cc, sv_to_ref, output, out_index, nlev)
            out_index += 1

    return cc, out_index


def run_fabm_loop(
    engine: FABMEngine,
    params: RuntimeParams,
    output: RuntimeOutput,
    hydro_T: np.ndarray,
    hydro_S: np.ndarray,
    hydro_rho: np.ndarray,
    hydro_h: np.ndarray,
    hydro_nuh: np.ndarray,
    hydro_rad: np.ndarray,
    forcing_u10: np.ndarray | None = None,
    forcing_v10: np.ndarray | None = None,
    forcing_yearday: np.ndarray | None = None,
    forcing_secondsofday: np.ndarray | None = None,
    hydro_taub: np.ndarray | None = None,
) -> None:
    """Run pyfabm over every stored GOTM hydro step, then fill reference outputs.

    The compiled Numba loop already completed. Hydrodynamic state buffers hold
    T, S, rho, h, nuh, rad at every step (shape ``(nt+1, nlev+1)``). This
    function walks those steps in Python, coupling pyfabm to stored GOTM state
    without entering any Numba kernel.

    Reference profile and scalar outputs are written into
    ``output.reference_z_profiles`` and ``output.reference_scalars`` at the
    same output-slot indices as the physics outputs.

    Optional *forcing_u10* / *forcing_v10* (shape ``(nt+1,)``) provide surface
    wind components for FABM models that need ``wind_speed``. *forcing_yearday*
    and *forcing_secondsofday* provide the fractional GOTM calendar day for
    models needing
    ``number_of_days_since_start_of_the_year``.
    """

    _, out_index = run_fabm_chunk(
        engine,
        params,
        output,
        hydro_T,
        hydro_S,
        hydro_rho,
        hydro_h,
        hydro_nuh,
        hydro_rad,
        cc_in=None,
        out_index_base=0,
        forcing_u10=forcing_u10,
        forcing_v10=forcing_v10,
        forcing_yearday=forcing_yearday,
        forcing_secondsofday=forcing_secondsofday,
        hydro_taub=hydro_taub,
        is_first_chunk=True,
    )
    _record_scalar_diagnostics(engine, output, out_index)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fabm_day_of_year(
    yearday: float | None,
    secondsofday: float | None,
) -> float | None:
    if yearday is None:
        return None
    if secondsofday is None:
        return float(yearday)
    return float(yearday) - 1.0 + float(secondsofday) / 86400.0


def _center_depths(h: np.ndarray, nlev: int) -> np.ndarray:
    depth = np.zeros(nlev, dtype=np.float64)
    if nlev == 0:
        return depth
    depth[nlev - 1] = 0.5 * h[nlev]
    for idx in range(nlev - 2, -1, -1):
        depth[idx] = depth[idx + 1] + 0.5 * (h[idx + 1] + h[idx + 2])
    return depth


def _update_light_from_diagnostics(
    engine: FABMEngine,
    nlev: int,
    h: np.ndarray,
    rad: np.ndarray,
    light_A: float,
    light_g2: float,
) -> None:
    if light_g2 <= 0.0:
        return
    diagnostics = engine.diagnostics()
    raw_attenuation = diagnostics.get(
        "attenuation_coefficient_of_photosynthetic_radiative_flux"
    )
    if raw_attenuation is None:
        return
    attenuation = np.asarray(raw_attenuation, dtype=np.float64)
    if attenuation.shape[0] < nlev:
        return
    par_col, surface_par = _par_with_bioext_from_attenuation(
        attenuation[:nlev],
        h,
        rad,
        nlev,
        light_A,
        light_g2,
    )
    _try_set(engine, "downwelling_photosynthetic_radiative_flux", par_col)
    _try_set_scalar(
        engine, "surface_downwelling_photosynthetic_radiative_flux", surface_par
    )


def _par_with_bioext_from_attenuation(
    attenuation: np.ndarray,
    h: np.ndarray,
    rad: np.ndarray,
    nlev: int,
    light_A: float,
    light_g2: float,
) -> tuple[np.ndarray, float]:
    depth = _center_depths(h, nlev)
    h_col = h[1 : nlev + 1]
    surface_par = float(rad[nlev]) * (1.0 - light_A)
    par_col = np.zeros(nlev, dtype=np.float64)
    local_ext = np.maximum(0.0, attenuation[:nlev])
    bioext = 0.0
    for idx in range(nlev - 1, -1, -1):
        bioext += float(local_ext[idx] * h_col[idx] * 0.5)
        par_col[idx] = max(
            0.0,
            surface_par * math.exp(-float(depth[idx]) / light_g2 - bioext),
        )
        bioext += float(local_ext[idx] * h_col[idx] * 0.5)
    return par_col, surface_par


def _apply_sinking(
    vert_move: np.ndarray,
    h_step: np.ndarray,
    cc: np.ndarray,
    nlev: int,
    dt: float,
    n_vars: int,
    _y: np.ndarray,
    _ws: np.ndarray,
    _adv_cu: np.ndarray,
) -> None:
    """Apply sinking/rising advection for each FABM state variable.

    Replicates Fortran gotm_fabm.F90 lines 1190-1199:
    - Thickness-weighted interpolation of cell-centre velocities to faces.
    - Conservative adv_center with zero-flux boundary conditions.

    Parameters
    ----------
    vert_move : shape (n_vars, nlev) — cell-centre vertical velocities [m s⁻¹],
                positive upward, as returned by engine.get_vertical_movement().
    h_step    : shape (nlev+1,) — cell thicknesses for the current step.
    """
    if nlev < 2:
        return

    # Thickness-weighted face-velocity interpolation (Fortran iweights formula).
    # Face k (k=1..nlev-1) lies between cell k (lower) and cell k+1 (upper).
    # iweight_k = h[k+1] / (h[k] + h[k+1])  (Fortran 1-indexed)
    # Python: h_step[k] = h[k], so face k → h_lower=h_step[k], h_upper=h_step[k+1]
    h_lower = h_step[1:nlev]       # shape (nlev-1,): lower-cell thicknesses
    h_upper = h_step[2 : nlev + 1]  # shape (nlev-1,): upper-cell thicknesses
    h_sum = h_lower + h_upper
    iw = h_upper / h_sum  # shape (nlev-1,): weight for the lower cell

    for var in range(n_vars):
        vm = vert_move[var]  # shape (nlev,): cell-centre velocity for this var
        if not np.any(vm != 0.0):
            continue

        # Boundary faces stay zero (Fortran ws1d(0)=ws1d(nlev)=0).
        _ws[:] = 0.0
        # Interior faces 1..nlev-1:
        # ws[k] = iw[k-1]*vm[k-1] + (1-iw[k-1])*vm[k]
        # vm[k-1] = lower-cell velocity (0-indexed → Fortran cell k)
        # vm[k]   = upper-cell velocity (0-indexed → Fortran cell k+1)
        _ws[1:nlev] = iw * vm[: nlev - 1] + (1.0 - iw) * vm[1:nlev]

        _y[1 : nlev + 1] = cc[var, :]
        adv_center(
            nlev,
            dt,
            h_step,
            h_step,
            _ws,
            FLUX,
            FLUX,
            0.0,
            0.0,
            P2_PDM,
            CONSERVATIVE,
            _y,
            _adv_cu,
        )
        cc[var, :] = _y[1 : nlev + 1]


def _read_model_state_into(model: object, cc: np.ndarray) -> None:
    if cc.shape[0] == 0:
        return
    state_attr = getattr(model, "state", None)
    if state_attr is not None:
        arr = np.asarray(state_attr, dtype=np.float64)
        if arr.shape == cc.shape:
            cc[:] = arr
            return
    for attr in ("state_variables", "stateVariables"):
        variables = getattr(model, attr, None)
        if variables is None:
            continue
        for idx, var in enumerate(variables):
            if idx >= cc.shape[0]:
                break
            val = getattr(var, "value", 0.0)
            if isinstance(val, np.ndarray):
                cc[idx, : val.shape[0]] = val
            else:
                cc[idx, :] = float(val)
        return


def _set_model_state(model: object, cc: np.ndarray) -> None:
    state_attr = getattr(model, "state", None)
    if state_attr is not None and isinstance(state_attr, np.ndarray):
        if state_attr.shape == cc.shape:
            np.copyto(state_attr, cc)
            return
    for attr in ("state_variables", "stateVariables"):
        variables = getattr(model, attr, None)
        if variables is None:
            continue
        for idx, var in enumerate(variables):
            if idx >= cc.shape[0]:
                break
            val = getattr(var, "value", None)
            if isinstance(val, np.ndarray):
                np.copyto(val, cc[idx])
            else:
                var.value = cc[idx]
        return


def _set_environment(
    engine: FABMEngine,
    model: object,
    nlev: int,
    T: np.ndarray,
    S: np.ndarray,
    rho: np.ndarray,
    h: np.ndarray,
    rad: np.ndarray,
    u10: float | None = None,
    v10: float | None = None,
    yearday: float | None = None,
    taub: float | None = None,
    light_A: float = 0.0,
    light_g1: float = 1.0e10,
    light_g2: float = 1.0e10,
) -> None:
    T_col = np.ascontiguousarray(T[1 : nlev + 1], dtype=np.float64)
    S_col = np.ascontiguousarray(S[1 : nlev + 1], dtype=np.float64)
    rho_col = np.ascontiguousarray(rho[1 : nlev + 1], dtype=np.float64)
    h_col = np.ascontiguousarray(h[1 : nlev + 1], dtype=np.float64)

    try:
        model.cell_thickness = h_col  # type: ignore[attr-defined]
    except AttributeError:
        pass

    _try_set(engine, "temperature", T_col)
    _try_set(engine, "practical_salinity", S_col)
    _try_set(engine, "density", rho_col)

    # Pass visible-only PAR to FABM at cell centers. Fortran gotm_fabm.F90
    # links PAR on interior z-levels and computes it from the same center-depth
    # profile used for other FABM environmental dependencies.
    i_0 = float(rad[nlev])
    if light_g2 > 0.0 and i_0 > 0.0:
        depth = _center_depths(h, nlev)
        surface_par = i_0 * (1.0 - light_A)
        par_col = np.ascontiguousarray(
            np.maximum(0.0, surface_par * np.exp(-depth / light_g2)),
            dtype=np.float64,
        )
    else:
        par_col = np.ascontiguousarray(rad[1 : nlev + 1], dtype=np.float64)
        surface_par = i_0

    _try_set(engine, "downwelling_photosynthetic_radiative_flux", par_col)
    _try_set_scalar(
        engine, "surface_downwelling_photosynthetic_radiative_flux", surface_par
    )

    if u10 is not None and v10 is not None:
        wspd = float(np.sqrt(u10 * u10 + v10 * v10))
        _try_set_scalar(engine, "wind_speed", wspd)

    if yearday is not None:
        _try_set_scalar(engine, "number_of_days_since_start_of_the_year", yearday)

    if taub is not None:
        _try_set_scalar(engine, "bottom_stress", taub)


def _try_set(engine: FABMEngine, name: str, value: np.ndarray) -> None:
    try:
        if engine.has_dependency(name):
            engine.set_dependency(name, value)
    except (KeyError, RuntimeError):
        pass


def _try_set_scalar(engine: FABMEngine, name: str, value: float) -> None:
    try:
        if engine.has_dependency(name):
            engine.set_dependency(name, value)
    except (KeyError, RuntimeError):
        pass


def _record_fabm_output(
    engine: FABMEngine,
    cc: np.ndarray,
    sv_to_ref: list[np.ndarray | None],
    output: RuntimeOutput,
    slot: int,
    nlev: int,
) -> None:
    if slot >= output.nout:
        return

    for var_idx, arr in enumerate(sv_to_ref):
        if arr is None or var_idx >= cc.shape[0]:
            continue
        arr[slot, 1 : nlev + 1] = cc[var_idx, :]

    diags = engine.diagnostics()
    for name, diag_val in diags.items():
        norm_name = name.replace("/", "_")
        arr = output.reference_z_profiles.get(norm_name)
        if arr is None:
            continue
        if isinstance(diag_val, np.ndarray) and diag_val.ndim == 1:
            n = min(diag_val.shape[0], nlev)
            arr[slot, 1 : n + 1] = diag_val[:n]


def _record_scalar_diagnostics(
    engine: FABMEngine,
    output: RuntimeOutput,
    n_written: int,
) -> None:
    if n_written == 0:
        return
    diags = engine.diagnostics()
    norm_diags = {k.replace("/", "_"): v for k, v in diags.items()}
    for name, arr in output.reference_scalars.items():
        val = norm_diags.get(name)
        if val is None:
            continue
        if isinstance(val, (int, float, np.floating)):
            arr[:n_written] = float(val)
        elif isinstance(val, np.ndarray) and val.ndim == 0:
            arr[:n_written] = float(val)
