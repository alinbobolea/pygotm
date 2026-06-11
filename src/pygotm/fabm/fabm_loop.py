"""FABM time loop driven by stored GOTM hydrodynamic state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygotm.fabm.gotm_fabm import (
    center_depths_single,
    par_from_background_single,
    par_with_bioext_from_attenuation_single,
    step_fabm_lake_advection_single,
    step_fabm_post_rates_single,
    step_fabm_transport_single,
)
from pygotm.util.adv_center import CONSERVATIVE, FLUX, P2_PDM, adv_center

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
    forcing_precip: np.ndarray | None = None,
    hydro_taub: np.ndarray | None = None,
    hydro_Vc: np.ndarray | None = None,
    hydro_Vco: np.ndarray | None = None,
    hydro_Afo: np.ndarray | None = None,
    hydro_wq: np.ndarray | None = None,
    hydro_Qres: np.ndarray | None = None,
    hydro_stream_Q: np.ndarray | None = None,
    forcing_stream_flow: np.ndarray | None = None,
    forcing_stream_concentrations: dict[str, np.ndarray] | None = None,
    forcing_stream_concentration_masks: dict[str, np.ndarray] | None = None,
    step_offset: int = 0,
    is_first_chunk: bool = True,
    output_reduce_mode: int = 0,
    repair_state: bool = False,
) -> tuple[np.ndarray, int]:
    """Run pyfabm for one physics chunk and return updated (cc, out_index).

    Parameters
    ----------
    cc_in : np.ndarray or None
        None on first chunk — engine.start() sets initial conditions.
        Shape ``(n_vars, nlev)`` array carrying FABM state on subsequent chunks.
    out_index_base : int
        Output slot counter before this chunk starts.
    is_first_chunk : bool
        True on the first chunk — triggers engine.start() and records the IC
        diagnostic slot.
    output_reduce_mode : int
        Temporal output reduction matching GOTM ``time_method``: ``0`` =
        instantaneous snapshot (``point``), ``1`` = window mean (``mean``),
        ``2`` = window sum (``integrated``). For ``mean``/``integrated`` the
        FABM state and diagnostics are accumulated over every sub-step of each
        output window and the window mean/sum is written, mirroring the
        physics time loop's in-loop reduction. The initial-condition slot is
        always written instantaneously (GOTM writes ``reduced[0] = raw[0]``).
    repair_state : bool
        When True, clip every FABM state variable back inside its registered
        ``[minimum, maximum]`` after transport and after the source-term update,
        mirroring GOTM-FABM ``do_repair_state`` (enabled by the GOTM
        ``fabm: repair_state`` flag, e.g. lake_erken). Required to keep the
        explicit integration from driving tracers negative into NaN territory.

    Returns
    -------
    cc : np.ndarray
        Updated FABM state at end of chunk, shape ``(n_vars, nlev)``.
    out_index : int
        Updated output slot counter after this chunk.
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

    state_names = engine.state_variable_names()
    n_vars = len(state_names)
    n_interior = _interior_state_variable_count(model)
    n_surface = _surface_state_variable_count(model)
    n_bottom = _bottom_state_variable_count(model)

    state_z_refs: list[tuple[int, np.ndarray]] = []
    state_scalar_refs: list[tuple[int, np.ndarray, str]] = []
    for idx, name in enumerate(state_names):
        norm_name = name.replace("/", "_")
        z_arr = _z_profile_output(output, norm_name)
        if z_arr is not None and idx < n_interior:
            state_z_refs.append((idx, z_arr))
        scalar_arr = _scalar_output(output, norm_name)
        if scalar_arr is not None:
            state_scalar_refs.append((idx, scalar_arr, norm_name))

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
    nstreams = int(chunk_params.nstreams)
    _stream_concentrations = np.zeros((n_interior, nstreams), dtype=np.float64)
    _stream_has_concentration = np.zeros((n_interior, nstreams), dtype=np.int64)
    _no_river_dilution = _state_no_river_dilution(model, n_interior)
    stream_concentration_refs = _stream_concentration_refs(
        state_names,
        n_interior,
        nstreams,
        forcing_stream_concentrations,
        forcing_stream_concentration_masks,
    )

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
        engine.get_rates(surface=False, bottom=False, time=float(step_offset))
        _update_light_from_diagnostics(
            engine, nlev, hydro_h[0], hydro_rad[0], light_A, light_g2
        )
        engine.get_rates(surface=False, bottom=False, time=float(step_offset))
        engine.get_rates(surface=True, bottom=True, time=float(step_offset))
        _record_fabm_output(
            engine,
            cc,
            state_z_refs,
            state_scalar_refs,
            output,
            out_index,
            nlev,
        )
        out_index += 1
    else:
        assert cc_in is not None, "cc_in must be provided on non-first chunks"
        cc = cc_in.copy()

    # Temporal output reduction (GOTM ``time_method``). When active, accumulate
    # the FABM state and diagnostics over every sub-step of an output window and
    # emit the window mean/sum, matching the physics time loop. The point-mode
    # (``output_reduce_mode == 0``) path is left untouched so cases that request
    # instantaneous output remain bit-for-bit unchanged.
    reduce_output = output_reduce_mode != 0
    acc_cc = np.zeros((n_vars, nlev), dtype=np.float64) if reduce_output else cc
    acc_diags: dict[str, np.ndarray | float] = {}
    acc_count = 0

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
        has_vert_move = (
            vert_move is not None
            and vert_move.ndim == 2
            and vert_move.shape[0] >= n_interior
            and vert_move.shape[1] == nlev
        )

        # Turbulent diffusion
        precip = float(forcing_precip[step]) if forcing_precip is not None else 0.0
        if has_vert_move:
            assert vert_move is not None
            vert_move_arg = vert_move[:n_interior]
        else:
            vert_move_arg = cc
        if (
            chunk_params.lake != 0
            and hydro_Vc is not None
            and hydro_Vco is not None
            and hydro_Afo is not None
            and hydro_wq is not None
            and hydro_Qres is not None
            and hydro_stream_Q is not None
            and forcing_stream_flow is not None
        ):
            _fill_stream_concentration_step(
                step,
                stream_concentration_refs,
                _stream_concentrations,
                _stream_has_concentration,
            )
            step_fabm_lake_advection_single(
                nlev,
                dt,
                int(chunk_params.w_adv_discr),
                n_interior,
                forcing_stream_flow[step],
                hydro_stream_Q[step],
                _stream_concentrations,
                _stream_has_concentration,
                _no_river_dilution,
                h_step,
                hydro_Vco[step],
                hydro_Vc[step],
                hydro_Afo[step],
                hydro_wq[step],
                hydro_Qres[step],
                cc,
                _y,
                _adv_cu,
                _l_sour,
                _q_sour,
            )
            _l_sour.fill(0.0)
            _q_sour.fill(0.0)
        step_fabm_transport_single(
            nlev,
            dt,
            cnpar,
            precip,
            1 if has_vert_move else 0,
            n_interior,
            vert_move_arg,
            h_step,
            nuh_step,
            cc,
            _y,
            _ws,
            _adv_cu,
            _au,
            _bu,
            _cu,
            _du,
            _ru,
            _qu,
            _l_sour,
            _q_sour,
            _tau_r,
            _y_obs,
        )

        is_output = (step % output_every == 0) or (
            force_final and step == nt and nt % output_every != 0
        )

        # Source/sink rates computed on post-transport state (matches Fortran)
        _set_model_state(model, cc)
        # GOTM-FABM calls do_repair_state after advection/diffusion when
        # repair_state is enabled, clipping every state variable back inside its
        # registered [minimum, maximum] before the source terms are evaluated.
        # Without it the explicit update can drive a fast-cycling tracer (e.g.
        # selmaprotbas phosphate) negative, and the biogeochemical rate laws then
        # produce NaNs that contaminate the whole column.
        if repair_state:
            _repair_state(model, cc)
        fabm_time = float(step_offset + step)
        engine.get_rates(surface=False, bottom=False, time=fabm_time)
        _update_light_from_diagnostics(
            engine, nlev, h_step, hydro_rad[step], light_A, light_g2
        )
        bulk_rates = engine.get_rates(
            surface=False, bottom=False, time=fabm_time
        ).copy()
        surf_rates = engine.get_rates(surface=True, bottom=False, time=fabm_time).copy()
        bot_rates = engine.get_rates(surface=False, bottom=True, time=fabm_time)
        # Diagnostics reflect the post-transport state (matches GOTM
        # save_diagnostics in the first ODE-solver stage). In reduction mode we
        # accumulate them every sub-step; in point mode we only snapshot at the
        # output step, exactly as before.
        if reduce_output:
            _accumulate_diagnostics(acc_diags, engine.diagnostics())
            output_diagnostics = None
        else:
            output_diagnostics = engine.diagnostics() if is_output else None

        step_fabm_post_rates_single(
            nlev,
            dt,
            n_interior,
            n_surface,
            n_bottom,
            bulk_rates,
            surf_rates,
            bot_rates,
            cc,
        )
        _set_model_state(model, cc)
        # Second do_repair_state, after the source-term integration.
        if repair_state:
            _repair_state(model, cc)

        if reduce_output:
            # Accumulate the post-reaction state (the value GOTM writes at the
            # end of each timestep) over the output window.
            acc_cc += cc
            acc_count += 1
            if is_output and acc_count > 0:
                inv = 1.0 / acc_count if output_reduce_mode == 1 else 1.0
                mean_cc = acc_cc * inv
                mean_diags = _reduce_accumulated_diagnostics(acc_diags, inv)
                _record_fabm_output(
                    engine,
                    mean_cc,
                    state_z_refs,
                    state_scalar_refs,
                    output,
                    out_index,
                    nlev,
                    diagnostics=mean_diags,
                )
                out_index += 1
                acc_cc.fill(0.0)
                acc_diags.clear()
                acc_count = 0
        elif is_output:
            _record_fabm_output(
                engine,
                cc,
                state_z_refs,
                state_scalar_refs,
                output,
                out_index,
                nlev,
                diagnostics=output_diagnostics,
            )
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
    forcing_precip: np.ndarray | None = None,
    hydro_taub: np.ndarray | None = None,
    hydro_Vc: np.ndarray | None = None,
    hydro_Vco: np.ndarray | None = None,
    hydro_Afo: np.ndarray | None = None,
    hydro_wq: np.ndarray | None = None,
    hydro_Qres: np.ndarray | None = None,
    hydro_stream_Q: np.ndarray | None = None,
    forcing_stream_flow: np.ndarray | None = None,
    forcing_stream_concentrations: dict[str, np.ndarray] | None = None,
    forcing_stream_concentration_masks: dict[str, np.ndarray] | None = None,
    output_reduce_mode: int = 0,
    repair_state: bool = False,
) -> None:
    """Run pyfabm over every stored GOTM hydro step, then fill reference outputs.

    The compiled Numba loop already completed. Hydrodynamic state buffers hold
    T, S, rho, h, nuh, rad at every step (shape ``(nt+1, nlev+1)``). This
    function walks those steps in Python, coupling pyfabm to stored GOTM state
    without entering any Numba kernel.

    FABM profile and scalar outputs are written into ``output.fabm_z_profiles``
    and ``output.fabm_scalars`` at the same output-slot indices as the physics
    outputs.

    Optional *forcing_u10* / *forcing_v10* (shape ``(nt+1,)``) provide surface
    wind components for FABM models that need ``wind_speed``. *forcing_yearday*
    and *forcing_secondsofday* provide the fractional GOTM calendar day for
    models needing
    ``number_of_days_since_start_of_the_year``.
    """

    run_fabm_chunk(
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
        forcing_precip=forcing_precip,
        hydro_taub=hydro_taub,
        hydro_Vc=hydro_Vc,
        hydro_Vco=hydro_Vco,
        hydro_Afo=hydro_Afo,
        hydro_wq=hydro_wq,
        hydro_Qres=hydro_Qres,
        hydro_stream_Q=hydro_stream_Q,
        forcing_stream_flow=forcing_stream_flow,
        forcing_stream_concentrations=forcing_stream_concentrations,
        forcing_stream_concentration_masks=forcing_stream_concentration_masks,
        is_first_chunk=True,
        output_reduce_mode=output_reduce_mode,
        repair_state=repair_state,
    )


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


def _state_no_river_dilution(model: object, n_interior: int) -> np.ndarray:
    flags = np.zeros(n_interior, dtype=np.int64)
    variables = getattr(model, "state_variables", None)
    if variables is None:
        variables = getattr(model, "stateVariables", None)
    if variables is None:
        return flags
    for idx, variable in enumerate(list(variables)[:n_interior]):
        flags[idx] = 1 if bool(getattr(variable, "no_river_dilution", False)) else 0
    return flags


def _stream_concentration_refs(
    state_names: tuple[str, ...],
    n_interior: int,
    nstreams: int,
    concentrations: dict[str, np.ndarray] | None,
    masks: dict[str, np.ndarray] | None,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if nstreams == 0 or not concentrations or not masks:
        return []
    refs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for idx, name in enumerate(state_names[:n_interior]):
        norm_name = name.replace("/", "_")
        values = concentrations.get(norm_name)
        mask = masks.get(norm_name)
        if values is None or mask is None:
            continue
        if values.ndim != 2 or values.shape[1] != nstreams:
            continue
        if mask.ndim != 1 or mask.shape[0] != nstreams:
            continue
        refs.append((idx, values, mask))
    return refs


def _fill_stream_concentration_step(
    step: int,
    refs: list[tuple[int, np.ndarray, np.ndarray]],
    concentrations: np.ndarray,
    has_concentration: np.ndarray,
) -> None:
    concentrations.fill(0.0)
    has_concentration.fill(0)
    for idx, values, mask in refs:
        concentrations[idx, :] = values[step, :]
        has_concentration[idx, :] = mask


def _center_depths(h: np.ndarray, nlev: int) -> np.ndarray:
    depth = np.zeros(nlev, dtype=np.float64)
    center_depths_single(nlev, h, depth)
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
    raw_attenuation = engine.diagnostic(
        "attenuation_coefficient_of_photosynthetic_radiative_flux",
        copy=False,
    )
    if raw_attenuation is None:
        return
    if not isinstance(raw_attenuation, np.ndarray) or raw_attenuation.ndim == 0:
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
    depth = np.zeros(nlev, dtype=np.float64)
    par_col = np.zeros(nlev, dtype=np.float64)
    surface_par = par_with_bioext_from_attenuation_single(
        nlev,
        attenuation,
        h,
        rad,
        light_A,
        light_g2,
        depth,
        par_col,
    )
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
    h_lower = h_step[1:nlev]  # shape (nlev-1,): lower-cell thicknesses
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


def _interior_state_variable_count(model: object) -> int:
    variables = getattr(model, "interior_state_variables", None)
    if variables is None:
        variables = getattr(model, "bulk_state_variables", None)
    if variables is None:
        variables = getattr(model, "state_variables", None)
    return len(variables) if variables is not None else 0


def _surface_state_variable_count(model: object) -> int:
    variables = getattr(model, "surface_state_variables", None)
    return len(variables) if variables is not None else 0


def _bottom_state_variable_count(model: object) -> int:
    variables = getattr(model, "bottom_state_variables", None)
    return len(variables) if variables is not None else 0


def _repair_state(model: object, cc: np.ndarray) -> None:
    """Clip FABM state to its registered bounds (GOTM ``do_repair_state``).

    Assumes the model's internal state already holds ``cc``. Uses pyfabm's
    ``check_state(repair=True)``, which clamps every state variable to its
    ``[minimum, maximum]`` (the same routine GOTM-FABM drives via
    ``fabm_check_state``), then reads the repaired state back into ``cc``. Falls
    back to clipping negatives to zero if the pyfabm build predates
    ``check_state``.
    """
    check = getattr(model, "check_state", None)
    if callable(check):
        check(repair=True)
        _read_model_state_into(model, cc)
    else:
        np.maximum(cc, 0.0, out=cc)


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
        depth = np.zeros(nlev, dtype=np.float64)
        par_col = np.zeros(nlev, dtype=np.float64)
        surface_par = par_from_background_single(
            nlev,
            h,
            rad,
            light_A,
            light_g2,
            depth,
            par_col,
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
    setter = getattr(engine, "set_dependency_if_present", None)
    if callable(setter):
        setter(name, value)
        return
    try:
        if engine.has_dependency(name):
            engine.set_dependency(name, value)
    except KeyError:
        pass


def _try_set_scalar(engine: FABMEngine, name: str, value: float) -> None:
    setter = getattr(engine, "set_dependency_if_present", None)
    if callable(setter):
        setter(name, value)
        return
    try:
        if engine.has_dependency(name):
            engine.set_dependency(name, value)
    except KeyError:
        pass


def _z_profile_output(output: RuntimeOutput, name: str) -> np.ndarray | None:
    return output.fabm_z_profiles.get(name)


def _scalar_output(output: RuntimeOutput, name: str) -> np.ndarray | None:
    return output.fabm_scalars.get(name)


def _record_fabm_output(
    engine: FABMEngine,
    cc: np.ndarray,
    state_z_refs: list[tuple[int, np.ndarray]],
    state_scalar_refs: list[tuple[int, np.ndarray, str]],
    output: RuntimeOutput,
    slot: int,
    nlev: int,
    *,
    diagnostics: dict[str, np.ndarray | float] | None = None,
) -> None:
    if slot >= output.nout:
        return

    for var_idx, profile_ref in state_z_refs:
        if var_idx >= cc.shape[0]:
            continue
        profile_ref[slot, 1 : nlev + 1] = cc[var_idx, :]

    for var_idx, scalar_ref, name in state_scalar_refs:
        if var_idx >= cc.shape[0]:
            continue
        scalar_ref[slot] = _scalar_from_profile(name, cc[var_idx], nlev)

    diags = diagnostics if diagnostics is not None else engine.diagnostics()
    for name, diag_val in diags.items():
        norm_name = name.replace("/", "_")
        profile_arr = _z_profile_output(output, norm_name)
        if (
            profile_arr is not None
            and isinstance(diag_val, np.ndarray)
            and diag_val.ndim == 1
        ):
            n = min(diag_val.shape[0], nlev)
            profile_arr[slot, 1 : n + 1] = diag_val[:n]
        scalar_arr = _scalar_output(output, norm_name)
        if scalar_arr is not None:
            scalar_arr[slot] = _scalar_from_value(norm_name, diag_val, nlev)


def _accumulate_diagnostics(
    acc: dict[str, np.ndarray | float],
    step_diags: dict[str, np.ndarray | float],
) -> None:
    """Add one sub-step's FABM diagnostics into the running window accumulator.

    On the first sub-step of a window we copy the value (so the accumulator can
    never alias an engine-owned buffer that the next ``get_rates`` overwrites);
    on later sub-steps we add in place. This computes the window sum; the caller
    divides by the sub-step count to obtain the temporal mean.
    """
    for name, value in step_diags.items():
        existing = acc.get(name)
        if existing is None:
            acc[name] = value.copy() if isinstance(value, np.ndarray) else float(value)
        elif isinstance(existing, np.ndarray) and isinstance(value, np.ndarray):
            existing += value
        else:
            acc[name] = float(existing) + float(value)


def _reduce_accumulated_diagnostics(
    acc: dict[str, np.ndarray | float],
    inv: float,
) -> dict[str, np.ndarray | float]:
    """Scale accumulated diagnostics by ``inv`` (1/count for mean, 1 for sum)."""
    reduced: dict[str, np.ndarray | float] = {}
    for name, value in acc.items():
        if isinstance(value, np.ndarray):
            reduced[name] = value * inv
        else:
            reduced[name] = float(value) * inv
    return reduced


def _copy_diagnostics(
    engine_diagnostics: dict[str, np.ndarray | float],
) -> dict[str, np.ndarray | float]:
    copied: dict[str, np.ndarray | float] = {}
    for name, value in engine_diagnostics.items():
        if isinstance(value, np.ndarray):
            copied[name] = value.copy()
        else:
            copied[name] = float(value)
    return copied


_SURFACE_SCALAR_NAMES = frozenset(
    {
        "jrc_med_ergom_OFL",
    }
)


def _scalar_from_value(
    name: str,
    value: np.ndarray | float,
    nlev: int,
) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return _scalar_from_profile(name, array, nlev)


def _scalar_from_profile(name: str, value: np.ndarray, nlev: int) -> float:
    del nlev
    if value.size == 0:
        return 0.0
    index = value.shape[0] - 1 if name in _SURFACE_SCALAR_NAMES else 0
    index = max(0, min(index, value.shape[0] - 1))
    return float(value[index])
