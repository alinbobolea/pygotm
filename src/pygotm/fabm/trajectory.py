"""FABM post-loop integration using stored physics trajectory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygotm.util.diff_center import NEUMANN, diff_center

if TYPE_CHECKING:
    from pygotm.fabm.engine import FABMEngine
    from pygotm.gotm.runtime_output import RuntimeOutput
    from pygotm.gotm.runtime_params import RuntimeParams

__all__ = ["integrate_fabm_chunk", "integrate_fabm_from_trajectory"]


def integrate_fabm_chunk(
    engine: FABMEngine,
    chunk_params: RuntimeParams,
    output: RuntimeOutput,
    traj_T: np.ndarray,
    traj_S: np.ndarray,
    traj_rho: np.ndarray,
    traj_h: np.ndarray,
    traj_nuh: np.ndarray,
    traj_rad: np.ndarray,
    cc_in: np.ndarray | None,
    out_index_base: int,
    forcing_u10: np.ndarray | None = None,
    forcing_v10: np.ndarray | None = None,
    forcing_yearday: np.ndarray | None = None,
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

    out_index = out_index_base

    if is_first_chunk:
        _set_environment(
            engine, model, nlev, traj_T[0], traj_S[0], traj_rho[0], traj_h[0], traj_rad[0],
            u10=float(forcing_u10[0]) if forcing_u10 is not None else None,
            v10=float(forcing_v10[0]) if forcing_v10 is not None else None,
            yearday=float(forcing_yearday[0]) if forcing_yearday is not None else None,
        )
        engine.start()
        cc = np.zeros((n_vars, nlev), dtype=np.float64)
        _read_model_state_into(model, cc)
        _set_model_state(model, cc)
        engine.get_rates(surface=False, bottom=False)
        _record_fabm_output(engine, cc, sv_to_ref, output, out_index, nlev)
        out_index += 1
    else:
        assert cc_in is not None, "cc_in must be provided on non-first chunks"
        cc = cc_in.copy()

    for step in range(1, nt + 1):
        h_step = traj_h[step]
        nuh_step = traj_nuh[step]

        _set_environment(
            engine, model, nlev, traj_T[step], traj_S[step], traj_rho[step],
            h_step, traj_rad[step],
            u10=float(forcing_u10[step]) if forcing_u10 is not None else None,
            v10=float(forcing_v10[step]) if forcing_v10 is not None else None,
            yearday=float(forcing_yearday[step]) if forcing_yearday is not None else None,
        )

        for var in range(n_vars):
            _y[1:nlev + 1] = cc[var, :]
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
            cc[var, :] = _y[1:nlev + 1]

        _set_model_state(model, cc)
        bulk_rates = engine.get_rates(surface=False, bottom=False).copy()
        surf_rates = engine.get_rates(surface=True,  bottom=False).copy()
        bot_rates  = engine.get_rates(surface=False, bottom=True)
        cc += dt * bulk_rates
        cc[:, -1] += dt * (surf_rates[:, -1] - bulk_rates[:, -1])
        cc[:, 0]  += dt * (bot_rates[:, 0]   - bulk_rates[:, 0])

        is_output = (step % output_every == 0) or (
            force_final and step == nt and nt % output_every != 0
        )
        if is_output:
            _record_fabm_output(engine, cc, sv_to_ref, output, out_index, nlev)
            out_index += 1

    return cc, out_index


def integrate_fabm_from_trajectory(
    engine: FABMEngine,
    params: RuntimeParams,
    output: RuntimeOutput,
    traj_T: np.ndarray,
    traj_S: np.ndarray,
    traj_rho: np.ndarray,
    traj_h: np.ndarray,
    traj_nuh: np.ndarray,
    traj_rad: np.ndarray,
    forcing_u10: np.ndarray | None = None,
    forcing_v10: np.ndarray | None = None,
    forcing_yearday: np.ndarray | None = None,
) -> None:
    """Run pyfabm over every stored physics step, then fill reference outputs.

    The compiled Numba loop already completed. Trajectory arrays hold T, S,
    rho, h, nuh, rad at every step (shape ``(nt+1, nlev+1)``). This function
    walks those steps in Python, coupling pyfabm to stored physics without
    entering any Numba kernel.

    Reference profile and scalar outputs are written into
    ``output.reference_z_profiles`` and ``output.reference_scalars`` at the
    same output-slot indices as the physics outputs.

    Optional *forcing_u10* / *forcing_v10* (shape ``(nt+1,)``) provide surface
    wind components for FABM models that need ``wind_speed``. *forcing_yearday*
    provides the calendar day-of-year for models needing
    ``number_of_days_since_start_of_the_year``.
    """

    _, out_index = integrate_fabm_chunk(
        engine,
        params,
        output,
        traj_T,
        traj_S,
        traj_rho,
        traj_h,
        traj_nuh,
        traj_rad,
        cc_in=None,
        out_index_base=0,
        forcing_u10=forcing_u10,
        forcing_v10=forcing_v10,
        forcing_yearday=forcing_yearday,
        is_first_chunk=True,
    )
    _record_scalar_diagnostics(engine, output, out_index)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _count_state_variables(model: object) -> int:
    for attr in ("state_variables", "stateVariables"):
        variables = getattr(model, attr, None)
        if variables is not None:
            return len(list(variables))
    state = getattr(model, "state", None)
    if state is not None:
        arr = np.asarray(state)
        return arr.shape[0] if arr.ndim >= 1 else 1
    return 0


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
) -> None:
    T_col = np.ascontiguousarray(T[1:nlev + 1], dtype=np.float64)
    S_col = np.ascontiguousarray(S[1:nlev + 1], dtype=np.float64)
    rho_col = np.ascontiguousarray(rho[1:nlev + 1], dtype=np.float64)
    h_col = np.ascontiguousarray(h[1:nlev + 1], dtype=np.float64)
    rad_col = np.ascontiguousarray(rad[1:nlev + 1], dtype=np.float64)

    try:
        model.cell_thickness = h_col
    except AttributeError:
        pass

    _try_set(engine, "temperature", T_col)
    _try_set(engine, "practical_salinity", S_col)
    _try_set(engine, "density", rho_col)
    _try_set(engine, "downwelling_photosynthetic_radiative_flux", rad_col)

    surface_par = float(rad[nlev])
    _try_set_scalar(engine, "surface_downwelling_photosynthetic_radiative_flux", surface_par)

    if u10 is not None and v10 is not None:
        wspd = float(np.sqrt(u10 * u10 + v10 * v10))
        _try_set_scalar(engine, "wind_speed", wspd)

    if yearday is not None:
        _try_set_scalar(engine, "number_of_days_since_start_of_the_year", yearday)


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
    sv_to_ref: list,
    output: RuntimeOutput,
    slot: int,
    nlev: int,
) -> None:
    if slot >= output.nout:
        return

    for var_idx, arr in enumerate(sv_to_ref):
        if arr is None or var_idx >= cc.shape[0]:
            continue
        arr[slot, 1:nlev + 1] = cc[var_idx, :]

    diags = engine.diagnostics()
    for name, diag_val in diags.items():
        norm_name = name.replace("/", "_")
        arr = output.reference_z_profiles.get(norm_name)
        if arr is None:
            continue
        if isinstance(diag_val, np.ndarray) and diag_val.ndim == 1:
            n = min(diag_val.shape[0], nlev)
            arr[slot, 1:n + 1] = diag_val[:n]


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
