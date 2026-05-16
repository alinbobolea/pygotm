# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: gotm_fabm_input
!
! !DESCRIPTION:
!  This module contains routines for initializing and reading one or more
!  data files containing observed profiles for FABM state variables.
!
! !REVISION HISTORY:
!  Original author(s): Jorn Bruggeman
!
!EOP
!-----------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pygotm.fabm.gotm_fabm import FabmState, register_observation
from pygotm.input.input import ProfileInput, ScalarInput, register_input

__all__ = [
    "FabmInputState",
    "InputVariable",
    "append_input",
    "configure_gotm_fabm_input",
    "fabm_input_create",
    "init_gotm_fabm_input",
]


@dataclass
class InputVariable:
    """Information on an observed FABM variable."""

    name: str
    scalar_input: ScalarInput | None = None
    profile_input: ProfileInput | None = None
    interior_id: Any | None = None
    horizontal_id: Any | None = None
    scalar_id: Any | None = None
    ncid: int = -1
    relax_tau: float = 1.0e15
    relax_tau_bot: float = 1.0e15
    relax_tau_surf: float = 1.0e15
    h_bot: float = 0.0
    h_surf: float = 0.0
    relax_tau_1d: np.ndarray | None = None


@dataclass
class FabmInputState:
    """Linked-list replacement for FABM input variables."""

    variables: list[InputVariable]

    def __init__(self) -> None:
        self.variables = []


def configure_gotm_fabm_input(
    state: FabmInputState, cfg: dict[str, Any] | None = None
) -> None:
    """Configure FABM input descriptors from a mapping."""

    if cfg is None:
        return
    for name, spec in cfg.items():
        if isinstance(spec, dict):
            fabm_input_create(state, name, spec)


def append_input(state: FabmInputState, input_variable: InputVariable) -> None:
    """Append an input variable to the current FABM input list."""

    state.variables.append(input_variable)


def fabm_input_create(
    state: FabmInputState,
    name: str,
    spec: dict[str, Any] | None = None,
    *,
    interior_id: Any | None = None,
    horizontal_id: Any | None = None,
    scalar_id: Any | None = None,
) -> InputVariable:
    """Create and append a FABM input descriptor."""

    raw = {} if spec is None else spec
    is_profile = interior_id is not None or raw.get("profile", False)
    variable = InputVariable(
        name=name,
        interior_id=interior_id,
        horizontal_id=horizontal_id,
        scalar_id=scalar_id,
        relax_tau=float(raw.get("relax_tau", 1.0e15)),
        relax_tau_bot=float(raw.get("relax_tau_bot", 1.0e15)),
        relax_tau_surf=float(raw.get("relax_tau_surf", 1.0e15)),
        h_bot=float(raw.get("thickness_bot", 0.0)),
        h_surf=float(raw.get("thickness_surf", 0.0)),
    )
    method = 2 if raw.get("method") == "file" else 0
    path = str(raw.get("file", ""))
    column = int(raw.get("column", 1))
    constant_value = float(raw.get("constant_value", 0.0))
    if is_profile:
        variable.profile_input = ProfileInput(
            name=name,
            method=method,
            path=path,
            index=column,
            constant_value=constant_value,
        )
    else:
        variable.scalar_input = ScalarInput(
            name=name,
            method=method,
            path=path,
            index=column,
            constant_value=constant_value,
        )
    append_input(state, variable)
    return variable


def init_gotm_fabm_input(
    state: FabmInputState,
    fabm: FabmState,
    nlev: int,
    h: np.ndarray,
) -> None:
    """Initialize input files and register observations with GOTM-FABM."""

    depth = float(np.sum(h[1 : nlev + 1]))
    for variable in state.variables:
        if variable.profile_input is not None:
            register_input(variable.profile_input)
            variable.relax_tau_1d = np.full(
                nlev + 1, variable.relax_tau, dtype=np.float64
            )
            db = 0.0
            ds = depth
            for k in range(1, nlev + 1):
                db += 0.5 * h[k]
                ds -= 0.5 * h[k]
                if db <= variable.h_bot:
                    variable.relax_tau_1d[k] = variable.relax_tau_bot
                if ds <= variable.h_surf:
                    variable.relax_tau_1d[k] = variable.relax_tau_surf
                db += 0.5 * h[k]
                ds -= 0.5 * h[k]
            assert variable.profile_input.data is not None
            register_observation(
                fabm,
                variable.interior_id,
                variable.profile_input.data,
                variable.relax_tau_1d,
            )
        elif variable.scalar_input is not None:
            register_input(variable.scalar_input)
            if variable.horizontal_id is not None:
                register_observation(
                    fabm,
                    variable.horizontal_id,
                    variable.scalar_input.value,
                    variable.relax_tau,
                )
            else:
                register_observation(
                    fabm, variable.scalar_id, variable.scalar_input.value
                )
