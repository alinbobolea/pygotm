"""State-buffer utilities for pyGOTM-owned FABM tracer arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "FABMStateBuffer",
    "FloatArray",
    "allocate_fabm_state",
    "fabm_state_variable_names",
]


@dataclass(slots=True)
class FABMStateBuffer:
    """pyGOTM-owned FABM state, rates, diagnostics, and light feedback arrays."""

    variable_names: tuple[str, ...]
    state: FloatArray
    rates: FloatArray
    bioshade: FloatArray
    diagnostics: dict[str, FloatArray | float] = field(default_factory=dict)

    def validate(self, nlev: int) -> None:
        """Raise if buffers violate the array contract used by the runtime."""

        expected = (len(self.variable_names), nlev + 1)
        for name, array, shape in (
            ("state", self.state, expected),
            ("rates", self.rates, expected),
            ("bioshade", self.bioshade, (nlev + 1,)),
        ):
            if array.dtype != np.float64:
                msg = f"{name} must have dtype float64, got {array.dtype}"
                raise TypeError(msg)
            if array.shape != shape:
                msg = f"{name} must have shape {shape}, got {array.shape}"
                raise ValueError(msg)
            if not array.flags.c_contiguous:
                msg = f"{name} must be C-contiguous"
                raise ValueError(msg)


def _variable_name(variable: object, index: int) -> str:
    for attr in ("name", "output_name", "long_name", "id"):
        value = getattr(variable, attr, None)
        if value:
            return str(value)
    return f"state_{index}"


def fabm_state_variable_names(model: Any) -> tuple[str, ...]:
    """Return stable state-variable names exposed by a pyfabm model."""

    variables = getattr(model, "state_variables", None)
    if variables is None:
        variables = getattr(model, "stateVariables", None)
    if variables is None:
        count = int(getattr(model, "state_variable_count", 0) or 0)
        return tuple(f"state_{index}" for index in range(count))
    return tuple(
        _variable_name(variable, index) for index, variable in enumerate(variables)
    )


def allocate_fabm_state(
    nlev: int,
    variable_names: tuple[str, ...],
) -> FABMStateBuffer:
    """Allocate contiguous pyGOTM-owned FABM buffers."""

    if nlev < 1:
        msg = f"nlev must be at least 1, got {nlev}"
        raise ValueError(msg)
    if not variable_names:
        msg = "FABM state variable list must not be empty"
        raise ValueError(msg)
    shape = (len(variable_names), nlev + 1)
    state = FABMStateBuffer(
        variable_names=variable_names,
        state=np.zeros(shape, dtype=np.float64),
        rates=np.zeros(shape, dtype=np.float64),
        bioshade=np.ones(nlev + 1, dtype=np.float64),
    )
    state.validate(nlev)
    return state
