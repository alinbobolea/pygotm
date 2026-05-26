"""Shared scalar utilities for ice thermodynamics kernels."""

import math

import numba
import numpy as np

from pygotm.icethm.constants import FREEZE_SLOPE, MU_TS
from pygotm.icethm.state import IceState


@numba.njit(cache=True)
def freezing_temperature(S: float) -> float:
    """Return the linear seawater freezing point used by GOTM simple ice."""

    return -FREEZE_SLOPE * S


@numba.njit(cache=True)
def freezing_temperature_winton(S: float) -> float:
    """Return Winton's linear seawater freezing point, ``Tf = -m S``."""

    return -MU_TS * S


@numba.njit(cache=True)
def clamp01(value: float) -> float:
    """Clamp a scalar into the closed unit interval."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@numba.njit(cache=True)
def safe_exp(value: float) -> float:
    """Exponentiate after clipping to avoid overflow in pathological inputs."""

    if value < -700.0:
        return 0.0
    if value > 700.0:
        return math.exp(700.0)
    return math.exp(value)


def require_ice_state(state: IceState) -> None:
    """Validate that an :class:`IceState` has Numba-compatible scalar arrays."""

    for name, value in state.__dict__.items():
        if not isinstance(value, np.ndarray):
            msg = f"{name} must be a NumPy array"
            raise TypeError(msg)
        if value.shape != (1,):
            msg = f"{name} must have shape (1,), got {value.shape}"
            raise ValueError(msg)
        if name == "ice_cover":
            if value.dtype != np.int32:
                msg = "ice_cover must use np.int32"
                raise TypeError(msg)
        elif value.dtype != np.float64:
            msg = f"{name} must use np.float64"
            raise TypeError(msg)
