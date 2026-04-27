"""Helpers for populating Taichi fields from NumPy arrays and reading results."""

from __future__ import annotations

import numpy as np
import taichi as ti

__all__ = [
    "fill_field_from_array",
    "fill_field_scalar",
    "make_equidistant_h",
    "read_field_array",
]


def fill_field_from_array(field: ti.Field, arr: np.ndarray, col: int = 0) -> None:
    """Copy a 1-D NumPy array into a Taichi field."""

    data = field.to_numpy()
    if data.ndim == 1:
        data[:] = arr
    else:
        data[col, :] = arr
    field.from_numpy(data)


def fill_field_scalar(
    field: ti.Field,
    value: float,
    col: int = 0,
    idx: int = 0,
) -> None:
    """Write a scalar value into a Taichi field slot."""

    data = field.to_numpy()
    if data.ndim == 1:
        data[idx] = value
    else:
        data[col, idx] = value
    field.from_numpy(data)


def read_field_array(field: ti.Field, col: int = 0) -> np.ndarray:
    """Read one Taichi field column back into a NumPy array."""

    data = np.asarray(field.to_numpy())
    if data.ndim == 1:
        return data.copy()
    return np.asarray(data[col, :]).copy()


def make_equidistant_h(nlev: int, depth: float) -> np.ndarray:
    """Return an equidistant GOTM layer-thickness array h[0..nlev]."""

    h = np.zeros(nlev + 1)
    h[1:] = depth / nlev
    return h
