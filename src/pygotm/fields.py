"""Shared abstractions for GOTM-style Taichi field collections.

The translated GOTM kernels store every vertical profile with an extra
sentinel slot, matching the Fortran convention of arrays declared on
``0:nlev``. Single-column fields therefore have shape ``(nlev + 1,)`` and
multi-column fields have shape ``(n_cols, nlev + 1)``.

Field allocation must happen after :func:`taichi.init` has configured the
runtime. This module deliberately keeps allocation explicit so each physics
module can own its state and work arrays without falling back to hidden global
fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import taichi as ti

__all__ = [
    "ColumnLayout",
    "TaichiFieldCollection",
]


type FieldShape = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ColumnLayout:
    """Describe the storage layout for a translated GOTM column state."""

    nlev: int
    n_cols: int | None = None

    def __post_init__(self) -> None:
        if self.nlev < 0:
            msg = "nlev must be non-negative"
            raise ValueError(msg)
        if self.n_cols is not None and self.n_cols < 1:
            msg = "n_cols must be at least 1 when provided"
            raise ValueError(msg)

    @property
    def storage_levels(self) -> int:
        """Return the allocated vertical extent, including the sentinel slot."""

        return self.nlev + 1

    @property
    def is_multi_column(self) -> bool:
        """Return whether the layout stores multiple independent columns."""

        return self.n_cols is not None

    @property
    def shape(self) -> FieldShape:
        """Return the Taichi field shape for this column layout."""

        if self.n_cols is None:
            return (self.storage_levels,)
        return (self.n_cols, self.storage_levels)


class TaichiFieldCollection:
    """Base helper for physics modules that own related Taichi fields.

    Subclasses usually keep one instance per translated module, allocate the
    required fields in their ``init()`` routine, and then pass those fields into
    kernels explicitly.
    """

    def __init__(self, layout: ColumnLayout) -> None:
        self.layout = layout
        self._fields: dict[str, ti.Field] = {}

    @property
    def shape(self) -> FieldShape:
        """Return the common Taichi shape shared by all registered fields."""

        return self.layout.shape

    def allocate(self, name: str, *, dtype: object = ti.f64) -> ti.Field:
        """Allocate and register a Taichi field under ``name``."""

        if not name:
            msg = "field name must be a non-empty string"
            raise ValueError(msg)
        if name in self._fields:
            msg = f"field {name!r} is already allocated"
            raise ValueError(msg)

        field = ti.field(dtype=dtype, shape=self.shape)
        self._fields[name] = field
        setattr(self, name, field)
        return field

    def allocate_many(
        self,
        names: Iterable[str],
        *,
        dtype: object = ti.f64,
    ) -> None:
        """Allocate several fields with the same dtype and shared shape."""

        for name in names:
            self.allocate(name, dtype=dtype)

    def get(self, name: str) -> ti.Field:
        """Return a previously allocated field by name."""

        try:
            return self._fields[name]
        except KeyError as exc:
            msg = f"unknown field {name!r}"
            raise KeyError(msg) from exc

    def items(self) -> Iterator[tuple[str, ti.Field]]:
        """Iterate over registered ``(name, field)`` pairs."""

        return iter(self._fields.items())

    def __getattr__(self, name: str) -> ti.Field:
        """Resolve dynamically allocated fields as typed attributes."""

        try:
            return self._fields[name]
        except KeyError as exc:
            msg = f"{type(self).__name__!s} has no attribute {name!r}"
            raise AttributeError(msg) from exc

    def names(self) -> tuple[str, ...]:
        """Return the currently registered field names."""

        return tuple(self._fields)
