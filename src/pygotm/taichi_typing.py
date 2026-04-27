# ruff: noqa: UP047
"""Typing helpers for Taichi kernels on the current project stack.

Taichi 1.7.x expects runtime annotation objects such as ``ti.template()`` and
``ti.types.ndarray()`` on kernel parameters, while static type checkers reject
those forms in function signatures. This module provides mypy-friendly marker
aliases and typed decorator wrappers that rewrite those markers back to the
runtime Taichi annotations immediately before decoration.
"""

from collections.abc import Callable
from typing import Annotated, Any, TypeVar, cast, get_args

import taichi as ti

__all__ = [
    "NdarrayArg",
    "TemplateArg",
    "ti_func",
    "ti_kernel",
]

_TEMPLATE_MARKER = "taichi_template"
_NDARRAY_MARKER = "taichi_ndarray"
F = TypeVar("F", bound=Callable[..., object])

TemplateArg = Annotated[Any, _TEMPLATE_MARKER]
NdarrayArg = Annotated[Any, _NDARRAY_MARKER]


def _rewrite_annotations(fn: Callable[..., object]) -> None:
    annotations = dict(getattr(fn, "__annotations__", {}))
    for name, annotation in list(annotations.items()):
        metadata = get_args(annotation)[1:]
        if _TEMPLATE_MARKER in metadata:
            annotations[name] = ti.template()
        elif _NDARRAY_MARKER in metadata:
            annotations[name] = ti.types.ndarray()
    fn.__annotations__ = annotations


def ti_kernel(fn: F) -> F:
    """Decorate *fn* as a Taichi kernel while preserving static typing."""

    _rewrite_annotations(fn)
    return cast(F, ti.kernel(fn))


def ti_func(fn: F) -> F:
    """Decorate *fn* as a Taichi function while preserving static typing."""

    return cast(F, ti.func(fn))
