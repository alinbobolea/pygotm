"""Backwards-compatible facade for the compiled runtime builder.

The implementation now lives in focused sibling modules (runtime_core,
runtime_params_from_run, runtime_forcing_from_run, runtime_from_run,
runtime_dataset). This module re-exports the full historical surface so every
existing ``from pygotm.gotm.runtime_builder import ...`` keeps resolving to the
identical objects.
"""

from pygotm.gotm.runtime_core import (
    RuntimeBundle,
    RuntimePhaseTimings,
    TimeLoopRunner,
    UnsupportedConfigurationError,
    build_runtime,
    build_runtime_forcing,
    build_runtime_output,
    build_runtime_params,
    build_runtime_state,
    build_runtime_work,
    select_time_loop,
)
from pygotm.gotm.runtime_dataset import runtime_output_to_dataset
from pygotm.gotm.runtime_forcing_from_run import (
    _copy_absolute_salinity_from_practical,
    _make_salinity_conversion_cache,
    build_runtime_forcing_from_run,
)
from pygotm.gotm.runtime_from_run import build_runtime_from_run

__all__ = [
    "RuntimeBundle",
    "RuntimePhaseTimings",
    "TimeLoopRunner",
    "UnsupportedConfigurationError",
    "_copy_absolute_salinity_from_practical",
    "_make_salinity_conversion_cache",
    "build_runtime",
    "build_runtime_forcing",
    "build_runtime_forcing_from_run",
    "build_runtime_from_run",
    "build_runtime_output",
    "build_runtime_params",
    "build_runtime_state",
    "build_runtime_work",
    "runtime_output_to_dataset",
    "select_time_loop",
]
