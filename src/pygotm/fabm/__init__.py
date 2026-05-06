"""
Framework for Aquatic Biogeochemical Models (FABM) coupling.

Provides the :class:`FabmState` and :class:`FabmObservation` containers
and the input-variable infrastructure for coupling GOTM with FABM-compatible
biogeochemical models.  FABM is an external library; this module provides
the Python-side state containers and I/O adapters only.
"""

from pygotm.fabm.gotm_fabm import FabmObservation, FabmState
from pygotm.fabm.gotm_fabm_input import FabmInputState, InputVariable

__all__ = [
    "FabmInputState",
    "FabmObservation",
    "FabmState",
    "InputVariable",
]
