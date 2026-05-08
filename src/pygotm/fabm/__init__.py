"""
Framework for Aquatic Biogeochemical Models (FABM) coupling.

Provides the :class:`FabmState` and :class:`FabmObservation` containers
and the input-variable infrastructure for coupling GOTM with FABM-compatible
biogeochemical models.  FABM is an external library; this module provides
the Python-side state containers and I/O adapters only.
"""

from pygotm.fabm.config import FABMConfig
from pygotm.fabm.engine import FABMEngine
from pygotm.fabm.gotm_fabm import FabmObservation, FabmState
from pygotm.fabm.gotm_fabm_input import FabmInputState, InputVariable
from pygotm.fabm.state import FABMStateBuffer

__all__ = [
    "FABMConfig",
    "FABMEngine",
    "FABMStateBuffer",
    "FabmInputState",
    "FabmObservation",
    "FabmState",
    "InputVariable",
]
