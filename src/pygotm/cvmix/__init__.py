"""
CVMix community vertical-mixing library interface.

Provides :class:`CVMixState` — the state container for the Community Vertical
Mixing (CVMix) parameterisations.  CVMix is an optional alternative to the
GOTM two-equation closures, intended for use in ocean general circulation
models that use CVMix as their mixing library.
"""

from pygotm.cvmix.gotm_cvmix import CVMixState

__all__ = ["CVMixState"]
