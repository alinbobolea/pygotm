# ruff: noqa: E501
"""
Internal-wave background mixing.

Implements GOTM Section 4.7.45 (internal_wave.F90) — imposes an eddy viscosity
and diffusivity characteristic of internal wave activity and shear instability
when the turbulence kinetic energy is below a threshold ``klimiw``.  Following
Kantha and Clayson (1994), when :math:`k < k_{\\mathrm{limiw}}` the mixing
coefficients are set to empirical values (Eq. 204):

.. math::

   \\nu_t = \\nu_t^{IW} + \\nu_t^{SI}, \\quad
   \\kappa_t = \\kappa_t^{IW} + \\kappa_t^{SI} \\comma

where the internal-wave background values are (Eq. 205):

.. math::

   \\nu_t^{IW} = 10^{-4}\\,\\mathrm{m^2\\,s^{-1}}, \\quad
   \\kappa_t^{IW} = 5 \\times 10^{-5}\\,\\mathrm{m^2\\,s^{-1}} \\comma

and the shear-instability parts depend on the gradient Richardson number
:math:`R_i = N^2 / M^2`:

.. math::

   \\nu_t^{SI} = \\kappa_t^{SI} = 0, \\qquad R_i > 0.7 \\quad (\\text{Eq. 206})

.. math::

   \\nu_t^{SI} = \\kappa_t^{SI} = 5 \\times 10^{-3}
       \\left(1 - \\left(\\frac{R_i}{0.7}\\right)^2\\right)^3,
       \\quad 0 < R_i < 0.7 \\quad (\\text{Eq. 207})

.. math::

   \\nu_t^{SI} = \\kappa_t^{SI} = 5 \\times 10^{-3},
       \\qquad R_i < 0 \\quad (\\text{Eq. 208})

All diffusivities are in :math:`\\mathrm{m^2\\,s^{-1}}`.

This model is activated by ``iw_model = 2``.  The contribution is added to
:math:`\\nu_t` and :math:`\\kappa_t` after the turbulence-closure update.

Authors (original Fortran): Karsten Bolding, Hans Burchard, Manuel Ruiz Villarreal.
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = ["InternalWaveWorkspace", "step_internal_wave", "step_internal_wave_single"]

_SHEAR_EPSILON = 1.0e-10


class InternalWaveWorkspace(ColumnWorkspace):
    """Workspace arrays for internal-wave mixing."""

    tke: np.ndarray
    num: np.ndarray
    nuh: np.ndarray
    NN: np.ndarray
    SS: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.nuh = make_column_array(nlev, n_cols=n_cols)
        self.NN = make_column_array(nlev, n_cols=n_cols)
        self.SS = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_internal_wave(
    nlev: int,
    iw_model: int,
    klimiw: float,
    rich_cr: float,
    numiw: float,
    nuhiw: float,
    numshear: float,
    tke: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
) -> None:
    r"""Apply the Kantha-Clayson internal-wave mixing fallback (single column)."""
    if iw_model == 2:
        rich2 = rich_cr * rich_cr

        for i in range(1, nlev):
            if tke[i] <= klimiw:
                rich = NN[i] / (SS[i] + _SHEAR_EPSILON)
                if rich < rich_cr:
                    if rich > 0.0:
                        pot = 1.0 - rich * rich / rich2
                        x = numshear * pot * pot * pot
                        num[i] = numiw + x
                        nuh[i] = nuhiw + x
                    else:
                        num[i] = numiw + numshear
                        nuh[i] = nuhiw + numshear
                else:
                    num[i] = numiw
                    nuh[i] = nuhiw


@numba.njit(parallel=True, cache=True)
def step_internal_wave(
    batch_size: int,
    nlev: int,
    iw_model: int,
    klimiw: float,
    rich_cr: float,
    numiw: float,
    nuhiw: float,
    numshear: float,
    tke: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
) -> None:
    r"""Apply the Kantha-Clayson internal-wave mixing fallback (batch)."""
    for b in numba.prange(batch_size):
        _step_internal_wave(
            nlev, iw_model, klimiw, rich_cr, numiw, nuhiw, numshear,
            tke[b], num[b], nuh[b], NN[b], SS[b],
        )


step_internal_wave_single = _step_internal_wave
