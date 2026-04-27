r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The Coriolis rotation \label{sec:coriolis}
!
! !INTERFACE:
!   subroutine coriolis(nlev,dt)
!
! !DESCRIPTION:
!  This subroutine carries out the Coriolis rotation by applying a
!  $2\times 2$ rotation matrix with the angle $f\Delta t$ on the
!  horizontal velocity vector $(U,V)$.
!
! !USES:
!   USE meanflow, only: u,v,cori
!   USE stokes_drift, only: usprof, vsprof
!
! !INPUT PARAMETERS:
!   integer, intent(in)                 :: nlev
!   REALTYPE, intent(in)                :: dt
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                   :: i
!   REALTYPE                  :: ua,omega,cosomega,sinomega
!   REALTYPE                  :: ul, vl
!
!-----------------------------------------------------------------------
!BOC
!
!   omega=cori*dt
!   cosomega=cos(omega)
!   sinomega=sin(omega)
!
!   do i=1,nlev
!!     KK-TODO: move calculation of Lagrangian velocities to a more
!!              central place.
!      ul = u(i) + usprof%data(i)
!      vl = v(i) + vsprof%data(i)
!
!      ua = ul
!      ul =  ul*cosomega + vl*sinomega
!      vl = -ua*sinomega + vl*cosomega
!
!!     KK-TODO: In GETM we distinguish between old and new Stokes drift.
!      u(i) = ul - usprof%data(i)
!      v(i) = vl - vsprof%data(i)
!   end do
!
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import math

import numpy as np
import taichi as ti

from pygotm.meanflow.meanflow import MeanflowState
from pygotm.taichi_typing import TemplateArg, ti_kernel

__all__ = [
    "CoriolisWorkspace",
    "coriolis",
    "step_coriolis",
]


class CoriolisWorkspace:
    """Minimal workspace for the Taichi coriolis kernel.

    Stores per-column scalars needed by :func:`step_coriolis` so they can be
    passed as Taichi fields (Taichi kernels cannot receive plain Python scalars
    for per-column values in the multi-column path).
    """

    def __init__(self, n_cols: int = 1) -> None:
        self.n_cols = n_cols
        # cosomega and sinomega are uniform across all columns
        # (same cori and dt) so we pass them as scalar kernel arguments.


@ti_kernel
def step_coriolis(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    cosomega: ti.f64,
    sinomega: ti.f64,
    u: TemplateArg,
    v: TemplateArg,
    usprof: TemplateArg,
    vsprof: TemplateArg,
):
    r"""Advance the Coriolis rotation for all columns.

    Applies a 2x2 rotation matrix with angle omega = cori*dt to the
    Lagrangian velocity vector (U + us, V + vs) for each layer, then
    removes the Stokes drift to recover the Eulerian velocity:

        ul = u(i) + usprof(i)
        vl = v(i) + vsprof(i)
        ul' =  ul*cos(omega) + vl*sin(omega)
        vl' = -ul*sin(omega) + vl*cos(omega)
        u(i) = ul' - usprof(i)
        v(i) = vl' - vsprof(i)

    Outer loop is parallel across columns (GPU threads).
    """
    for col in range(n_cols):  # parallel across columns
        for i in range(1, nlev + 1):  # serial over vertical layers
            # KK-TODO: move calculation of Lagrangian velocities to a more
            #          central place.
            ul = u[col, i] + usprof[col, i]
            vl = v[col, i] + vsprof[col, i]

            ua = ul
            ul = ul * cosomega + vl * sinomega
            vl = -ua * sinomega + vl * cosomega

            # KK-TODO: In GETM we distinguish between old and new Stokes drift.
            u[col, i] = ul - usprof[col, i]
            v[col, i] = vl - vsprof[col, i]


def coriolis(
    state: MeanflowState,
    nlev: int,
    dt: float,
    usprof: np.ndarray | None = None,
    vsprof: np.ndarray | None = None,
) -> None:
    """Apply the Coriolis rotation to horizontal velocities for one column.

    Implements coriolis.F90 as a pure-numpy single-column function.
    :func:`step_coriolis` is the multi-column GPU entry point.

    Parameters
    ----------
    state:
        MeanflowState carrying ``u``, ``v``, and ``cori``.  ``state.u`` and
        ``state.v`` are updated in-place for indices 1..nlev.
    nlev:
        Number of model layers.
    dt:
        Time step [s].
    usprof:
        Stokes drift profile in x at layer centres, shape ``(nlev+1,)``
        [m/s].  Index 0 unused; layers 1..nlev used.  Defaults to zeros.
    vsprof:
        Stokes drift profile in y at layer centres, shape ``(nlev+1,)``
        [m/s].  Defaults to zeros.
    """
    assert state.u is not None
    assert state.v is not None

    _usprof = usprof if usprof is not None else np.zeros(nlev + 1)
    _vsprof = vsprof if vsprof is not None else np.zeros(nlev + 1)

    omega = state.cori * dt
    cosomega = math.cos(omega)
    sinomega = math.sin(omega)

    u = state.u
    v = state.v

    for i in range(1, nlev + 1):
        # KK-TODO: move calculation of Lagrangian velocities to a more
        #          central place.
        ul = u[i] + _usprof[i]
        vl = v[i] + _vsprof[i]

        ua = ul
        ul = ul * cosomega + vl * sinomega
        vl = -ua * sinomega + vl * cosomega

        # KK-TODO: In GETM we distinguish between old and new Stokes drift.
        u[i] = ul - _usprof[i]
        v[i] = vl - _vsprof[i]
