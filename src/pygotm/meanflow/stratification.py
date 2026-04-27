# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Calculation of the stratification\label{sec:stratification}
!
! !INTERFACE:
!   subroutine stratification(nlev)
!
! !DESCRIPTION:
! This routine computes the mean potential density, $\mean{\rho}$, the mean
! potential buoyancy, $B$, defined in \eq{DefBuoyancy}, and the mean buoyancy
! frequency,
!  \begin{equation}
!    \label{DefBuoyancyFrequency}
!     N^2 = - \dfrac{g}{\rho_0} \partder{\rho}{z} = \partder{B}{z}
!     \comma
!  \end{equation}
! which is based on potential density or buoyancy such that for $N^2=0$, the entropy
! is constant in the whole water column and mixing does not work against buoyancy
! forces. If GOTM used as a turbulence library in your own three-dimensional model,
! you have to insure that the $N^2$ computed by you, and passed to the turbulence
! routines in GOTM, is consistent with the concept of potential density and your
! equation of state.
!
! The mean potential density is evaluated from the equation of state, \eq{DefEOS},
! according to
!  \begin{equation}
!    \label{DefPotentialDensity}
!     \mean{\rho} = \hat{\rho} (\Theta,S,P_R)
!     \comma
!  \end{equation}
!  where $\Theta$ denotes the mean potential temperature, $S$ the mean salinity
!  and $P_R$ the mean reference pressure. The buoyancy frequency defined in
! \eq{DefBuoyancyFrequency} can be decomposed into contributions due to
!  potential temperature and salinity stratification,
!  \begin{equation}
!    \label{NDecompostionA}
!     N^2 = N_\Theta^2 + N_S^2
!     \comma
!  \end{equation}
!  where we introduced the quantities
!  \begin{equation}
!    \label{NNT}
!     N_\Theta^2  = - \dfrac{g}{\rho_0} \partder{\rho}{z} \Big|_{S}
!                 = g \alpha(\Theta,S,P_R) \partder{\Theta}{z}
!     \comma
!  \end{equation}
!  with the thermal expansion coefficient defined in \eq{eosAlpha}, and
!  \begin{equation}
!    \label{NNS}
!     N_S^2  = - \dfrac{g}{\rho_0} \partder{\rho}{z} \Big|_{\Theta}
!                 = - g \beta(\Theta,S,P_R) \partder{S}{z}
!  \comma
!  \end{equation}
!  with the saline contraction coefficient defined in \eq{eosBeta}. It is important
!  to note that in the actual code the reference pressure, $P_R$, has been replaced by
!  the (approximate) hydrostatic pressure. Only if this dependence is replaced by
!  the constant reference pressure at the surface in the equation of state,
!  see \sect{sec:eqstate}, the model is truely based on potential temperature and density.
!  Otherwise,  the model is based on \emph{in-situ} quantities.
!
!  Alternatively to the procedure outlined above, depending on the values of the
!  parameter {\tt buoy\_method}, the buoyancy may be calculated by means of the
!  transport equation \eq{bEq}. This equation then replaces the computation of $\Theta$
!  and $S$ and is only recommended for idealized studies.
!
! !USES:
!   use density,    only: alpha,beta
!   use density,    only: rho0,rho_p
!   use meanflow,   only: h,T,S
!   use meanflow,   only: buoy
!   use meanflow,   only: NN,NNT,NNS
!   use meanflow,   only: gravity
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!  number of vertical layers
!   integer,  intent(in)                :: nlev
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
"""

import numpy as np
import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.meanflow.meanflow import MeanflowState
from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.util.density import DensityState

__all__ = [
    "StratificationWorkspace",
    "step_stratification",
    "stratification",
]


class StratificationWorkspace(TaichiFieldCollection):
    """Taichi fields for the stratification kernel."""

    h: ti.Field
    T: ti.Field
    S: ti.Field
    alpha: ti.Field
    beta: ti.Field
    NN: ti.Field
    NNT: ti.Field
    NNS: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("h", "T", "S", "alpha", "beta"))
        self.allocate_many(("NN", "NNT", "NNS"))


@ti_kernel
def step_stratification(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    gravity: ti.f64,
    h: TemplateArg,
    T: TemplateArg,
    S: TemplateArg,
    alpha: TemplateArg,
    beta: TemplateArg,
    NN: TemplateArg,
    NNT: TemplateArg,
    NNS: TemplateArg,
):
    """Compute N², NNT, and NNS at layer interfaces for all columns."""

    for col in range(n_cols):
        for k in range(1, nlev):
            idz = 2.0 / (h[col, k] + h[col, k + 1])
            dT = T[col, k + 1] - T[col, k]
            dS = S[col, k + 1] - S[col, k]
            NNT[col, k] = alpha[col, k] * gravity * dT * idz
            NNS[col, k] = -beta[col, k] * gravity * dS * idz
            NN[col, k] = NNT[col, k] + NNS[col, k]

        NN[col, 0] = 0.0
        NN[col, nlev] = 0.0
        NNT[col, 0] = 0.0
        NNT[col, nlev] = 0.0
        NNS[col, 0] = 0.0
        NNS[col, nlev] = 0.0


def stratification(
    state: MeanflowState,
    density_state: DensityState,
    nlev: int,
) -> None:
    r"""Compute buoyancy frequency squared (N²) at layer interfaces.

    Updates ``state.NN``, ``state.NNT``, and ``state.NNS`` in-place.

    The buoyancy frequency is decomposed into temperature and salinity
    contributions:

        N² = NNT + NNS

    where

        NNT = g * α * ∂Θ/∂z
        NNS = -g * β * ∂S/∂z

    using the centred finite-difference approximation

        ∂Θ/∂z ≈ 2*(Θ_{k+1} - Θ_k) / (h_k + h_{k+1})

    Boundary values at indices 0 and nlev are set to zero (no buoyancy
    flux through the seabed or surface).

    Parameters
    ----------
    state:
        MeanflowState with h, T, S, NN, NNT, NNS, gravity.
        All arrays have shape (nlev+1,); index 0 = seabed, nlev = surface.
    density_state:
        DensityState with alpha and beta interface arrays, shape (nlev+1,).
    nlev:
        Number of model layers.
    """
    assert state.h is not None
    assert state.T is not None
    assert state.S is not None
    assert state.NN is not None
    assert state.NNT is not None
    assert state.NNS is not None
    # state.buoy is intentionally not asserted here — see note below.
    assert density_state.alpha is not None
    assert density_state.beta is not None

    h = state.h
    T = state.T
    S = state.S
    alpha = density_state.alpha
    beta = density_state.beta
    g = state.gravity

    # Interior interfaces: i = 1 .. nlev-1
    # Matches Fortran: idz(1:nlev-1) = 2/(h(1:nlev-1)+h(2:nlev))
    i = np.arange(1, nlev)

    idz = 2.0 / (h[i] + h[i + 1])
    dT = T[i + 1] - T[i]
    dS = S[i + 1] - S[i]

    state.NNT[i] = alpha[i] * g * dT * idz
    state.NNS[i] = -beta[i] * g * dS * idz
    state.NN[i] = state.NNT[i] + state.NNS[i]

    # Boundary values: zero flux at seabed (0) and surface (nlev)
    state.NNT[0] = 0.0
    state.NNT[nlev] = 0.0
    state.NNS[0] = 0.0
    state.NNS[nlev] = 0.0
    state.NN[0] = 0.0
    state.NN[nlev] = 0.0

    # NOTE: state.buoy (mean potential buoyancy) is not updated here.
    # In the standard T/S-based path (buoy_method=0), buoy is computed by
    # do_density() in density.py from the density perturbation rho_p.
    # The buoy transport equation path (buoy_method=1) is not yet implemented.
