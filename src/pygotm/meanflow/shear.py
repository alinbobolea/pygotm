r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Calculation of the vertical shear \label{sec:shear}
!
! !INTERFACE:
!   subroutine shear(nlev,cnpar)
!
! !DESCRIPTION:
!%  The (square of the) shear frequency is defined as
!% \begin{equation}
!%   \label{MSquared}
!%    M^2 = \left( \partder{U}{z} \right)^2 +
!%          \left( \partder{V}{z} \right)^2
!%    \point
!% \end{equation}
!% It is an important parameter in almost all turbulence models.
!% The $U$- and $V$-contributions to $M^2$ are computed using a new scheme
!% which guarantees conservation of kinetic energy for the convertion
!% from mean to turbulent kinetic energy, see \cite{Burchard2002}. With this method,
!% the discretisation of the $U$-contribution can be written as
!% \begin{equation}
!%   \label{shearsquared}
!%    \left( \partder{U}{z} \right)^2 \approx \frac{(\bar U_{j+1}-\bar U_j)
!%    (\tilde U_{j+1}-\tilde U_j)}{(z_{j+1}-z_j)^2}
!% \end{equation}
!% where $\tilde U_j=\frac12(\hat U_j+U_j)$. The $V$-contribution is computed analogously.
!% The shear obtained from \eq{shearsquared}
!% plus the $V$-contribution is then used for the computation of the turbulence
!% shear production, see equation \eq{computeP}.
!
! The (square of the) shear frequency is defined as
! \begin{equation}
!   \label{MSquared}
!    M^2 = \left( \partder{U}{z} \right)^2 +
!          \left( \partder{V}{z} \right)^2
!    \point
! \end{equation}
! It is an important parameter in almost all turbulence models.
! The $U$- and $V$-contributions to $M^2$ are computed using a new scheme
! which guarantees conservation of kinetic energy for the conversion
! from mean to turbulent kinetic energy, see \cite{Burchard2002}.
! The shear is calculated by dividing the energy-consistent
! form of the shear production (see equation (14) by \cite{Burchard2002},
! but note the typo in that equation)
! by the eddy viscosity. The correct form of the right hand side of
! equation (14) of
! \cite{Burchard2002} should be:
! \begin{equation}
!   \label{mean_kinetic_energy_dissipation}
! \begin{array}{rcl}
! \displaystyle
!    \left(D_{kin} \right)_j & = &
! \displaystyle
! \phantom{+}   \frac{\nu_{j+1/2}}{2}
! \frac{\sigma \left(\hat U_{j+1}-\hat U_j\right)\left(\hat U_{j+1}-U_j\right)+
! (1-\sigma)\left(U_{j+1}-U_j\right)\left(U_{j+1}-\hat U_j\right)}
! {(z_{j+1/2}-z_{j-1/2})(z_{j+1}-z_j)} \\ \\
! &&
! \displaystyle
! +
!  \frac{\nu_{j-1/2}}{2}\frac{\sigma \left(\hat U_{j}-\hat U_{j-1}\right)
! \left(U_{j}-\hat U_{j-1}\right)+
! (1-\sigma)\left(U_{j}-U_{j-1}\right)\left(\hat U_{j}-U_{j-1}\right)}
! {(z_{j+3/2}-z_{j+1/2})(z_{j+1}-z_j)} \\ \\
! & = &
! \displaystyle
! P_{j+1/2}^l + P_{j-1/2}^u,
! \end{array}
! \end{equation}
! with the mean kinetic energy dissipation, $\left(D_{kin} \right)_j$.
! The two terms on the right hand side are the contribution
! of energy dissipation from below the interface at $j+1/2$ and the
! contribution from above the interface at $j-1/2$.
! With (\ref{mean_kinetic_energy_dissipation}),
! an energy-conserving discretisation of the shear production at $j+1/2$ should
! be
! \begin{equation}
! P_{j+1/2} = P_{j+1/2}^l + P_{j+1/2}^u,
! \end{equation}
! such that a consistent discretisation of the square of the shear in
! $x$-direction should be
! \begin{equation}
!   \label{shearsquared}
! \begin{array}{rcl}
! \displaystyle
!    \left( \partder{U}{z} \right)^2 & \approx &
! \displaystyle
! \frac{P_{j+1/2}}{\nu_{j+1/2}} \\ \\
! &=&
! \displaystyle
! \phantom{+}   \frac12\frac{\sigma \left(\hat U_{j+1}-\hat U_j\right)
! \left(\hat U_{j+1}-U_j\right)+
! (1-\sigma)\left(U_{j+1}-U_j\right)\left(U_{j+1}-\hat U_j\right)}
! {(z_{j+1/2}-z_{j-1/2})(z_{j+1}-z_j)} \\ \\
! &&
! \displaystyle
! +
!  \frac12\frac{\sigma \left(\hat U_{j+1}-\hat U_j\right)
! \left(U_{j+1}-\hat U_j\right)+
! (1-\sigma)\left(U_{j+1}-U_j\right)\left(\hat U_{j+1}-U_j\right)}
! {(z_{j+3/2}-z_{j+1/2})(z_{j+1}-z_j)}.
! \end{array}
! \end{equation}
! The $V$-contribution is computed analogously.
! The shear obtained from \eq{shearsquared}
! plus the $V$-contribution is then used for the computation of the turbulence
! shear production, see equation \eq{computeP}.
!
! !USES:
!   use meanflow,   only: h,u,v,uo,vo
!   use meanflow,   only: SS,SSU,SSV
!   use meanflow,     only: SSCSTK, SSSTK
!   use stokes_drift, only: dusdz, dvsdz
!
! !INPUT PARAMETERS:
!
!  number of vertical layers
!   integer,  intent(in)                :: nlev
!
!  numerical "implicitness" parameter
!   REALTYPE, intent(in)                :: cnpar
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
!BOC
!  Discretisation of vertical shear squared according to Burchard (2002)
!  in order to guarantee conservation of kinetic energy when transformed
!  from mean kinetic energy to turbulent kinetic energy.
"""

from __future__ import annotations

import numpy as np

from pygotm.meanflow.meanflow import MeanflowState

__all__ = ["shear"]


def shear(
    state: MeanflowState,
    nlev: int,
    cnpar: float,
    dusdz: np.ndarray | None = None,
    dvsdz: np.ndarray | None = None,
) -> None:
    """Compute the shear-frequency squared (M²) at layer interfaces.

    Updates ``state.SS``, ``state.SSU``, ``state.SSV``, ``state.SSCSTK``,
    and ``state.SSSTK`` in-place using the energy-conserving Burchard (2002)
    discretisation.

    Parameters
    ----------
    state:
        MeanflowState with h, u, v, uo, vo, SS, SSU, SSV, SSCSTK, SSSTK.
        All arrays have shape (nlev+1,); index 0 = seabed, nlev = surface.
    nlev:
        Number of model layers.
    cnpar:
        Numerical implicitness parameter (0 = explicit, 1 = fully implicit;
        0.5 for Crank-Nicolson). Controls weighting between old and new
        time-level velocity differences in the shear production formula.
    dusdz:
        Stokes drift shear in x, shape (nlev+1,) [s⁻¹]. Defaults to zeros.
    dvsdz:
        Stokes drift shear in y, shape (nlev+1,) [s⁻¹]. Defaults to zeros.
    """
    assert state.h is not None
    assert state.u is not None
    assert state.v is not None
    assert state.uo is not None
    assert state.vo is not None
    assert state.SS is not None
    assert state.SSU is not None
    assert state.SSV is not None
    assert state.SSCSTK is not None
    assert state.SSSTK is not None

    h = state.h
    u = state.u
    v = state.v
    uo = state.uo
    vo = state.vo

    n = nlev + 1
    _dusdz = dusdz if dusdz is not None else np.zeros(n)
    _dvsdz = dvsdz if dvsdz is not None else np.zeros(n)

    # Vectorised over interior interfaces i = 1 .. nlev-1
    i = np.arange(1, nlev)  # shape (nlev-1,)

    h_mid = 0.5 * (h[i + 1] + h[i])  # (z_{j+1} - z_j) in Burchard notation

    # Energy-conserving U-shear squared (Burchard 2002, eq. 14 corrected)
    num_a_u = (
        cnpar * (u[i + 1] - u[i]) * (u[i + 1] - uo[i])
        + (1.0 - cnpar) * (uo[i + 1] - uo[i]) * (uo[i + 1] - u[i])
    )
    num_b_u = (
        cnpar * (u[i + 1] - u[i]) * (uo[i + 1] - u[i])
        + (1.0 - cnpar) * (uo[i + 1] - uo[i]) * (u[i + 1] - uo[i])
    )
    state.SSU[i] = 0.5 * (num_a_u / h_mid / h[i] + num_b_u / h_mid / h[i + 1])

    # Energy-conserving V-shear squared
    num_a_v = (
        cnpar * (v[i + 1] - v[i]) * (v[i + 1] - vo[i])
        + (1.0 - cnpar) * (vo[i + 1] - vo[i]) * (vo[i + 1] - v[i])
    )
    num_b_v = (
        cnpar * (v[i + 1] - v[i]) * (vo[i + 1] - v[i])
        + (1.0 - cnpar) * (vo[i + 1] - vo[i]) * (v[i + 1] - vo[i])
    )
    state.SSV[i] = 0.5 * (num_a_v / h_mid / h[i] + num_b_v / h_mid / h[i + 1])

    state.SS[i] = state.SSU[i] + state.SSV[i]

    # Stokes-Eulerian cross-shear
    state.SSCSTK[i] = (
        _dusdz[i] * (u[i + 1] - u[i]) / h_mid
        + _dvsdz[i] * (v[i + 1] - v[i]) / h_mid
    )

    # Stokes shear squared
    state.SSSTK[i] = _dusdz[i] ** 2 + _dvsdz[i] ** 2

    # Boundary fill: copy nearest interior value to ghost/boundary cells
    state.SSU[0] = state.SSU[1]
    state.SSU[nlev] = state.SSU[nlev - 1]

    state.SSV[0] = state.SSV[1]
    state.SSV[nlev] = state.SSV[nlev - 1]

    state.SS[0] = state.SS[1]
    state.SS[nlev] = state.SS[nlev - 1]

    state.SSCSTK[0] = state.SSCSTK[1]
    state.SSCSTK[nlev] = state.SSCSTK[nlev - 1]

    state.SSSTK[0] = state.SSSTK[1]
    state.SSSTK[nlev] = state.SSSTK[nlev - 1]
