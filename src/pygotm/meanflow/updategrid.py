r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The vertical grid \label{sec:updategrid}
!
! !INTERFACE:
!   subroutine updategrid(nlev,dt,zeta)
!
! !DESCRIPTION:
!  This subroutine calculates for each time step new layer thicknesses
!  in order to fit them to the changing water depth.
!  Three different grids can be specified:
!  \begin{enumerate}
!  \item Equidistant grid with possible zooming towards surface and bottom.
!  The number of layers, {\tt nlev}, and the zooming factors,
!  {\tt ddu}=$d_u$ and  {\tt ddl}=$d_l$,
!  are specified in {\tt gotm.yaml}.
!  Zooming is applied according to the formula
!  \begin{equation}\label{formula_Antoine}
!    \gamma_i = \frac{\mbox{tanh}\left( (d_l+d_u)\frac{i}{M}-d_l\right)
!    +\mbox{tanh}(d_l)}{\mbox{tanh}(d_l)+\mbox{tanh}(d_u)}-1
!   \point
!  \end{equation}
!  with $\gamma_i$ being the non-dimensional vertical position of the
!  $i$-th interface in \eq{grid}.
!
!  From this formula, the following grids are constructed:
!  \begin{itemize}
!    \item $d_l=d_u=0$ results in equidistant discretisations.
!    \item $d_l>0, d_u=0$ results in zooming near the bottom.
!    \item $d_l=0, d_u>0$ results in zooming near the surface.
!    \item $d_l>0, d_u>0$ results in double zooming nea both,
!          the surface and the bottom.
!  \end{itemize}
!
!  \item Sigma-layers. The fraction that every layer occupies is
!  read-in from file, see {\tt gotm.yaml}.
!  \item Cartesian layers. The height of every layer is read in from file,
!  see {\tt gotm.yaml}.
!  This method is not recommended when a varying sea surface is considered.
!  \end{enumerate}
!
!  Furthermore, vertical velocity profiles are calculated here, if
!  {\tt w\_adv\_method} is 1 or 2, which has to be chosen in the
!  {\tt w\_advspec} in {\tt gotm.yaml}. The profiles of vertical
!  velocity are determined by two values,
!  the height of maximum absolute value of vertical velocity, {\tt w\_height},
!  and the vertical velocity at this height, {\tt w\_adv}. From {\tt w\_height},
!  the vertical velocity is linearly decreasing towards the surface and
!  the bottom, where its value is zero.
!
! !USES:
!   use meanflow,     only: grid_ready
!   use meanflow,     only: depth0,depth
!   use meanflow,     only: ga,z,zi,h,ho,ddu,ddl,grid_method
!   use meanflow,     only: grid_file,w
!   use observations, only: w_adv_input,w_height_input
!
! !INPUT PARAMETERS:
!   integer, intent(in)                 :: nlev
!   REALTYPE, intent(in)                :: dt,zeta
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path

import numba
import numpy as np

from pygotm.meanflow.meanflow import MeanflowState

__all__ = ["step_updategrid_single", "updategrid"]


@numba.njit(cache=True)
def step_updategrid_single(
    nlev: int,
    depth0: float,
    zeta: float,
    grid_method: int,
    ga: np.ndarray,
    h: np.ndarray,
    ho: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
) -> None:
    """Recompute layer thicknesses and interface depths for a new sea-surface elevation.

    Only handles the per-step update (after grid_ready=True). Grid initialisation
    (building ga from ddu/ddl) must have already run in Python before this kernel
    is called from the compiled time loop.
    """
    current_depth = depth0 + zeta
    if grid_method == 1:
        for i in range(1, nlev + 1):
            ho[i] = h[i]
            h[i] = ga[i] * current_depth
    elif grid_method == 2:
        for i in range(1, nlev + 1):
            ho[i] = h[i]
    else:  # methods 0 and 3 (equidistant/zoomed sigma)
        for i in range(1, nlev + 1):
            ho[i] = h[i]
            h[i] = (ga[i] - ga[i - 1]) * current_depth
    zi[0] = -depth0
    for i in range(1, nlev + 1):
        zi[i] = zi[i - 1] + h[i]
        z[i] = zi[i - 1] + 0.5 * h[i]


def updategrid(
    state: MeanflowState,
    nlev: int,
    dt: float,  # kept for API parity with Fortran; not used in current implementation
    zeta: float,
) -> None:
    """Update vertical grid layer thicknesses and coordinates each time step.

    ! !DESCRIPTION:
    !  This subroutine calculates for each time step new layer thicknesses
    !  in order to fit them to the changing water depth.
    !
    ! grid_method values:
    !   0 — equidistant/zoomed sigma coordinates (Antoine Garapon zooming)
    !   1 — external sigma fractions read from grid_file (surface-first order)
    !   2 — external Cartesian layer thicknesses read from grid_file
    !   3 — adaptive sigma grid (same zooming as 0 but ga offset to [-1, 0])
    !
    ! Array indexing follows GOTM convention: index 0 = seabed, index nlev = surface.

    Parameters
    ----------
    state:
        MeanflowState instance with grid arrays allocated by post_init_meanflow.
    nlev:
        Number of model layers.
    dt:
        Time step [s] — present for API compatibility; not used here.
    zeta:
        Sea surface elevation [m]; positive upward from mean sea level.
    """
    assert state.ga is not None
    assert state.h is not None
    assert state.ho is not None
    assert state.z is not None
    assert state.zi is not None

    ga = state.ga
    h = state.h
    ho = state.ho
    z = state.z
    zi = state.zi

    # BOC
    if not state.grid_ready:  # Build up dimensionless grid (0 <= ga <= 1)
        method = state.grid_method

        if method == 0:
            # Equidistant grid with possible zooming to surface and bottom
            if state.ddu <= 0 and state.ddl <= 0:
                for i in range(1, nlev + 1):
                    ga[i] = ga[i - 1] + 1.0 / nlev
            else:
                # Zooming routine from Antoine Garapon, ICCH, DK
                for i in range(1, nlev + 1):
                    ga[i] = (
                        math.tanh((state.ddl + state.ddu) * i / nlev - state.ddl)
                        + math.tanh(state.ddl)
                    )
                    ga[i] /= math.tanh(state.ddl) + math.tanh(state.ddu)

            state.depth = state.depth0 + zeta
            for i in range(1, nlev + 1):
                h[i] = (ga[i] - ga[i - 1]) * state.depth

        elif method == 1:
            # Sigma — the fraction each layer occupies is read from file
            _read_sigma_grid_file(state, nlev, ga)

        elif method == 2:
            # Cartesian — layer thickness is read from file
            _read_cartesian_grid_file(state, nlev, h)

        elif method == 3:
            # Adaptive grid (same zooming as method 0, ga offset to [-1, 0])
            ga[0] = -1.0
            if state.ddu <= 0 and state.ddl <= 0:
                for i in range(1, nlev + 1):
                    ga[i] = ga[i - 1] + 1.0 / float(nlev)
            else:
                # Zooming from Antoine Garapon, ICCH, DK
                for i in range(1, nlev + 1):
                    ga[i] = (
                        math.tanh((state.ddl + state.ddu) * i / nlev - state.ddl)
                        + math.tanh(state.ddl)
                    )
                    ga[i] = ga[i] / (math.tanh(state.ddl) + math.tanh(state.ddu)) - 1.0

            state.depth = state.depth0 + zeta
            for i in range(1, nlev + 1):
                h[i] = (ga[i] - ga[i - 1]) * state.depth

        else:
            raise ValueError(
                f"updategrid: No valid grid_method specified: {state.grid_method}"
            )

        state.grid_ready = True  # Grid is now initialised!

    state.depth = state.depth0 + zeta

    method = state.grid_method
    if method == 0:
        # ho saves the layer thicknesses from the previous time step
        ho[1:] = h[1:]
        h[1:] = (ga[1:] - ga[:-1]) * state.depth
    elif method == 1:
        ho[:] = h[:]
        h[:] = ga[:] * state.depth
    elif method == 2:
        ho[:] = h[:]
    else:
        raise ValueError(
            f"updategrid: No valid grid_method specified: {state.grid_method}"
        )

    # Compute interface depths zi and layer-centre depths z.
    # zi[0] = seabed; zi[nlev] = sea surface (= zeta when depth = depth0 + zeta).
    # z[i] = centre of layer i.
    zi[0] = -state.depth0
    zi[1:] = zi[0] + np.cumsum(h[1:])
    z[1:] = zi[:-1] + 0.5 * h[1:]


# ---------------------------------------------------------------------------
# Private helpers for file-based grid methods
# ---------------------------------------------------------------------------


def _read_sigma_grid_file(
    state: MeanflowState,
    nlev: int,
    ga: np.ndarray,
) -> None:
    """Read sigma layer fractions from *state.grid_file* (method 1).

    !  Sigma, the fraction each layer occupies is read from file.
    !  The first layer in the file is the surface layer (i=nlev).
    !  All fractions must sum to 1 within 1e-8.
    """
    path = Path(state.grid_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Unable to open grid file for reading: {state.grid_file}"
        )

    with path.open() as f:
        nlayers = int(f.readline().strip())
        if nlayers != nlev:
            raise ValueError(
                f"Number of layers specified in file ({nlayers}) "
                f"!= number of model layers ({nlev})"
            )

        depth_sum = 0.0
        count = 0
        for i in range(nlev, 0, -1):  # surface layer first in file
            line = f.readline()
            if not line:
                raise OSError(f"Error reading grid file: {state.grid_file}")
            ga[i] = float(line.split()[0])
            depth_sum += ga[i]
            count += 1

    if count != nlayers:
        raise ValueError(
            f"Number of layers read from file ({count}) "
            f"!= number of model layers ({nlayers})"
        )

    if abs(depth_sum - 1.0) > 1.0e-8:
        raise ValueError(
            f"Sum of all sigma fractions in grid_file should be 1.0, got {depth_sum}"
        )


def _read_cartesian_grid_file(
    state: MeanflowState,
    nlev: int,
    h: np.ndarray,
) -> None:
    """Read Cartesian layer thicknesses from *state.grid_file* (method 2).

    !  Cartesian, the layer thickness is read from file.
    !  The first layer in the file is the surface layer (i=nlev).
    !  The sum of all thicknesses must equal depth0 within 1e-5 m.
    """
    path = Path(state.grid_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Unable to open grid file for reading: {state.grid_file}"
        )

    with path.open() as f:
        nlayers = int(f.readline().strip())
        if nlayers != nlev:
            raise ValueError(
                f"nlev ({nlev}) must equal the number of layers in: {state.grid_file}"
            )

        depth_sum = 0.0
        count = 0
        for i in range(nlev, 0, -1):  # surface layer first in file
            line = f.readline()
            if not line:
                raise OSError(f"Error reading grid file: {state.grid_file}")
            h[i] = float(line.split()[0])
            depth_sum += h[i]
            count += 1

    if count != nlayers:
        raise ValueError(
            f"Number of layers read from file ({count}) "
            f"!= number of model layers ({nlayers})"
        )

    if abs(depth_sum - state.depth0) > 1.0e-5:
        raise ValueError(
            f"Sum of all layer thicknesses should equal total depth "
            f"{state.depth0}, got {depth_sum}"
        )
