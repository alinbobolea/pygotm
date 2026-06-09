"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: hypsograph
!
! !DESCRIPTION:
!  This module is responsible for reading in values for the hypography from a
!  file specified in the meanflow namelist and updating it according to the
!  GOTM (time dependant) grid layers.
!  The hypsograph is only used if lake is true.
!
! !REVISION HISTORY:
!  Original author(s): Lennart Schueler
!
!EOP
!-----------------------------------------------------------------------
"""

from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np

__all__ = [
    "HypsographState",
    "read_hypsograph",
    "update_hypsograph",
    "vc2zi",
    "vc2zi_kernel",
    "zi2vc",
    "zi2vc_kernel",
]

_ONE_THIRD = 1.0 / 3.0


@dataclass(slots=True)
class HypsographState:
    """Hypsograph input table and derived frustum volumes."""

    nlev_input: int
    zi_input: np.ndarray
    af_input: np.ndarray
    sqrt_af_input: np.ndarray
    v_input: np.ndarray


def read_hypsograph(path: str | Path, depth0: float) -> HypsographState:
    """Read a GOTM lake hypsograph file.

    ! !IROUTINE: Initial read in of the hypsograph from specified file
    !
    ! !DESCRIPTION:
    !  Reads in the hypsograph from file at "unit" and saves everything
    !  to the *_input variables.
    !
    ! !REVISION HISTORY:
    !  Original author(s): Lennart Schueler
    """

    lines: list[str] = []
    config_path = Path(path)
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    if not lines:
        msg = f"empty hypsograph file: {config_path}"
        raise ValueError(msg)

    header = lines[0].split()
    if len(header) < 2:
        msg = f"hypsograph header must contain line count and read order: {path}"
        raise ValueError(msg)
    nlines = int(header[0])
    read_order = int(header[1])
    nlev_input = nlines - 1
    if len(lines[1:]) < nlines:
        msg = f"hypsograph expected {nlines} data rows, got {len(lines) - 1}"
        raise ValueError(msg)

    zi_input = np.zeros(nlev_input + 1, dtype=np.float64)
    af_input = np.zeros(nlev_input + 1, dtype=np.float64)

    def read_row(source_index: int, target_index: int) -> None:
        parts = lines[1 + source_index].split()
        if len(parts) < 2:
            msg = f"hypsograph row {source_index + 2} must contain z and area"
            raise ValueError(msg)
        zi_input[target_index] = float(parts[0])
        af_input[target_index] = float(parts[1])

    if read_order == 1:
        for i in range(nlev_input + 1):
            read_row(i, i)
    elif read_order == 2:
        for source, i in enumerate(range(nlev_input, -1, -1)):
            read_row(source, i)
    elif read_order == 3:
        for i in range(nlev_input + 1):
            read_row(i, i)
        bottom = zi_input[0]
        for i in range(nlev_input, -1, -1):
            zi_input[i] = zi_input[i] - bottom - depth0
    elif read_order == 4:
        for source, i in enumerate(range(nlev_input, -1, -1)):
            read_row(source, i)
        bottom = zi_input[0]
        for i in range(nlev_input, -1, -1):
            zi_input[i] = zi_input[i] - bottom - depth0
    else:
        msg = f"unsupported hypsograph read order {read_order}"
        raise ValueError(msg)

    if np.any(np.diff(zi_input) <= 0.0):
        msg = "hypsograph interface depths must increase from bottom to surface"
        raise ValueError(msg)
    if np.any(af_input < 0.0):
        msg = "hypsograph interface areas must be non-negative"
        raise ValueError(msg)

    sqrt_af_input = np.sqrt(af_input)
    v_input = np.zeros(nlev_input + 1, dtype=np.float64)
    for i in range(1, nlev_input + 1):
        dz = zi_input[i] - zi_input[i - 1]
        v_input[i] = (
            _ONE_THIRD
            * dz
            * (af_input[i - 1] + sqrt_af_input[i - 1] * sqrt_af_input[i] + af_input[i])
        )

    return HypsographState(
        nlev_input=nlev_input,
        zi_input=zi_input,
        af_input=af_input,
        sqrt_af_input=sqrt_af_input,
        v_input=v_input,
    )


@numba.njit(cache=True)
def zi2vc_kernel(
    nlev: int,
    nlev_input: int,
    zi_input: np.ndarray,
    af_input: np.ndarray,
    sqrt_af_input: np.ndarray,
    v_input: np.ndarray,
    zi: np.ndarray,
    af: np.ndarray,
    vc: np.ndarray,
) -> None:
    """Calculate volumes of layers."""

    af[0] = af_input[0]
    for i in range(nlev + 1):
        vc[i] = 0.0

    d_v_filled = 0.0
    ii = 1
    for i in range(1, nlev + 1):
        while ii < nlev_input and zi_input[ii] < zi[i]:
            vc[i] = vc[i] + v_input[ii] - d_v_filled
            d_v_filled = 0.0
            ii += 1

        h_frust = zi[i] - zi_input[ii - 1]
        theta = h_frust / (zi_input[ii] - zi_input[ii - 1])
        sqrt_ab = sqrt_af_input[ii - 1]
        sqrt_at = sqrt_af_input[ii]
        sqrt_af = theta * sqrt_at + (1.0 - theta) * sqrt_ab
        af[i] = sqrt_af * sqrt_af
        v_frust = _ONE_THIRD * h_frust * (af_input[ii - 1] + sqrt_ab * sqrt_af + af[i])
        vc[i] = vc[i] + v_frust - d_v_filled
        d_v_filled = v_frust


@numba.njit(cache=True)
def vc2zi_kernel(
    nlev: int,
    nlev_input: int,
    depth0: float,
    zi_input: np.ndarray,
    af_input: np.ndarray,
    sqrt_af_input: np.ndarray,
    v_input: np.ndarray,
    vc: np.ndarray,
    zi: np.ndarray,
) -> None:
    """Calculate layer heights."""

    zi[0] = -depth0
    v_frust = 0.0
    ii = 1
    for i in range(1, nlev + 1):
        d_v_filled = 0.0
        while ii < nlev_input and v_input[ii] - v_frust < vc[i] - d_v_filled:
            d_v_filled = d_v_filled + v_input[ii] - v_frust
            v_frust = 0.0
            ii += 1

        v_frust = v_frust + vc[i] - d_v_filled
        theta = v_frust / v_input[ii]
        sqrt_ab = sqrt_af_input[ii - 1]
        sqrt_at = sqrt_af_input[ii]
        if abs(sqrt_at - sqrt_ab) < 1.0e-14:
            area = af_input[ii - 1]
            if area <= 0.0:
                h_frust = 0.0
            else:
                h_frust = v_frust / area
        else:
            sqrt_af = (theta * sqrt_at**3 + (1.0 - theta) * sqrt_ab**3) ** (_ONE_THIRD)
            h_frust = (
                (zi_input[ii] - zi_input[ii - 1])
                * (sqrt_af - sqrt_ab)
                / (sqrt_at - sqrt_ab)
            )
        zi[i] = zi_input[ii - 1] + h_frust


def zi2vc(
    state: HypsographState,
    nlev: int,
    zi: np.ndarray,
    af: np.ndarray,
    vc: np.ndarray,
) -> None:
    """Calculate interface areas and cell volumes for *zi*."""

    zi2vc_kernel(
        nlev,
        state.nlev_input,
        state.zi_input,
        state.af_input,
        state.sqrt_af_input,
        state.v_input,
        zi,
        af,
        vc,
    )


def vc2zi(
    state: HypsographState,
    nlev: int,
    depth0: float,
    vc: np.ndarray,
    zi: np.ndarray,
) -> None:
    """Calculate interface depths from cell volumes."""

    vc2zi_kernel(
        nlev,
        state.nlev_input,
        depth0,
        state.zi_input,
        state.af_input,
        state.sqrt_af_input,
        state.v_input,
        vc,
        zi,
    )


def update_hypsograph(
    state: HypsographState,
    nlev: int,
    zi: np.ndarray,
    af: np.ndarray,
    vc: np.ndarray,
) -> None:
    """Update hypsography for the current grid."""

    zi2vc(state, nlev, zi, af, vc)
