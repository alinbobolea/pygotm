# ruff: noqa: E501
"""
Seagrass canopy parameterisation — translation of ``seagrass.F90``.

Treats seagrass canopies as Lagrangian tracers that either advect passively
with the horizontal current speed or rest at their excursion limits and
thereby exert friction on the mean flow (Verduin and Backhaus, 2000).
Turbulence generation due to seagrass friction is included as an extra
production term in the TKE balance.

Public interface: :func:`init_seagrass`, :func:`post_init_seagrass`,
:func:`do_seagrass`, :func:`end_seagrass`, :class:`SeagrassState`,
:data:`MISS_VALUE`.

Original authors: Hans Burchard, Karsten Bolding.
"""

from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np

__all__ = [
    "MISS_VALUE",
    "SeagrassState",
    "do_seagrass",
    "end_seagrass",
    "init_seagrass",
    "post_init_seagrass",
]

MISS_VALUE: float = -999.0


@dataclass
class SeagrassState:
    """State for the optional seagrass drag parameterization."""

    seagrass_calc: bool = False
    method: int = 0
    grassfile: str = "seagrass.dat"
    alpha: float = 0.0
    grassind: int = 0
    grassn: int = 0
    xx: np.ndarray | None = None
    yy: np.ndarray | None = None
    xxP: np.ndarray | None = None
    exc: np.ndarray | None = None
    vfric: np.ndarray | None = None
    grassz: np.ndarray | None = None
    excur: np.ndarray | None = None
    grassfric: np.ndarray | None = None


def init_seagrass(
    state: SeagrassState,
    *,
    method: int = 0,
    grassfile: str = "seagrass.dat",
    alpha: float = 0.0,
) -> None:
    """Read seagrass configuration from a YAML-equivalent mapping."""

    state.method = int(method)
    state.grassfile = grassfile
    state.alpha = float(alpha)
    state.seagrass_calc = state.method != 0


def _read_grass_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    values: list[list[float]] = []
    grassn = 0
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        parts = [float(part) for part in stripped.split()]
        if grassn == 0:
            grassn = int(parts[0])
            continue
        if len(parts) < 3:
            msg = f"seagrass row in {path} must contain z, excursion, friction"
            raise ValueError(msg)
        values.append(parts[:3])
    if grassn != len(values):
        msg = f"seagrass file {path} declared {grassn} rows, found {len(values)}"
        raise ValueError(msg)

    grassz = np.zeros(grassn + 1, dtype=np.float64)
    exc = np.zeros(grassn + 1, dtype=np.float64)
    vfric = np.zeros(grassn + 1, dtype=np.float64)
    for i, row in enumerate(values, start=1):
        grassz[i] = row[0]
        exc[i] = row[1]
        vfric[i] = row[2]
    return grassz, exc, vfric


def post_init_seagrass(state: SeagrassState, nlev: int, h: np.ndarray) -> None:
    """Allocate and initialise seagrass memory from file."""

    if not state.seagrass_calc:
        return

    state.grassz, state.exc, state.vfric = _read_grass_file(state.grassfile)
    state.grassn = state.grassz.size - 1
    state.xx = np.zeros(nlev + 1, dtype=np.float64)
    state.yy = np.zeros(nlev + 1, dtype=np.float64)
    state.xxP = np.zeros(nlev + 1, dtype=np.float64)
    state.excur = np.zeros(nlev + 1, dtype=np.float64)
    state.grassfric = np.zeros(nlev + 1, dtype=np.float64)

    z = 0.5 * h[1]
    state.grassind = 1
    for i in range(2, nlev + 1):
        z += 0.5 * (h[i - 1] + h[i])
        if state.grassz[state.grassn] > z:
            state.grassind = i

    state.xx[state.grassind + 1 : nlev + 1] = MISS_VALUE
    state.yy[state.grassind + 1 : nlev + 1] = MISS_VALUE


@numba.njit(cache=True)
def _interp_profile(
    nobs: int,
    obs_z: np.ndarray,
    obs_y: np.ndarray,
    nlev: int,
    model_z: np.ndarray,
    out: np.ndarray,
) -> None:
    for i in range(nlev + 1):
        out[i] = 0.0
    for i in range(1, nlev + 1):
        if model_z[i] >= obs_z[nobs]:
            out[i] = obs_y[nobs]
        elif model_z[i] <= obs_z[1]:
            out[i] = obs_y[1]
        else:
            ii = 1
            while obs_z[ii] <= model_z[i]:
                ii += 1
            rat = (model_z[i] - obs_z[ii - 1]) / (obs_z[ii] - obs_z[ii - 1])
            out[i] = (1.0 - rat) * obs_y[ii - 1] + rat * obs_y[ii]


@numba.njit(cache=True)
def _do_seagrass_kernel(
    nlev: int,
    dt: float,
    alpha: float,
    grassind: int,
    grassn: int,
    grassz: np.ndarray,
    exc: np.ndarray,
    vfric: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    drag: np.ndarray,
    xP: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xxP: np.ndarray,
    excur: np.ndarray,
    grassfric: np.ndarray,
) -> None:
    z = np.zeros(nlev + 1, dtype=np.float64)
    z[1] = 0.5 * h[1]
    for i in range(2, nlev + 1):
        z[i] = z[i - 1] + 0.5 * (h[i - 1] + h[i])

    _interp_profile(grassn, grassz, exc, nlev, z, excur)
    _interp_profile(grassn, grassz, vfric, nlev, z, grassfric)

    for i in range(1, grassind + 1):
        xx[i] += dt * u[i]
        yy[i] += dt * v[i]
        dist = np.sqrt(xx[i] * xx[i] + yy[i] * yy[i])
        if dist > excur[i] and dist > 0.0:
            xx[i] = excur[i] / dist * xx[i]
            yy[i] = excur[i] / dist * yy[i]
            drag[i] += grassfric[i]
            speed = np.sqrt(u[i] * u[i] + v[i] * v[i])
            xxP[i] = alpha * grassfric[i] * speed**3
        else:
            xxP[i] = 0.0

    for i in range(1, nlev):
        xP[i] = 0.5 * (xxP[i] + xxP[i + 1])


def do_seagrass(
    state: SeagrassState,
    nlev: int,
    dt: float,
    u: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    drag: np.ndarray,
    xP: np.ndarray,
) -> None:
    """Update the sea grass model."""

    if not state.seagrass_calc:
        return
    assert state.xx is not None
    assert state.yy is not None
    assert state.xxP is not None
    assert state.exc is not None
    assert state.vfric is not None
    assert state.grassz is not None
    assert state.excur is not None
    assert state.grassfric is not None

    _do_seagrass_kernel(
        nlev,
        dt,
        state.alpha,
        state.grassind,
        state.grassn,
        state.grassz,
        state.exc,
        state.vfric,
        u,
        v,
        h,
        drag,
        xP,
        state.xx,
        state.yy,
        state.xxP,
        state.excur,
        state.grassfric,
    )


def end_seagrass(state: SeagrassState) -> None:
    """Release seagrass work arrays."""

    state.xx = None
    state.yy = None
    state.xxP = None
    state.exc = None
    state.vfric = None
    state.grassz = None
    state.excur = None
    state.grassfric = None
