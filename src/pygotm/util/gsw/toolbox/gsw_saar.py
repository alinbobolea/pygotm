"""gsw_saar -- Absolute Salinity Anomaly Ratio from GOTM's GSW data.

Direct translation of:
    gotm-model/code/extern/gsw/toolbox/gsw_saar.f90

The SAAR grid is loaded lazily from GOTM's bundled
``gsw_mod_saar_data.f90`` data, packaged as
``pygotm.util.gsw.data.saar_2011_gotm.npz``. The external ``gsw`` package is not
used here because its newer SAAR grid does not match GOTM's 2011 data and breaks
Fortran parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files

import numpy as np

__all__ = ["gsw_saar"]

_DATA_PACKAGE = "pygotm.util.gsw.data"
_DATA_FILE = "saar_2011_gotm.npz"
_DELI = (0, 1, 1, 0)
_DELJ = (0, 0, 1, 1)
_LONGS_PAN = np.asarray([260.00, 272.59, 276.50, 278.65, 280.73, 292.0])
_LATS_PAN = np.asarray([19.55, 13.97, 9.60, 8.10, 9.33, 3.4])


@dataclass(frozen=True, slots=True)
class _SaarData:
    p_ref: np.ndarray
    lats_ref: np.ndarray
    longs_ref: np.ndarray
    saar_ref: np.ndarray
    ndepth_ref: np.ndarray


@lru_cache(maxsize=1)
def _load_saar_data() -> _SaarData:
    resource = files(_DATA_PACKAGE).joinpath(_DATA_FILE)
    try:
        with as_file(resource) as path, np.load(path) as data:
            return _SaarData(
                p_ref=np.asarray(data["p_ref"], dtype=np.float64),
                lats_ref=np.asarray(data["lats_ref"], dtype=np.float64),
                longs_ref=np.asarray(data["longs_ref"], dtype=np.float64),
                saar_ref=np.asarray(data["saar_ref"], dtype=np.float64),
                ndepth_ref=np.asarray(data["ndepth_ref"], dtype=np.int64),
            )
    except FileNotFoundError as exc:
        msg = (
            f"missing packaged GOTM SAAR data {_DATA_PACKAGE}.{_DATA_FILE}; "
            "regenerate it with scripts/generate_gotm_saar_data.py"
        )
        raise RuntimeError(msg) from exc


def _util_indx(x: np.ndarray, z: float) -> int:
    n = x.size
    if z > x[0] and z < x[n - 1]:
        kl = 0
        ku = n - 1
        while ku - kl > 1:
            km = (ku + kl) // 2
            if z > x[km]:
                kl = km
            else:
                ku = km
        ki = kl
        if z == x[ki + 1]:
            ki += 1
        return ki
    if z <= x[0]:
        return 0
    return n - 2


def _util_indx_vector(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(x, z, side="right") - 1
    return np.clip(idx, 0, x.size - 2)


def _add_mean(values: np.ndarray) -> np.ndarray:
    good = np.abs(values) <= 100.0
    count = np.sum(good, axis=1)
    total = np.sum(np.where(good, values, 0.0), axis=1)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return np.where(good, values, mean[:, np.newaxis])


def _add_barrier(
    values: np.ndarray,
    long: float,
    lat: float,
    long_grid: float,
    lat_grid: float,
    dlong_grid: float,
    dlat_grid: float,
) -> np.ndarray:
    k = _util_indx(_LONGS_PAN, long)
    r = (long - _LONGS_PAN[k]) / (_LONGS_PAN[k + 1] - _LONGS_PAN[k])
    lats_line = _LATS_PAN[k] + r * (_LATS_PAN[k + 1] - _LATS_PAN[k])
    above_line0 = lats_line <= lat

    k = _util_indx(_LONGS_PAN, long_grid)
    r = (long_grid - _LONGS_PAN[k]) / (_LONGS_PAN[k + 1] - _LONGS_PAN[k])
    lats_line = _LATS_PAN[k] + r * (_LATS_PAN[k + 1] - _LATS_PAN[k])
    above_line = np.empty(4, dtype=bool)
    above_line[0] = lats_line <= lat_grid
    above_line[3] = lats_line <= lat_grid + dlat_grid

    k = _util_indx(_LONGS_PAN, long_grid + dlong_grid)
    r = (long_grid + dlong_grid - _LONGS_PAN[k]) / (_LONGS_PAN[k + 1] - _LONGS_PAN[k])
    lats_line = _LATS_PAN[k] + r * (_LATS_PAN[k + 1] - _LATS_PAN[k])
    above_line[1] = lats_line <= lat_grid
    above_line[2] = lats_line <= lat_grid + dlat_grid

    good = (np.abs(values) <= 100.0) & (above_line == above_line0)
    count = np.sum(good, axis=1)
    total = np.sum(np.where(good, values, 0.0), axis=1)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    keep = (np.abs(values) < 1.0e10) & (above_line == above_line0)
    return np.where(keep, values, mean[:, np.newaxis])


def _prepare_saar_values(
    values: np.ndarray,
    *,
    long360: float,
    lat: float,
    long_grid: float,
    lat_grid: float,
    dlong_grid: float,
    dlat_grid: float,
) -> np.ndarray:
    in_panama = (
        _LONGS_PAN[0] <= long360 <= _LONGS_PAN[-1] - 0.001
        and _LATS_PAN[-1] <= lat <= _LATS_PAN[0]
    )
    if in_panama:
        return _add_barrier(
            values, long360, lat, long_grid, lat_grid, dlong_grid, dlat_grid
        )

    invalid = np.abs(np.sum(values, axis=1)) >= 1.0e10
    if not np.any(invalid):
        return values
    prepared = values.copy()
    prepared[invalid] = _add_mean(prepared[invalid])
    return prepared


def _saar_from_data(
    p: object, long: float, lat: float, data: _SaarData
) -> np.ndarray | float:
    p_array = np.asarray(p, dtype=np.float64)
    scalar = p_array.ndim == 0
    p_flat = np.atleast_1d(p_array).astype(np.float64, copy=False).ravel()

    long360 = float(long)
    if long360 < 0.0:
        long360 += 360.0

    nx = data.longs_ref.size
    ny = data.lats_ref.size
    indx0 = (
        int(
            np.floor(
                1.0
                + (nx - 1)
                * (long360 - data.longs_ref[0])
                / (data.longs_ref[-1] - data.longs_ref[0])
            )
        )
        - 1
    )
    indy0 = (
        int(
            np.floor(
                1.0
                + (ny - 1)
                * (float(lat) - data.lats_ref[0])
                / (data.lats_ref[-1] - data.lats_ref[0])
            )
        )
        - 1
    )

    result = np.empty_like(p_flat, dtype=np.float64)
    if not (0 <= indx0 < nx and 0 <= indy0 < ny):
        result.fill(np.nan)
        return float(result[0]) if scalar else result.reshape(p_array.shape)
    if indx0 == nx - 1:
        indx0 = nx - 2
    if indy0 == ny - 1:
        indy0 = ny - 2

    ndepth_max = -1
    for offset in range(4):
        ndepth = data.ndepth_ref[indy0 + _DELJ[offset], indx0 + _DELI[offset]]
        if ndepth > 0 and ndepth < 99:
            ndepth_max = max(ndepth_max, int(ndepth))
    if ndepth_max == -1:
        result.fill(0.0)
        return float(result[0]) if scalar else result.reshape(p_array.shape)

    p_tmp = np.minimum(p_flat, data.p_ref[ndepth_max - 1])
    indz0 = _util_indx_vector(data.p_ref, p_tmp)
    dlong = data.longs_ref[indx0 + 1] - data.longs_ref[indx0]
    dlat = data.lats_ref[indy0 + 1] - data.lats_ref[indy0]
    r1 = (long360 - data.longs_ref[indx0]) / dlong
    s1 = (float(lat) - data.lats_ref[indy0]) / dlat
    t1 = (p_tmp - data.p_ref[indz0]) / (data.p_ref[indz0 + 1] - data.p_ref[indz0])

    upper_values = np.stack(
        [
            data.saar_ref[indz0, indy0 + _DELJ[offset], indx0 + _DELI[offset]]
            for offset in range(4)
        ],
        axis=1,
    )
    upper_values = _prepare_saar_values(
        upper_values,
        long360=long360,
        lat=float(lat),
        long_grid=float(data.longs_ref[indx0]),
        lat_grid=float(data.lats_ref[indy0]),
        dlong_grid=float(dlong),
        dlat_grid=float(dlat),
    )
    sa_upper = (1.0 - s1) * (
        upper_values[:, 0] + r1 * (upper_values[:, 1] - upper_values[:, 0])
    ) + s1 * (upper_values[:, 3] + r1 * (upper_values[:, 2] - upper_values[:, 3]))

    lower_values = np.stack(
        [
            data.saar_ref[indz0 + 1, indy0 + _DELJ[offset], indx0 + _DELI[offset]]
            for offset in range(4)
        ],
        axis=1,
    )
    lower_values = _prepare_saar_values(
        lower_values,
        long360=long360,
        lat=float(lat),
        long_grid=float(data.longs_ref[indx0]),
        lat_grid=float(data.lats_ref[indy0]),
        dlong_grid=float(dlong),
        dlat_grid=float(dlat),
    )
    sa_lower = (1.0 - s1) * (
        lower_values[:, 0] + r1 * (lower_values[:, 1] - lower_values[:, 0])
    ) + s1 * (lower_values[:, 3] + r1 * (lower_values[:, 2] - lower_values[:, 3]))
    sa_lower = np.where(np.abs(sa_lower) >= 1.0e10, sa_upper, sa_lower)
    result[:] = sa_upper + t1 * (sa_lower - sa_upper)
    result = np.where(np.abs(result) >= 1.0e10, np.nan, result)
    return float(result[0]) if scalar else result.reshape(p_array.shape)


def gsw_saar(p: object, long: float, lat: float) -> np.ndarray | float:
    """Calculate the Absolute Salinity Anomaly Ratio."""

    data = _load_saar_data()
    return _saar_from_data(p, long, lat, data)
