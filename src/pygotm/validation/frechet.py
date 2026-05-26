"""Discrete Frechet distance and dynamic normalization for validation."""

from typing import TypedDict

import numpy as np
import numpy.typing as npt
from numba import njit

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "FrechetResult",
    "discrete_frechet_iter",
    "discrete_frechet_iter_numba",
    "dynamic_log_range_normalize_pair",
    "frechet_raw_and_normalized",
]


class FrechetResult(TypedDict):
    """Raw and normalized Frechet distances for one paired series."""

    d_raw: float
    d_norm: float
    normalization_mode: str


def _as_points(values: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.ascontiguousarray(arr, dtype=np.float64)


def discrete_frechet_iter(P: npt.ArrayLike, Q: npt.ArrayLike) -> float:
    """Reference Python implementation of iterative discrete Frechet distance."""

    p_arr = _as_points(P)
    q_arr = _as_points(Q)
    n, m = len(p_arr), len(q_arr)
    if n == 0 or m == 0:
        return 0.0

    c = np.empty((n, m), dtype=np.float64)
    c[0, 0] = np.linalg.norm(p_arr[0] - q_arr[0])
    for i in range(1, n):
        c[i, 0] = max(c[i - 1, 0], float(np.linalg.norm(p_arr[i] - q_arr[0])))
    for j in range(1, m):
        c[0, j] = max(c[0, j - 1], float(np.linalg.norm(p_arr[0] - q_arr[j])))

    for i in range(1, n):
        p_i = p_arr[i]
        c_im1 = c[i - 1]
        c_i = c[i]
        for j in range(1, m):
            d_ij = float(np.linalg.norm(p_i - q_arr[j]))
            c_i[j] = max(min(c_im1[j], c_im1[j - 1], c_i[j - 1]), d_ij)
    return float(c[n - 1, m - 1])


@njit(cache=True, fastmath=False)
def _euclid(Pi: FloatArray, Qj: FloatArray) -> float:
    """Euclidean distance between two one-dimensional points."""

    total = 0.0
    for k in range(Pi.shape[0]):
        delta = Pi[k] - Qj[k]
        total += delta * delta
    return float(total**0.5)


@njit(cache=True, fastmath=False)
def discrete_frechet_iter_numba(P: FloatArray, Q: FloatArray) -> float:
    """Numba-accelerated iterative discrete Frechet distance."""

    n = P.shape[0]
    m = Q.shape[0]
    if n == 0 or m == 0:
        return 0.0

    c = np.empty((n, m), dtype=np.float64)
    c[0, 0] = _euclid(P[0], Q[0])
    for i in range(1, n):
        distance = _euclid(P[i], Q[0])
        previous = c[i - 1, 0]
        c[i, 0] = previous if previous > distance else distance
    for j in range(1, m):
        distance = _euclid(P[0], Q[j])
        previous = c[0, j - 1]
        c[0, j] = previous if previous > distance else distance

    for i in range(1, n):
        p_i = P[i]
        for j in range(1, m):
            d_ij = _euclid(p_i, Q[j])
            a = c[i - 1, j]
            b = c[i - 1, j - 1]
            d = c[i, j - 1]
            min_ab = a if a < b else b
            min_prev = min_ab if min_ab < d else d
            c[i, j] = min_prev if min_prev > d_ij else d_ij

    return float(c[n - 1, m - 1])


def _prepare_pair(a: npt.ArrayLike, b: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.size != b_arr.size:
        msg = "Input series must have the same length for paired comparison."
        raise ValueError(msg)
    finite = np.isfinite(a_arr) & np.isfinite(b_arr)
    return a_arr[finite], b_arr[finite]


def _robust_range(
    values: FloatArray, q_low: float, q_high: float
) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    return (
        float(np.percentile(values, q_low)),
        float(np.percentile(values, q_high)),
    )


def _compute_normalization_params(
    finite_mag: FloatArray,
    robust: bool,
    q_low: float,
    q_high: float,
    switch_oom: float,
    eps_floor: float,
) -> dict[str, float | str]:
    """Derive normalization mode and range from full-data magnitudes.

    Computing percentiles, span, and mode-selection on the full data set (not
    on a subsequently downsampled subset) keeps the normalization stable when
    the downsampler skips peak-bearing strides.
    """

    if finite_mag.size == 0:
        return {"mode": "degenerate"}

    if robust:
        low_est, high_est = _robust_range(finite_mag, q_low, q_high)
        if not np.isfinite(low_est) or not np.isfinite(high_est) or high_est <= 0.0:
            low_est = float(np.min(finite_mag))
            high_est = float(np.max(finite_mag))
    else:
        low_est = float(np.min(finite_mag))
        high_est = float(np.max(finite_mag))

    if not np.isfinite(high_est) or high_est <= 0.0:
        return {"mode": "degenerate"}

    low_pos = max(low_est, eps_floor)
    span_decades = float(np.log10(high_est) - np.log10(low_pos))

    if span_decades >= switch_oom:
        pos_all = finite_mag[finite_mag > 0.0]
        min_pos = float(np.min(pos_all)) if pos_all.size else 0.0
        eps_dyn = max(eps_floor, 0.1 * min_pos)
        log_all = np.log10(finite_mag + eps_dyn)
        finite_log = log_all[np.isfinite(log_all)]
        if robust:
            log_lo, log_hi = _robust_range(finite_log, q_low, q_high)
            if not np.isfinite(log_lo) or not np.isfinite(log_hi) or log_hi <= log_lo:
                log_lo = float(np.min(finite_log))
                log_hi = float(np.max(finite_log))
        else:
            log_lo = float(np.min(finite_log))
            log_hi = float(np.max(finite_log))
        denom = log_hi - log_lo
        return {
            "mode": "log",
            "span_decades": span_decades,
            "eps_dyn": eps_dyn,
            "log_lo": log_lo,
            "denom": denom,
        }

    lin_lo, lin_hi = low_est, high_est
    if not np.isfinite(lin_lo) or not np.isfinite(lin_hi) or lin_hi <= lin_lo:
        lin_lo = float(np.min(finite_mag))
        lin_hi = float(np.max(finite_mag))
    denom = lin_hi - lin_lo
    return {
        "mode": "linear",
        "span_decades": span_decades,
        "lin_lo": lin_lo,
        "denom": denom,
    }


def _apply_normalization_params(
    a: FloatArray,
    b: FloatArray,
    params: dict[str, float | str],
) -> tuple[FloatArray, FloatArray]:
    """Map two arrays to [0, 1] using normalization params from another set."""

    mode = params.get("mode", "degenerate")
    mag_a = np.abs(a)
    mag_b = np.abs(b)
    if mode == "degenerate":
        return np.zeros_like(a, dtype=np.float64), np.zeros_like(b, dtype=np.float64)

    if mode == "log":
        eps_dyn = float(params["eps_dyn"])
        log_lo = float(params["log_lo"])
        denom = float(params["denom"])
        log_a = np.log10(mag_a + eps_dyn)
        log_b = np.log10(mag_b + eps_dyn)
        if not np.isfinite(denom) or denom <= 1.0e-15:
            return (
                np.zeros_like(a, dtype=np.float64),
                np.zeros_like(b, dtype=np.float64),
            )
        a_norm = (log_a - log_lo) / denom
        b_norm = (log_b - log_lo) / denom
        return np.clip(a_norm, 0.0, 1.0), np.clip(b_norm, 0.0, 1.0)

    # linear
    lin_lo = float(params["lin_lo"])
    denom = float(params["denom"])
    if not np.isfinite(denom) or denom <= 1.0e-15:
        return np.zeros_like(a, dtype=np.float64), np.zeros_like(b, dtype=np.float64)
    a_norm = (mag_a - lin_lo) / denom
    b_norm = (mag_b - lin_lo) / denom
    return np.clip(a_norm, 0.0, 1.0), np.clip(b_norm, 0.0, 1.0)


def dynamic_log_range_normalize_pair(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    robust: bool = True,
    q_low: float = 1.0,
    q_high: float = 99.0,
    switch_oom: float = 2.0,
    eps_floor: float = 1.0e-12,
) -> tuple[FloatArray, FloatArray, dict[str, float | str]]:
    """Normalize a paired series to [0, 1] using dynamic linear/log scaling.

    Public API preserved for backward compatibility — derives normalization
    params from the input pair and applies them to the same pair.
    """

    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    mag_all = np.concatenate([np.abs(a_arr), np.abs(b_arr)])
    finite_mag = mag_all[np.isfinite(mag_all)]
    params = _compute_normalization_params(
        finite_mag, robust, q_low, q_high, switch_oom, eps_floor
    )
    a_norm, b_norm = _apply_normalization_params(a_arr, b_arr, params)
    meta_keys = ("mode", "span_decades")
    meta = {k: params[k] for k in meta_keys if k in params}
    return a_norm, b_norm, meta


def _downsample_pair(
    a: FloatArray,
    b: FloatArray,
    max_points: int,
) -> tuple[FloatArray, FloatArray]:
    if max_points <= 0 or a.size <= max_points:
        return a, b
    indices = np.linspace(0, a.size - 1, max_points, dtype=np.int64)
    return a[indices], b[indices]


def _frechet_distance(P: FloatArray, Q: FloatArray) -> float:
    try:
        return float(discrete_frechet_iter_numba(P, Q))
    except Exception:
        return float(discrete_frechet_iter(P, Q))


def frechet_raw_and_normalized(
    runA: npt.ArrayLike,
    runB: npt.ArrayLike,
    abs_tolerance: float = 1.0e-12,
    rel_tolerance: float = 1.0e-6,
    norm_tolerance: float = 0.01,
    robust: bool = True,
    q_low: float = 1.0,
    q_high: float = 99.0,
    switch_oom: float = 2.0,
    eps_floor: float = 1.0e-12,
    frechet_k: int = 400,
) -> FrechetResult:
    """Return raw and dynamically normalized Frechet distances.

    Normalization mode and range are computed from the *full* aligned series
    so that the downsampling stride cannot flip the mode (log/linear) by
    landing on or skipping isolated peaks.
    """

    del norm_tolerance
    a, b = _prepare_pair(runA, runB)
    if a.size == 0:
        return {"d_raw": 0.0, "d_norm": 0.0, "normalization_mode": "degenerate"}

    mag_all = np.concatenate([np.abs(a), np.abs(b)])
    finite_mag = mag_all[np.isfinite(mag_all)]
    norm_params = _compute_normalization_params(
        finite_mag, robust, q_low, q_high, switch_oom, eps_floor
    )

    a_sample, b_sample = _downsample_pair(a, b, frechet_k)
    p_raw = np.ascontiguousarray(a_sample.reshape(-1, 1), dtype=np.float64)
    q_raw = np.ascontiguousarray(b_sample.reshape(-1, 1), dtype=np.float64)
    d_raw = _frechet_distance(p_raw, q_raw)
    if d_raw < abs_tolerance:
        return {"d_raw": 0.0, "d_norm": 0.0, "normalization_mode": "abs_tolerance"}

    signal_scale = max(
        float(np.max(np.abs(a))), float(np.max(np.abs(b))), eps_floor
    )
    if d_raw < rel_tolerance * signal_scale:
        return {"d_raw": d_raw, "d_norm": 0.0, "normalization_mode": "rel_tolerance"}

    a_norm, b_norm = _apply_normalization_params(a_sample, b_sample, norm_params)
    p_norm = np.ascontiguousarray(a_norm.reshape(-1, 1), dtype=np.float64)
    q_norm = np.ascontiguousarray(b_norm.reshape(-1, 1), dtype=np.float64)
    d_norm = _frechet_distance(p_norm, q_norm)
    return {
        "d_raw": d_raw,
        "d_norm": d_norm,
        "normalization_mode": str(norm_params.get("mode", "degenerate")),
    }
