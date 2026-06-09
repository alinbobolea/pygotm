"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: streams
!
! !DESCRIPTION:
! This module is responsible for all calculations related to streams. This means
! reading in prescribed values from a given file, calculating the depth where
! the inflowing water masses interleave, and calculating vertical fluxes. These
! vertical fluxes are used by other routines to calculate vertical advection
! velocities.
!
! !REVISION HISTORY:
!  Original author(s): Lennart Schueler
!
!EOP
!-----------------------------------------------------------------------
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numba
import numpy as np

from pygotm.config.settings import InputSetting
from pygotm.input.input import ScalarInput, register_input

__all__ = [
    "BOTTOM_FLOW",
    "DEPTH_RANGE",
    "INTERLEAVING",
    "SURFACE_FLOW",
    "Stream",
    "StreamsState",
    "configure_streams_from_document",
    "post_init_streams",
    "unesco_density",
    "update_streams",
    "update_streams_kernel",
]

SURFACE_FLOW = 1
BOTTOM_FLOW = 2
DEPTH_RANGE = 3
INTERLEAVING = 4

_STREAM_METHODS = {
    "surface": SURFACE_FLOW,
    "bottom": BOTTOM_FLOW,
    "prescribed_depth_range": DEPTH_RANGE,
    "depth_range": DEPTH_RANGE,
    "interleaving": INTERLEAVING,
    "density": INTERLEAVING,
}
_SCALAR_METHOD = {"constant": 0, "file": 2, "off": 0}
_STREAM_CORE_KEYS = frozenset(("method", "zu", "zl", "flow", "temp", "salt"))


@dataclass(slots=True)
class Stream:
    """One lake stream configuration and its registered scalar inputs."""

    name: str
    method: int = SURFACE_FLOW
    zl: float = 0.0
    zu: float = 0.0
    flow: ScalarInput = field(default_factory=lambda: _stream_scalar("flow", None, 0.0))
    temp: ScalarInput = field(
        default_factory=lambda: _stream_scalar("temp", None, -1.0)
    )
    salt: ScalarInput = field(
        default_factory=lambda: _stream_scalar("salt", None, -1.0)
    )
    has_T: bool = False
    has_S: bool = False
    concentrations: dict[str, ScalarInput] = field(default_factory=dict)


@dataclass(slots=True)
class StreamsState:
    """Lake stream list plus work arrays used by update_streams."""

    streams: list[Stream] = field(default_factory=list)
    int_inflow: float = 0.0
    int_outflow: float = 0.0
    methods: np.ndarray | None = None
    zl: np.ndarray | None = None
    zu: np.ndarray | None = None
    has_T: np.ndarray | None = None
    has_S: np.ndarray | None = None
    flow_values: np.ndarray | None = None
    temp_values: np.ndarray | None = None
    salt_values: np.ndarray | None = None
    concentration_names: tuple[str, ...] = ()
    concentration_has: dict[str, np.ndarray] = field(default_factory=dict)
    concentration_values: dict[str, np.ndarray] = field(default_factory=dict)
    weights: np.ndarray | None = None
    Q: np.ndarray | None = None

    @property
    def nstreams(self) -> int:
        return len(self.streams)

    def update_values_from_inputs(self) -> None:
        """Copy current ScalarInput values into contiguous arrays."""

        if self.flow_values is None:
            return
        temp_values = self.temp_values
        salt_values = self.salt_values
        if temp_values is None or salt_values is None:
            raise ValueError("stream value arrays are not allocated")
        for i, stream in enumerate(self.streams):
            self.flow_values[i] = stream.flow.value
            temp_values[i] = stream.temp.value
            salt_values[i] = stream.salt.value
            for name, input_ in stream.concentrations.items():
                values = self.concentration_values.get(name)
                if values is not None:
                    values[i] = input_.value


def _method_token(value: object) -> int:
    if isinstance(value, bool):
        return SURFACE_FLOW
    if isinstance(value, int):
        if value in (SURFACE_FLOW, BOTTOM_FLOW, DEPTH_RANGE, INTERLEAVING):
            return value
        msg = f"unsupported stream method {value!r}"
        raise ValueError(msg)
    if value is None:
        return SURFACE_FLOW
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return SURFACE_FLOW
    if token.isdigit():
        return _method_token(int(token))
    try:
        return _STREAM_METHODS[token]
    except KeyError as exc:
        msg = f"unsupported stream method {value!r}"
        raise ValueError(msg) from exc


def _stream_scalar(
    name: str,
    raw: object | None,
    default_value: float,
) -> ScalarInput:
    setting = InputSetting.model_validate(
        {"constant_value": default_value} if raw is None else raw
    )
    try:
        method = _SCALAR_METHOD[str(setting.method)]
    except KeyError as exc:
        msg = f"unsupported stream scalar method {setting.method!r} for {name!r}"
        raise ValueError(msg) from exc
    return ScalarInput(
        name=name,
        method=method,
        path=setting.file,
        index=setting.column,
        constant_value=setting.constant_value,
        value=setting.constant_value if method == 0 else 0.0,
        scale_factor=setting.scale_factor,
        add_offset=setting.offset,
        method_constant=0,
        method_file=2,
    )


def configure_streams_from_document(
    document: Mapping[str, Any],
    *,
    lake: bool,
) -> StreamsState:
    """Build stream descriptors from a GOTM YAML document."""

    state = StreamsState()
    if not lake:
        return state

    streams_doc = document.get("streams")
    if not isinstance(streams_doc, Mapping):
        return state

    for name, raw_stream in streams_doc.items():
        if not isinstance(raw_stream, Mapping):
            continue
        method = _method_token(raw_stream.get("method", SURFACE_FLOW))
        temp = _stream_scalar(f"{name}_temp", raw_stream.get("temp"), -1.0)
        salt = _stream_scalar(f"{name}_salt", raw_stream.get("salt"), -1.0)
        has_T = not (temp.method == 0 and temp.constant_value < 0.0)
        has_S = not (salt.method == 0 and salt.constant_value < 0.0)
        if method == INTERLEAVING and not has_T:
            msg = "interleaving streams require a temperature input"
            raise ValueError(msg)
        concentrations = {
            str(key): _stream_scalar(f"{name}_{key}", raw_value, 0.0)
            for key, raw_value in raw_stream.items()
            if str(key) not in _STREAM_CORE_KEYS
        }
        state.streams.append(
            Stream(
                name=str(name),
                method=method,
                zl=float(raw_stream.get("zl", 0.0)),
                zu=float(raw_stream.get("zu", 0.0)),
                flow=_stream_scalar(f"{name}_flow", raw_stream.get("flow"), 0.0),
                temp=temp,
                salt=salt,
                has_T=has_T,
                has_S=has_S,
                concentrations=concentrations,
            )
        )
    return state


def post_init_streams(state: StreamsState, nlev: int) -> None:
    """Register stream inputs and allocate stream work arrays."""

    nstreams = state.nstreams
    state.methods = np.zeros(nstreams, dtype=np.int64)
    state.zl = np.zeros(nstreams, dtype=np.float64)
    state.zu = np.zeros(nstreams, dtype=np.float64)
    state.has_T = np.zeros(nstreams, dtype=np.int64)
    state.has_S = np.zeros(nstreams, dtype=np.int64)
    state.flow_values = np.zeros(nstreams, dtype=np.float64)
    state.temp_values = np.zeros(nstreams, dtype=np.float64)
    state.salt_values = np.zeros(nstreams, dtype=np.float64)
    concentration_names = list(
        dict.fromkeys(
            name for stream in state.streams for name in stream.concentrations
        )
    )
    state.concentration_names = tuple(concentration_names)
    state.concentration_has = {
        name: np.zeros(nstreams, dtype=np.int64) for name in concentration_names
    }
    state.concentration_values = {
        name: np.zeros(nstreams, dtype=np.float64) for name in concentration_names
    }
    state.weights = np.zeros((nstreams, nlev + 1), dtype=np.float64)
    state.Q = np.zeros((nstreams, nlev + 1), dtype=np.float64)

    for i, stream in enumerate(state.streams):
        register_input(stream.flow)
        register_input(stream.temp)
        register_input(stream.salt)
        for name, input_ in stream.concentrations.items():
            register_input(input_)
            state.concentration_has[name][i] = 1
        state.methods[i] = stream.method
        state.zl[i] = stream.zl
        state.zu[i] = stream.zu
        state.has_T[i] = 1 if stream.has_T else 0
        state.has_S[i] = 1 if stream.has_S else 0
        if stream.method == SURFACE_FLOW:
            state.weights[i, nlev] = 1.0
        elif stream.method == BOTTOM_FLOW:
            state.weights[i, 1] = 1.0
    state.update_values_from_inputs()


@numba.njit(cache=True, fastmath=False)
def unesco_density(S: float, T: float, p: float, use_pressure: bool) -> float:
    """Compute UNESCO density following gotm-lake util/eqstate.F90."""

    T2 = T * T
    T3 = T * T2
    T4 = T2 * T2
    T5 = T * T4
    S2 = S * S
    S3 = S * S2
    S15 = math.sqrt(S3) if S3 >= 0.0 else math.nan

    x = 999.842594 + 6.793952e-02 * T - 9.09529e-03 * T2 + 1.001685e-04 * T3
    x = x - 1.120083e-06 * T4 + 6.536332e-09 * T5
    x = x + S * (0.824493 - 4.0899e-03 * T + 7.6438e-05 * T2 - 8.2467e-07 * T3)
    x = x + S * 5.3875e-09 * T4
    x = x + S15 * (-5.72466e-03 + 1.0227e-04 * T - 1.6546e-06 * T2)
    x = x + 4.8314e-04 * S2

    if use_pressure and p > 0.0:
        p2 = p * p
        K = (
            19652.21
            + 148.4206 * T
            - 2.327105 * T2
            + 1.360477e-2 * T3
            - 5.155288e-5 * T4
            + 3.239908 * p
            + 1.43713e-3 * T * p
            + 1.16092e-4 * T2 * p
            - 5.77905e-7 * T3 * p
            + 8.50935e-5 * p2
            - 6.12293e-6 * T * p2
            + 5.2787e-8 * T2 * p2
            + 54.6746 * S
            - 0.603459 * T * S
            + 1.09987e-2 * T2 * S
            - 6.1670e-5 * T3 * S
            + 7.944e-2 * S15
            + 1.6483e-2 * T * S15
            - 5.3009e-4 * T2 * S15
            + 2.2838e-3 * p * S
            - 1.0981e-5 * T * p * S
            - 1.6078e-6 * T2 * p * S
            + 1.91075e-4 * p * S15
            - 9.9348e-7 * p2 * S
            + 2.0816e-8 * T * p2 * S
            + 9.1697e-10 * T2 * p2 * S
        )
        x = x / (1.0 - p / K)
    return x


@numba.njit(cache=True, fastmath=False)
def _get_weights(
    nlev: int,
    stream_index: int,
    method: int,
    zl: float,
    zu: float,
    nmin: int,
    nmax: int,
    h: np.ndarray,
    zi: np.ndarray,
    weights: np.ndarray,
) -> None:
    for n in range(nlev + 1):
        weights[stream_index, n] = 0.0

    if nmin == nmax:
        weights[stream_index, nmin] = 1.0
        return

    if method == DEPTH_RANGE:
        d = -(zl - zu)
        if nmax - nmin == 1:
            yh = zu - zi[nmax - 1]
            yl = -(zl - zi[nmin])
            weights[stream_index, nmax] = yh / d
            weights[stream_index, nmin] = yl / d
            return
        yh = zu - zi[nmax - 1]
        yi = 0.0
        for n in range(nmin + 1, nmax):
            yi += h[n]
        yl = -(zl - zi[nmin])
        weights[stream_index, nmax] = yh / d
        for n in range(nmin + 1, nmax):
            weights[stream_index, n] = h[n] / d
        weights[stream_index, nmin] = yl / d
        _ = yi
    elif method == INTERLEAVING:
        d = zi[nmax] - zi[nmin - 1]
        for n in range(nmin, nmax + 1):
            weights[stream_index, n] = h[n] / d


@numba.njit(cache=True, fastmath=False)
def update_streams_kernel(
    nlev: int,
    dt: float,
    methods: np.ndarray,
    zl: np.ndarray,
    zu: np.ndarray,
    has_T: np.ndarray,
    has_S: np.ndarray,
    flow_values: np.ndarray,
    temp_values: np.ndarray,
    salt_values: np.ndarray,
    S: np.ndarray,
    T: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    h: np.ndarray,
    weights: np.ndarray,
    stream_Q: np.ndarray,
    Qs: np.ndarray,
    Qt: np.ndarray,
    Ls: np.ndarray,
    Lt: np.ndarray,
    Qlayer: np.ndarray,
) -> tuple[float, float]:
    """Calculate stream layer weights and source/sink arrays."""

    for n in range(nlev + 1):
        Qs[n] = 0.0
        Qt[n] = 0.0
        Ls[n] = 0.0
        Lt[n] = 0.0
        Qlayer[n] = 0.0

    int_inflow = 0.0
    int_outflow = 0.0
    nstreams = methods.shape[0]
    for si in range(nstreams):
        flow = flow_values[si]
        for n in range(nlev + 1):
            stream_Q[si, n] = 0.0
        if flow == 0.0:
            continue

        method = methods[si]
        if method == SURFACE_FLOW:
            nmin = nlev
            nmax = nlev
            _get_weights(nlev, si, method, zl[si], zu[si], nmin, nmax, h, zi, weights)
        elif method == BOTTOM_FLOW:
            nmin = 1
            nmax = 1
            _get_weights(nlev, si, method, zl[si], zu[si], nmin, nmax, h, zi, weights)
        elif method == DEPTH_RANGE:
            nmin = nlev
            for n in range(1, nlev + 1):
                if zl[si] < zi[n]:
                    nmin = n
                    break
            nmax = nmin
            for n in range(nlev, nmin - 1, -1):
                if zi[n - 1] < zu[si]:
                    nmax = n
                    break
            _get_weights(nlev, si, method, zl[si], zu[si], nmin, nmax, h, zi, weights)
        elif method == INTERLEAVING:
            ti = temp_values[si]
            si_salt = salt_values[si]
            nmin = nlev
            rho_i = unesco_density(si_salt, ti, 0.0, False)
            for n in range(1, nlev + 1):
                depth = zi[nlev] - z[n]
                rho = unesco_density(S[n], T[n], depth / 10.0, False)
                if rho_i > rho:
                    nmin = n
                    break

            nmax = nmin
            for n in range(nlev, nmin - 1, -1):
                depth = zi[nlev] - z[n]
                rho = unesco_density(S[n], T[n], depth / 10.0, False)
                if rho_i < rho:
                    nmax = n
                    break
            _get_weights(nlev, si, method, zl[si], zu[si], nmin, nmax, h, zi, weights)
        else:
            raise ValueError("unsupported stream method")

        for n in range(1, nlev + 1):
            stream_Q[si, n] = weights[si, n] * flow

        if flow > 0.0:
            int_inflow += dt * flow
            if has_T[si] != 0:
                ti = temp_values[si]
            else:
                ti = 0.0
                for n in range(nmin, nmax + 1):
                    ti += T[n]
                ti = ti / (nmax - nmin + 1)
            si_salt = salt_values[si] if has_S[si] != 0 else 0.0
            for n in range(1, nlev + 1):
                Qt[n] += ti * stream_Q[si, n]
                Qs[n] += si_salt * stream_Q[si, n]
        else:
            int_outflow += dt * flow
            if has_T[si] != 0:
                ti = temp_values[si]
                for n in range(1, nlev + 1):
                    Qt[n] += ti * stream_Q[si, n]
            else:
                for n in range(1, nlev + 1):
                    Lt[n] += stream_Q[si, n]
            if has_S[si] != 0:
                si_salt = salt_values[si]
                for n in range(1, nlev + 1):
                    Qs[n] += si_salt * stream_Q[si, n]
            else:
                for n in range(1, nlev + 1):
                    Ls[n] += stream_Q[si, n]

        for n in range(1, nlev + 1):
            Qlayer[n] += stream_Q[si, n]

    return int_inflow, int_outflow


def update_streams(
    state: StreamsState,
    nlev: int,
    dt: float,
    S: np.ndarray,
    T: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    h: np.ndarray,
    Qs: np.ndarray,
    Qt: np.ndarray,
    Ls: np.ndarray,
    Lt: np.ndarray,
    Qlayer: np.ndarray,
) -> None:
    """Update stream forcing arrays in-place for the current input values."""

    if state.nstreams == 0:
        Qs.fill(0.0)
        Qt.fill(0.0)
        Ls.fill(0.0)
        Lt.fill(0.0)
        Qlayer.fill(0.0)
        return
    if (
        state.methods is None
        or state.zl is None
        or state.zu is None
        or state.has_T is None
        or state.has_S is None
        or state.flow_values is None
        or state.temp_values is None
        or state.salt_values is None
        or state.weights is None
        or state.Q is None
    ):
        raise RuntimeError("call post_init_streams before update_streams")

    state.update_values_from_inputs()
    int_inflow, int_outflow = update_streams_kernel(
        nlev,
        dt,
        state.methods,
        state.zl,
        state.zu,
        state.has_T,
        state.has_S,
        state.flow_values,
        state.temp_values,
        state.salt_values,
        S,
        T,
        z,
        zi,
        h,
        state.weights,
        state.Q,
        Qs,
        Qt,
        Ls,
        Lt,
        Qlayer,
    )
    state.int_inflow += int_inflow
    state.int_outflow += int_outflow
