"""
Observation input — translation of ``input.F90``.

Reads time-series and profile observations from text files and interpolates
them linearly to the current simulation time.  Two input modes are supported:

* ``method_constant`` (0) — constant value supplied at configuration time.
* ``method_file`` (2) — ASCII text file with one or more time-stamped records;
  values are linearly interpolated between bracketing records.

The Fortran constant ``method_unsupported = huge(1)`` is mapped to
:data:`method_unsupported` ``= math.inf``.

Fortran types are mapped to Python classes:

* ``type_input`` → :class:`InputBase`
* ``type_scalar_input`` → :class:`ScalarInput`
* ``type_profile_input`` → :class:`ProfileInput`
* ``type_scalar_input_list`` → ``list[ScalarInput]`` (via :class:`InputManager`)

Public interface: :func:`init_input`, :func:`do_input`, :func:`close_input`,
:func:`register_input`, :func:`register_scalar_input`,
:func:`register_profile_input`, :func:`read_obs`, :func:`read_profiles`,
:class:`InputManager`, :class:`ScalarInput`, :class:`ProfileInput`.

Original authors: Jorn Bruggeman.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np

from pygotm.util.gridinterpol import gridinterpol
from pygotm.util.time import julian_day, time_diff

__all__ = [
    "InputManager",
    "ProfileInput",
    "ScalarInput",
    "close_input",
    "do_input",
    "init_input",
    "method_unsupported",
    "read_obs",
    "read_profiles",
    "register_input",
    "register_profile_input",
    "register_scalar_input",
]

method_unsupported = math.inf
maxpathlen = 256


def _parse_timestamp(text: str) -> tuple[int, int, int, int, int, int]:
    try:
        year = int(text[0:4])
        month = int(text[5:7])
        day = int(text[8:10])
        hour = int(text[11:13])
        minute = int(text[14:16])
        second = int(text[17:19])
    except (ValueError, IndexError) as exc:
        msg = f"invalid timestamp {text!r}"
        raise ValueError(msg) from exc
    return year, month, day, hour, minute, second


def _timestamp_to_julian_seconds(
    timestamp: tuple[int, int, int, int, int, int],
) -> tuple[int, int]:
    year, month, day, hour, minute, second = timestamp
    jul = julian_day(year, month, day)
    secs = hour * 3600 + minute * 60 + second
    return jul, secs


def _non_comment_lines(stream: TextIO, line_number: int) -> tuple[str, int]:
    while True:
        raw = stream.readline()
        if raw == "":
            raise EOFError
        line_number += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        return raw.rstrip("\n"), line_number


@dataclass
class InputBase:
    """Base class for scalar and profile inputs."""

    name: str = ""
    method: int = 0
    scale_factor: float = 1.0
    path: str = ""
    index: int = 1
    add_offset: float = 0.0
    constant_value: float = 0.0
    minimum: float = -math.inf
    maximum: float = math.inf
    method_off: float = method_unsupported
    method_constant: int = 0
    method_file: int = 2

    def configure(
        self,
        *,
        method: int | None = None,
        path: str | None = None,
        index: int | None = None,
        constant_value: float | None = None,
        scale_factor: float | None = None,
        add_offset: float | None = None,
        name: str | None = None,
    ) -> None:
        if method is not None:
            self.method = method
        if path is not None:
            self.path = path
        if index is not None and self.method == self.method_file:
            self.index = index
        if constant_value is not None:
            self.constant_value = constant_value
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if add_offset is not None:
            self.add_offset = add_offset
        if name is not None:
            self.name = name


@dataclass
class ProfileInput(InputBase):
    data: np.ndarray | None = None


@dataclass
class ScalarInput(InputBase):
    value: float = 0.0


@dataclass
class _ProfileFile:
    path: Path
    variables: list[ProfileInput] = field(default_factory=list)
    prof1: np.ndarray | None = None
    prof2: np.ndarray | None = None
    alpha: np.ndarray | None = None
    jul1: int = 0
    secs1: int = 0
    jul2: int = 0
    secs2: int = 0
    lines: int = 0
    nprofiles: int = 0
    one_profile: bool = False
    handle: TextIO | None = None

    def initialize(self, nlev: int) -> None:
        if self.handle is not None:
            return
        self.handle = self.path.open("r", encoding="utf-8")
        nvar = max(variable.index for variable in self.variables)
        self.prof1 = np.zeros((nlev + 1, nvar), dtype=np.float64)
        self.prof2 = np.zeros((nlev + 1, nvar), dtype=np.float64)
        self.alpha = np.zeros((nlev + 1, nvar), dtype=np.float64)

    def update(self, jul: int, secs: int, nlev: int, z: np.ndarray) -> None:
        self.initialize(nlev)
        if self.one_profile:
            return
        assert self.prof1 is not None
        assert self.prof2 is not None
        assert self.alpha is not None
        assert self.handle is not None

        if time_diff(self.jul2, self.secs2, jul, secs) < 0.0:
            while True:
                self.jul1 = self.jul2
                self.secs1 = self.secs2
                self.prof1[:, :] = self.prof2
                try:
                    timestamp, profiles, self.lines = read_profiles(
                        self.handle,
                        nlev,
                        self.prof2.shape[1],
                        z,
                        line_number=self.lines,
                    )
                except EOFError:
                    if self.nprofiles == 1:
                        self.one_profile = True
                        for variable in self.variables:
                            variable.data = self.prof1[:, variable.index - 1].copy()
                        return
                    msg = (
                        f"end of file reached while updating profile inputs from "
                        f"{self.path}"
                    )
                    raise RuntimeError(msg) from None

                for variable in self.variables:
                    column = variable.index - 1
                    self.prof2[:, column] = (
                        variable.scale_factor * profiles[:, column]
                        + variable.add_offset
                    )
                    if np.any(self.prof2[:, column] < variable.minimum):
                        msg = (
                            f"{variable.name} profile in {self.path} fell below "
                            f"minimum {variable.minimum}"
                        )
                        raise ValueError(msg)
                    if np.any(self.prof2[:, column] > variable.maximum):
                        msg = (
                            f"{variable.name} profile in {self.path} exceeded "
                            f"maximum {variable.maximum}"
                        )
                        raise ValueError(msg)

                self.nprofiles += 1
                self.jul2, self.secs2 = _timestamp_to_julian_seconds(timestamp)
                if time_diff(self.jul2, self.secs2, jul, secs) > 0.0:
                    break

            if self.nprofiles == 1:
                msg = f"simulation starts before the first profile in {self.path}"
                raise RuntimeError(msg)

            dt = time_diff(self.jul2, self.secs2, self.jul1, self.secs1)
            self.alpha[:, :] = (self.prof2 - self.prof1) / dt

        t = time_diff(jul, secs, self.jul1, self.secs1)
        for variable in self.variables:
            column = variable.index - 1
            variable.data = self.prof1[:, column] + t * self.alpha[:, column]

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.prof1 = None
        self.prof2 = None
        self.alpha = None


@dataclass
class _TimeseriesFile:
    path: Path
    variables: list[ScalarInput] = field(default_factory=list)
    obs1: np.ndarray | None = None
    obs2: np.ndarray | None = None
    alpha: np.ndarray | None = None
    jul1: int = 0
    secs1: int = 0
    jul2: int = 0
    secs2: int = 0
    lines: int = 0
    n: int = 0
    handle: TextIO | None = None

    def initialize(self) -> None:
        if self.handle is not None:
            return
        self.handle = self.path.open("r", encoding="utf-8")
        nvar = max(variable.index for variable in self.variables)
        self.obs1 = np.zeros(nvar, dtype=np.float64)
        self.obs2 = np.zeros(nvar, dtype=np.float64)
        self.alpha = np.zeros(nvar, dtype=np.float64)

    def update(self, jul: int, secs: int) -> None:
        self.initialize()
        assert self.obs1 is not None
        assert self.obs2 is not None
        assert self.alpha is not None
        assert self.handle is not None

        if time_diff(self.jul2, self.secs2, jul, secs) < 0.0:
            while True:
                self.jul1 = self.jul2
                self.secs1 = self.secs2
                self.obs1[:] = self.obs2
                try:
                    timestamp, observations, self.lines = read_obs(
                        self.handle,
                        self.obs2.shape[0],
                        line_number=self.lines,
                    )
                except EOFError as exc:
                    msg = (
                        f"end of file reached while updating time series from "
                        f"{self.path}"
                    )
                    raise RuntimeError(msg) from exc

                self.obs2[:] = observations
                for variable in self.variables:
                    column = variable.index - 1
                    self.obs2[column] = (
                        variable.scale_factor * self.obs2[column] + variable.add_offset
                    )
                    if self.obs2[column] < variable.minimum:
                        msg = (
                            f"{variable.name} in {self.path} fell below minimum "
                            f"{variable.minimum}"
                        )
                        raise ValueError(msg)
                    if self.obs2[column] > variable.maximum:
                        msg = (
                            f"{variable.name} in {self.path} exceeded maximum "
                            f"{variable.maximum}"
                        )
                        raise ValueError(msg)

                self.n += 1
                self.jul2, self.secs2 = _timestamp_to_julian_seconds(timestamp)
                if time_diff(self.jul2, self.secs2, jul, secs) > 0.0:
                    break

            if self.n == 1:
                msg = f"simulation starts before the first observation in {self.path}"
                raise RuntimeError(msg)

            dt = time_diff(self.jul2, self.secs2, self.jul1, self.secs1)
            self.alpha[:] = (self.obs2 - self.obs1) / dt

        t = time_diff(jul, secs, self.jul1, self.secs1)
        for variable in self.variables:
            column = variable.index - 1
            interpolated = self.obs1[column] + t * self.alpha[column]
            low = min(self.obs1[column], self.obs2[column])
            high = max(self.obs1[column], self.obs2[column])
            variable.value = min(high, max(low, interpolated))

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.obs1 = None
        self.obs2 = None
        self.alpha = None


@dataclass
class InputManager:
    """Stateful translation of the GOTM ``input`` module."""

    nlev: int | None = None
    scalar_inputs: list[ScalarInput] = field(default_factory=list)
    profile_inputs: list[ProfileInput] = field(default_factory=list)
    _profile_files: dict[Path, _ProfileFile] = field(default_factory=dict)
    _timeseries_files: dict[Path, _TimeseriesFile] = field(default_factory=dict)

    def register_profile_input(self, input_: ProfileInput) -> None:
        if not input_.name:
            raise ValueError("profile input has not had a name assigned")
        if self.nlev is None:
            msg = "input module has not been initialised with depth information"
            raise RuntimeError(msg)

        input_.data = np.zeros(self.nlev + 1, dtype=np.float64)
        self.profile_inputs.append(input_)
        if input_.method == input_.method_constant:
            input_.data.fill(input_.constant_value)
        elif input_.method == input_.method_file:
            if not input_.path:
                msg = f"empty file path specified for profile input {input_.name}"
                raise ValueError(msg)
            file_state = self._profile_files.setdefault(
                Path(input_.path), _ProfileFile(Path(input_.path))
            )
            file_state.variables.append(input_)
        else:
            input_.data.fill(0.0)

    def register_scalar_input(self, input_: ScalarInput) -> None:
        if not input_.name:
            raise ValueError("scalar input has not had a name assigned")

        self.scalar_inputs.append(input_)
        if input_.method == input_.method_constant:
            input_.value = input_.constant_value
        elif input_.method == input_.method_file:
            if not input_.path:
                msg = f"empty file path specified for scalar input {input_.name}"
                raise ValueError(msg)
            file_state = self._timeseries_files.setdefault(
                Path(input_.path), _TimeseriesFile(Path(input_.path))
            )
            file_state.variables.append(input_)
        else:
            input_.value = 0.0

    def do_input(
        self,
        jul: int,
        secs: int,
        nlev: int | None = None,
        z: np.ndarray | None = None,
    ) -> None:
        if self._profile_files and (nlev is None or z is None):
            msg = "depth-varying inputs require nlev and z in do_input"
            raise ValueError(msg)

        if nlev is not None and z is not None:
            for profile_file in self._profile_files.values():
                profile_file.update(jul, secs, nlev, z)

        for timeseries_file in self._timeseries_files.values():
            timeseries_file.update(jul, secs)

    def close(self) -> None:
        for profile_file in self._profile_files.values():
            profile_file.close()
        for timeseries_file in self._timeseries_files.values():
            timeseries_file.close()
        self._profile_files.clear()
        self._timeseries_files.clear()
        self.scalar_inputs.clear()
        self.profile_inputs.clear()
        self.nlev = None


def read_obs(
    stream: TextIO,
    n: int,
    *,
    line_number: int = 0,
) -> tuple[tuple[int, int, int, int, int, int], np.ndarray, int]:
    """Read a non-profile observation row from *stream*."""

    raw, next_line = _non_comment_lines(stream, line_number)
    timestamp = _parse_timestamp(raw[:19])
    values = np.fromstring(raw[19:], sep=" ", dtype=np.float64)
    if values.size < n:
        msg = f"expected {n} values after timestamp, found {values.size}"
        raise ValueError(msg)
    return timestamp, values[:n], next_line


def read_profiles(
    stream: TextIO,
    nlev: int,
    cols: int,
    z: np.ndarray,
    *,
    line_number: int = 0,
) -> tuple[tuple[int, int, int, int, int, int], np.ndarray, int]:
    """Read and vertically interpolate a profile block from *stream*."""

    header, next_line = _non_comment_lines(stream, line_number)
    timestamp = _parse_timestamp(header[:19])
    header_values = np.fromstring(header[19:], sep=" ", dtype=np.float64)
    if header_values.size < 2:
        msg = "profile block header must contain N and up_down"
        raise ValueError(msg)
    count = int(header_values[0])
    up_down = int(header_values[1])

    tmp_depth = np.zeros(count + 1, dtype=np.float64)
    tmp_profs = np.zeros((count + 1, cols), dtype=np.float64)
    if up_down == 1:
        indices = range(1, count + 1)
    else:
        indices = range(count, 0, -1)

    current_line = next_line
    for index in indices:
        raw, current_line = _non_comment_lines(stream, current_line)
        values = np.fromstring(raw, sep=" ", dtype=np.float64)
        if values.size < cols + 1:
            msg = f"profile row must contain depth plus {cols} values"
            raise ValueError(msg)
        tmp_depth[index] = values[0]
        tmp_profs[index, :] = values[1 : cols + 1]

    profiles = gridinterpol(tmp_depth, tmp_profs, z, nlev)
    return timestamp, profiles, current_line


_manager = InputManager()


def init_input(n: int | None = None) -> None:
    """Initialise the module-level input manager."""

    global _manager
    _manager = InputManager(nlev=n)


def register_profile_input(input_: ProfileInput) -> None:
    _manager.register_profile_input(input_)


def register_scalar_input(input_: ScalarInput) -> None:
    _manager.register_scalar_input(input_)


def register_input(input_: ScalarInput | ProfileInput) -> None:
    if isinstance(input_, ProfileInput):
        register_profile_input(input_)
    else:
        register_scalar_input(input_)


def do_input(
    jul: int,
    secs: int,
    nlev: int | None = None,
    z: np.ndarray | None = None,
) -> None:
    _manager.do_input(jul, secs, nlev=nlev, z=z)


def close_input() -> None:
    _manager.close()
