"""Tests for pygotm.input.input."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from pygotm.input.input import (
    InputManager,
    ProfileInput,
    ScalarInput,
    close_input,
    do_input,
    init_input,
    read_obs,
    read_profiles,
    register_input,
)
from pygotm.util.time import julian_day


@pytest.fixture(autouse=True)
def _cleanup_input_manager() -> Iterator[None]:
    close_input()
    yield
    close_input()


def test_scalar_constant_input_registration_sets_value() -> None:
    manager = InputManager()
    scalar = ScalarInput(name="cloud", method=0, constant_value=0.25)
    manager.register_scalar_input(scalar)
    assert scalar.value == pytest.approx(0.25)


def test_profile_registration_requires_depth_information() -> None:
    manager = InputManager()
    profile = ProfileInput(name="temp", method=0, constant_value=5.0)
    with pytest.raises(RuntimeError, match="depth information"):
        manager.register_profile_input(profile)


def test_read_obs_skips_comments_and_reads_values(tmp_path: Path) -> None:
    path = tmp_path / "series.dat"
    path.write_text(
        "# comment\n\n2000-01-01 00:00:00 1.0 2.0 3.0\n",
        encoding="utf-8",
    )
    with path.open("r", encoding="utf-8") as handle:
        timestamp, values, line = read_obs(handle, 3)
    assert timestamp == (2000, 1, 1, 0, 0, 0)
    assert np.allclose(values, np.array([1.0, 2.0, 3.0]))
    assert line == 3


def test_read_profiles_interpolates_to_model_grid(tmp_path: Path) -> None:
    path = tmp_path / "profiles.dat"
    path.write_text(
        "\n".join(
            [
                "2000-01-01 00:00:00 3 1",
                "0.0 0.0 10.0",
                "0.5 0.5 20.0",
                "1.0 1.0 30.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    model_z = np.linspace(0.0, 1.0, 6)
    with path.open("r", encoding="utf-8") as handle:
        timestamp, profiles, _ = read_profiles(handle, 5, 2, model_z)
    assert timestamp == (2000, 1, 1, 0, 0, 0)
    assert profiles.shape == (6, 2)
    assert profiles[5, 0] == pytest.approx(1.0)
    assert profiles[3, 1] == pytest.approx(22.0)


def test_timeseries_interpolation_updates_multiple_variables_from_one_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "series.dat"
    path.write_text(
        "\n".join(
            [
                "2000-01-01 00:00:00 1.0 10.0",
                "2000-01-01 01:00:00 3.0 14.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manager = InputManager()
    first = ScalarInput(name="a", method=2, path=str(path), index=1)
    second = ScalarInput(name="b", method=2, path=str(path), index=2)
    manager.register_scalar_input(first)
    manager.register_scalar_input(second)
    manager.do_input(julian_day(2000, 1, 1), 1800)
    assert first.value == pytest.approx(2.0)
    assert second.value == pytest.approx(12.0)


def test_profile_interpolation_reuses_single_profile_after_eof(tmp_path: Path) -> None:
    path = tmp_path / "profile.dat"
    path.write_text(
        "\n".join(
            [
                "2000-01-01 00:00:00 3 1",
                "0.0 1.0",
                "0.5 2.0",
                "1.0 3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manager = InputManager(nlev=5)
    profile = ProfileInput(name="temp", method=2, path=str(path), index=1)
    manager.register_profile_input(profile)
    z = np.linspace(0.0, 1.0, 6)
    assert profile.data is not None
    data_id = id(profile.data)
    manager.do_input(julian_day(2000, 1, 1), 0, 5, z)
    assert profile.data is not None
    assert id(profile.data) == data_id
    first = profile.data.copy()
    manager.do_input(julian_day(2000, 1, 2), 0, 5, z)
    assert profile.data is not None
    assert id(profile.data) == data_id
    assert np.allclose(profile.data, first)


def test_profile_interpolation_reuses_data_array_between_updates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile.dat"
    path.write_text(
        "\n".join(
            [
                "2000-01-01 00:00:00 3 1",
                "0.0 0.0",
                "0.5 2.0",
                "1.0 4.0",
                "2000-01-01 01:00:00 3 1",
                "0.0 10.0",
                "0.5 12.0",
                "1.0 14.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manager = InputManager(nlev=2)
    profile = ProfileInput(name="temp", method=2, path=str(path), index=1)
    manager.register_profile_input(profile)
    z = np.linspace(0.0, 1.0, 3)
    assert profile.data is not None
    data_id = id(profile.data)

    manager.do_input(julian_day(2000, 1, 1), 1800, 2, z)
    assert profile.data is not None
    assert id(profile.data) == data_id
    assert np.allclose(profile.data, np.array([0.0, 7.0, 9.0]))

    manager.do_input(julian_day(2000, 1, 1), 2700, 2, z)
    assert profile.data is not None
    assert id(profile.data) == data_id
    assert np.allclose(profile.data, np.array([0.0, 9.5, 11.5]))


def test_do_input_requires_nlev_and_z_for_profile_files(tmp_path: Path) -> None:
    init_input(5)
    path = tmp_path / "profile.dat"
    path.write_text(
        "\n".join(
            [
                "2000-01-01 00:00:00 2 1",
                "0.0 1.0",
                "1.0 2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    register_input(ProfileInput(name="temp", method=2, path=str(path), index=1))
    with pytest.raises(ValueError, match="nlev and z"):
        do_input(julian_day(2000, 1, 1), 0)


def test_profile_bounds_violation_raises(tmp_path: Path) -> None:
    path = tmp_path / "profile.dat"
    path.write_text(
        "\n".join(
            [
                "2000-01-01 00:00:00 2 1",
                "0.0 1.0",
                "1.0 5.0",
                "2000-01-01 01:00:00 2 1",
                "0.0 1.0",
                "1.0 6.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manager = InputManager(nlev=4)
    profile = ProfileInput(
        name="salt",
        method=2,
        path=str(path),
        index=1,
        maximum=5.5,
    )
    manager.register_profile_input(profile)
    z = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="exceeded maximum"):
        manager.do_input(julian_day(2000, 1, 1), 1800, 4, z)
