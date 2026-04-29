"""Tests for seagrass drag parameterization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pygotm.extras.seagrass.seagrass import (
    MISS_VALUE,
    SeagrassState,
    do_seagrass,
    end_seagrass,
    init_seagrass,
    post_init_seagrass,
)


def _write_grass(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "3",
                "0.1 0.01 0.10",
                "0.2 0.02 0.20",
                "0.3 0.03 0.30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_init_off_is_noop() -> None:
    state = SeagrassState()
    init_seagrass(state, method=0)
    assert not state.seagrass_calc
    do_seagrass(
        state,
        2,
        1.0,
        np.zeros(3),
        np.zeros(3),
        np.ones(3),
        np.zeros(3),
        np.zeros(3),
    )


def test_post_init_loads_file_and_marks_layers_above_canopy(tmp_path: Path) -> None:
    grass = tmp_path / "seagrass.dat"
    _write_grass(grass)
    state = SeagrassState()
    init_seagrass(state, method=1, grassfile=str(grass), alpha=0.5)
    h = np.ones(6)

    post_init_seagrass(state, 5, h)

    assert state.seagrass_calc
    assert state.grassn == 3
    assert state.xx is not None
    assert state.yy is not None
    assert state.xx[state.grassind + 1] == MISS_VALUE
    assert state.yy[state.grassind + 1] == MISS_VALUE


def test_do_seagrass_clamps_excursion_and_adds_drag_and_production(
    tmp_path: Path,
) -> None:
    grass = tmp_path / "seagrass.dat"
    _write_grass(grass)
    state = SeagrassState()
    init_seagrass(state, method=1, grassfile=str(grass), alpha=0.25)
    nlev = 5
    h = np.full(nlev + 1, 0.1)
    post_init_seagrass(state, nlev, h)

    u = np.full(nlev + 1, 1.0)
    v = np.zeros(nlev + 1)
    drag = np.zeros(nlev + 1)
    xP = np.zeros(nlev + 1)
    do_seagrass(state, nlev, 1.0, u, v, h, drag, xP)

    assert state.xx is not None
    assert state.excur is not None
    excursion = np.abs(state.xx[1 : state.grassind + 1])
    limit = state.excur[1 : state.grassind + 1] + 1.0e-14
    assert np.all(excursion <= limit)
    assert np.any(drag[1 : state.grassind + 1] > 0.0)
    assert np.any(xP[1:state.grassind] >= 0.0)
    assert np.isfinite(drag).all()
    assert np.isfinite(xP).all()


def test_end_seagrass_releases_arrays(tmp_path: Path) -> None:
    grass = tmp_path / "seagrass.dat"
    _write_grass(grass)
    state = SeagrassState()
    init_seagrass(state, method=1, grassfile=str(grass))
    post_init_seagrass(state, 3, np.ones(4))

    end_seagrass(state)

    assert state.xx is None
    assert state.grassz is None
