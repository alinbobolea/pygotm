"""Tests for ``pygotm run --progress json`` event semantics."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from pygotm.driver import GotmDriver
from pygotm.progress import ProgressReporter
from tests.fixtures import bundled_case_path

_COUETTE_CONFIG = bundled_case_path("couette")


def _write_short_couette_config(path: Path) -> None:
    config_text = _COUETTE_CONFIG.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "stop: 2005-01-02 00:00:00",
        "stop: 2005-01-01 00:00:20",
        1,
    )
    path.write_text(config_text.replace("nlev: 100", "nlev: 8", 1), encoding="utf-8")


def test_hydro_only_progress_is_indeterminate(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_short_couette_config(config_path)
    stream = StringIO()

    dataset = GotmDriver(config_path).run(
        max_steps=1,
        output=False,
        progress=ProgressReporter(stream=stream, mode="json", run_id="test-run"),
    )
    dataset.close()

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert events[0] == {
        "event": "started",
        "phase": "initializing",
        "run_id": "test-run",
    }
    assert any(
        event.get("event") == "phase"
        and event.get("phase") == "integrating"
        and event.get("progress_mode") == "indeterminate"
        for event in events
    )
    assert events[-1]["event"] == "finished"
    assert events[-1]["exit_code"] == 0
    assert not any(event.get("event") == "progress" for event in events)
