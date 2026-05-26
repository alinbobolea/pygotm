"""Daemon run-method tests."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import xarray as xr

from pygotm.serve import daemon
from tests.fixtures import bundled_case_path

_COUETTE_CONFIG = bundled_case_path("couette")
_CHANNEL_CONFIG = bundled_case_path("channel")


def test_daemon_run_invokes_driver_and_returns_attrs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "gotm.yaml"
    output_path = tmp_path / "out.nc"
    expected_output = output_path
    config_path.write_text("version: 7\n", encoding="utf-8")

    class FakeDriver:
        def __init__(self, config: Path) -> None:
            self.config = config

        def run(
            self,
            *,
            output_path: Path,
            max_steps: int | None,
            progress: object | None,
        ) -> xr.Dataset:
            assert self.config == config_path
            assert output_path == expected_output
            assert max_steps == 3
            assert progress is not None
            output_path.write_text("netcdf", encoding="utf-8")
            return xr.Dataset(attrs={"runtime": "compiled"})

    monkeypatch.setattr(daemon, "GotmDriver", FakeDriver)

    stdin = StringIO(
        json.dumps(
            {
                "id": "run-1",
                "method": "run",
                "params": {
                    "config_path": str(config_path),
                    "output_path": str(output_path),
                    "max_steps": 3,
                },
            }
        )
        + "\n"
        + '{"id":"stop","method":"shutdown"}\n'
    )
    stdout = StringIO()
    stderr = StringIO()

    daemon.serve_forever(
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        do_warmup=False,
    )

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["id"] == "run-1"
    assert responses[0]["ok"] is True
    assert responses[0]["result"]["attrs"]["runtime"] == "compiled"
    assert output_path.read_text(encoding="utf-8") == "netcdf"
    assert stderr.getvalue() == ""


def test_daemon_runs_reference_cases_sequentially_without_state_leak(
    tmp_path: Path,
) -> None:
    couette_output = tmp_path / "couette.nc"
    channel_output = tmp_path / "channel.nc"
    stdin = StringIO(
        json.dumps(
            {
                "id": "run-couette",
                "method": "run",
                "params": {
                    "config_path": str(_COUETTE_CONFIG),
                    "output_path": str(couette_output),
                    "max_steps": 1,
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "run-channel",
                "method": "run",
                "params": {
                    "config_path": str(_CHANNEL_CONFIG),
                    "output_path": str(channel_output),
                    "max_steps": 1,
                },
            }
        )
        + "\n"
        + '{"id":"stop","method":"shutdown"}\n'
    )
    stdout = StringIO()
    stderr = StringIO()

    daemon.serve_forever(
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        do_warmup=False,
    )

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [response["id"] for response in responses] == [
        "run-couette",
        "run-channel",
        "stop",
    ]
    assert responses[0]["ok"] is True
    assert responses[1]["ok"] is True
    assert responses[2]["ok"] is True
    assert couette_output.exists()
    assert channel_output.exists()

    with (
        xr.open_dataset(couette_output, engine="scipy") as couette,
        xr.open_dataset(
            channel_output,
            engine="scipy",
        ) as channel,
    ):
        assert couette.attrs["runtime"] == "compiled"
        assert channel.attrs["runtime"] == "compiled"
        assert couette.attrs["source_yaml"] == str(_COUETTE_CONFIG)
        assert channel.attrs["source_yaml"] == str(_CHANNEL_CONFIG)
        assert couette.attrs["source_yaml"] != channel.attrs["source_yaml"]
        assert len(couette.data_vars) > 0
        assert len(channel.data_vars) > 0

    progress_events = [json.loads(line) for line in stderr.getvalue().splitlines()]
    finished_outputs = {
        event.get("output_path")
        for event in progress_events
        if event.get("event") == "finished"
    }
    assert {str(couette_output), str(channel_output)}.issubset(finished_outputs)
