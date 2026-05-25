"""Daemon shutdown tests."""

from __future__ import annotations

import json
from io import StringIO

from pygotm.serve.daemon import serve_forever


def test_shutdown_returns_zero_and_stops() -> None:
    stdout = StringIO()
    status = serve_forever(
        stdin=StringIO(
            '{"id":"stop","method":"shutdown"}\n{"id":"late","method":"version"}\n'
        ),
        stdout=stdout,
        stderr=StringIO(),
        do_warmup=False,
    )

    assert status == 0
    assert [json.loads(line)["id"] for line in stdout.getvalue().splitlines()] == [
        "stop"
    ]
