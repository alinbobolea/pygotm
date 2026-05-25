"""Tests for pyGOTM serve JSON-RPC framing."""

from __future__ import annotations

import json
from io import StringIO

from pygotm.serve.daemon import serve_forever


def _run_daemon(requests: list[dict[str, object]]) -> list[dict[str, object]]:
    stdin = StringIO("".join(json.dumps(request) + "\n" for request in requests))
    stdout = StringIO()
    stderr = StringIO()
    serve_forever(stdin=stdin, stdout=stdout, stderr=stderr, do_warmup=False)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_version_request_returns_manifest_shaped_result() -> None:
    responses = _run_daemon(
        [
            {"id": "1", "method": "version"},
            {"id": "2", "method": "shutdown"},
        ]
    )

    assert responses[0]["id"] == "1"
    assert responses[0]["ok"] is True
    assert "pygotm_version" in responses[0]["result"]
    assert responses[0]["result"]["warmup"] == "skipped"
    assert responses[1] == {"id": "2", "ok": True, "result": {}}


def test_schema_requests_are_framed_as_stdout_json_only() -> None:
    responses = _run_daemon(
        [
            {"id": "1", "method": "schema_config"},
            {"id": "2", "method": "schema_output"},
            {"id": "3", "method": "shutdown"},
        ]
    )

    assert responses[0]["ok"] is True
    assert "properties" in responses[0]["result"]
    assert responses[1]["ok"] is True
    assert "variables" in responses[1]["result"]
