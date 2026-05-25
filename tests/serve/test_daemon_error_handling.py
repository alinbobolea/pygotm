"""Daemon error response tests."""

from __future__ import annotations

import json
from io import StringIO

from pygotm.serve.daemon import serve_forever


def test_malformed_json_returns_error_response() -> None:
    stdout = StringIO()
    serve_forever(
        stdin=StringIO("{not json}\n"),
        stdout=stdout,
        stderr=StringIO(),
        do_warmup=False,
    )

    response = json.loads(stdout.getvalue())
    assert response["ok"] is False
    assert response["error"]["code"] == 10
    assert "malformed JSON" in response["error"]["message"]


def test_unknown_method_returns_error_response() -> None:
    stdout = StringIO()
    serve_forever(
        stdin=StringIO('{"id":"1","method":"missing"}\n'),
        stdout=stdout,
        stderr=StringIO(),
        do_warmup=False,
    )

    response = json.loads(stdout.getvalue())
    assert response["id"] == "1"
    assert response["ok"] is False
    assert response["error"]["code"] == 10


def test_run_request_schema_error_returns_config_error_code() -> None:
    stdout = StringIO()
    serve_forever(
        stdin=StringIO('{"id":"bad-run","method":"run","params":{}}\n'),
        stdout=stdout,
        stderr=StringIO(),
        do_warmup=False,
    )

    response = json.loads(stdout.getvalue())
    assert response["id"] == "bad-run"
    assert response["ok"] is False
    assert response["error"]["code"] == 10
    assert "config_path" in response["error"]["message"]
