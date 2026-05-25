"""Line-oriented JSON-RPC daemon used by pyGOTM Studio subprocesses."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

from pygotm.config import GotmConfig, GotmSettings
from pygotm.driver import GotmDriver
from pygotm.errors import (
    EXIT_CONFIG_ERROR,
    EXIT_SUCCESS,
    IntegrationError,
    error_code_for_exception,
)
from pygotm.gotm.print_version import collect_version_info
from pygotm.progress import ProgressReporter
from pygotm.schema import config_schema, output_schema

__all__ = ["serve_forever", "warm_kernel"]


def _warmup_document() -> dict[str, object]:
    return {
        "version": 7,
        "title": "pyGOTM daemon warmup",
        "location": {"latitude": 0.0, "longitude": 0.0, "depth": 5.0},
        "time": {
            "start": "2000-01-01 00:00:00",
            "stop": "2000-01-01 00:10:00",
            "dt": 600.0,
        },
        "grid": {"nlev": 2},
        "temperature": {"method": "off"},
        "salinity": {"method": "off"},
        "surface": {
            "fluxes": {
                "method": "off",
                "tx": {"method": "constant", "constant_value": 0.0},
                "ty": {"method": "constant", "constant_value": 0.0},
                "heat": {"method": "constant", "constant_value": 0.0},
            },
            "swr": {"method": "constant", "constant_value": 0.0},
            "ice": {"model": "no_ice"},
        },
        "turbulence": {
            "turb_method": "second_order",
            "tke_method": "tke",
            "len_scale_method": "omega",
            "stab_method": "constant",
        },
        "output": {
            "warmup": {
                "time_unit": "dt",
                "time_step": 1,
                "time_method": "point",
                "variables": [{"source": "/*"}],
            }
        },
    }


def warm_kernel() -> str:
    """Warm the compiled runtime without using validation reference files."""

    document = _warmup_document()
    settings = GotmSettings.model_validate(document)
    dataset = GotmDriver(GotmConfig.from_settings(settings, document=document)).run(
        max_steps=1,
        output=False,
    )
    dataset.close()
    return "complete"


def _json_line(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def _write_response(stream: TextIO, payload: Mapping[str, object]) -> None:
    stream.write(_json_line(payload))
    stream.flush()


def _error_response(
    request_id: object,
    *,
    code: int,
    message: str,
) -> dict[str, object]:
    return {
        "id": request_id,
        "ok": False,
        "error": {"code": int(code), "message": message},
    }


def _params(request: Mapping[str, object]) -> Mapping[str, object]:
    raw = request.get("params", {})
    if isinstance(raw, Mapping):
        return raw
    return {}


def _handle_version(warmup_status: str) -> dict[str, object]:
    info: dict[str, object] = dict(collect_version_info())
    info["warmup"] = warmup_status
    return info


def _handle_run(
    params: Mapping[str, object],
    *,
    progress_stream: TextIO,
) -> dict[str, object]:
    config_path = params.get("config_path")
    output_path = params.get("output_path")
    if config_path is None or output_path is None:
        msg = "run params require config_path and output_path"
        raise IntegrationError(EXIT_CONFIG_ERROR, msg)
    max_steps_raw = params.get("max_steps")
    max_steps = None if max_steps_raw is None else int(str(max_steps_raw))
    dataset = GotmDriver(Path(str(config_path))).run(
        output_path=Path(str(output_path)),
        max_steps=max_steps,
        progress=ProgressReporter(stream=progress_stream, mode="json"),
    )
    try:
        return {
            "output_path": str(output_path),
            "attrs": dict(dataset.attrs),
        }
    finally:
        dataset.close()


def _method_handlers(
    *,
    warmup_status: str,
    stderr: TextIO,
) -> dict[str, Callable[[Mapping[str, object]], dict[str, object]]]:
    return {
        "version": lambda params: _handle_version(warmup_status),
        "schema_config": lambda params: config_schema(),
        "schema_output": lambda params: output_schema(
            None
            if params.get("config_path") is None
            else str(params.get("config_path"))
        ),
        "run": lambda params: _handle_run(params, progress_stream=stderr),
    }


def serve_forever(
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    do_warmup: bool = True,
) -> int:
    """Serve newline-delimited JSON-RPC requests until ``shutdown``."""

    warmup_status = "skipped"
    if do_warmup:
        try:
            warmup_status = warm_kernel()
        except Exception as exc:  # pragma: no cover - defensive startup guard.
            warmup_status = "partial"
            stderr.write(f"pygotm serve warmup partial: {exc}\n")
            stderr.flush()

    handlers = _method_handlers(warmup_status=warmup_status, stderr=stderr)

    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_response(
                stdout,
                _error_response(
                    None,
                    code=EXIT_CONFIG_ERROR,
                    message=f"malformed JSON request: {exc.msg}",
                ),
            )
            continue

        if not isinstance(request, dict):
            _write_response(
                stdout,
                _error_response(
                    None,
                    code=EXIT_CONFIG_ERROR,
                    message="JSON-RPC request must be an object",
                ),
            )
            continue

        request_id = request.get("id")
        method = request.get("method")
        if method == "shutdown":
            _write_response(stdout, {"id": request_id, "ok": True, "result": {}})
            return EXIT_SUCCESS
        if not isinstance(method, str) or method not in handlers:
            _write_response(
                stdout,
                _error_response(
                    request_id,
                    code=EXIT_CONFIG_ERROR,
                    message=f"unknown method: {method!r}",
                ),
            )
            continue

        try:
            result = handlers[method](_params(request))
        except Exception as exc:
            _write_response(
                stdout,
                _error_response(
                    request_id,
                    code=error_code_for_exception(exc),
                    message=str(exc),
                ),
            )
            continue

        _write_response(stdout, {"id": request_id, "ok": True, "result": result})

    return EXIT_SUCCESS
