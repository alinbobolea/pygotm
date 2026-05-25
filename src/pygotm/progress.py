"""Structured run-progress events for CLI and daemon integrations."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO
from uuid import uuid4

__all__ = ["ProgressReporter"]


@dataclass(slots=True)
class ProgressReporter:
    """Emit newline-delimited progress events to a text stream."""

    stream: TextIO = sys.stderr
    mode: str = "json"
    run_id: str = field(default_factory=lambda: uuid4().hex)

    def started(self, phase: str = "initializing") -> None:
        """Emit the start event."""

        self.emit({"event": "started", "run_id": self.run_id, "phase": phase})

    def phase(self, phase: str, **extra: object) -> None:
        """Emit a phase transition."""

        self.emit({"event": "phase", "run_id": self.run_id, "phase": phase, **extra})

    def progress(self, step: int, total_steps: int) -> None:
        """Emit determinate progress."""

        fraction = 1.0 if total_steps <= 0 else min(max(step / total_steps, 0.0), 1.0)
        self.emit(
            {
                "event": "progress",
                "run_id": self.run_id,
                "step": int(step),
                "total_steps": int(total_steps),
                "fraction": fraction,
            }
        )

    def finished(self, *, exit_code: int, output_path: str | Path | None) -> None:
        """Emit the finish event."""

        event: dict[str, object] = {
            "event": "finished",
            "run_id": self.run_id,
            "exit_code": int(exit_code),
        }
        if output_path is not None:
            event["output_path"] = str(output_path)
        self.emit(event)

    def failed(self, *, exit_code: int, message: str) -> None:
        """Emit a failed finish event."""

        self.emit(
            {
                "event": "finished",
                "run_id": self.run_id,
                "exit_code": int(exit_code),
                "error": message,
            }
        )

    def emit(self, event: dict[str, object]) -> None:
        """Write one event according to the configured mode."""

        if self.mode == "plain":
            self.stream.write(self._plain_event(event))
        else:
            self.stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
            self.stream.write("\n")
        self.stream.flush()

    @staticmethod
    def _plain_event(event: dict[str, object]) -> str:
        name = str(event.get("event", "event"))
        phase = event.get("phase")
        if name == "progress":
            step = event.get("step")
            total = event.get("total_steps")
            raw_fraction = event.get("fraction", 0.0)
            fraction = (
                float(raw_fraction)
                if isinstance(raw_fraction, (int, float, str))
                else 0.0
            )
            return f"progress {step}/{total} ({fraction:.1%})\n"
        if phase is not None:
            return f"{name}: {phase}\n"
        if "error" in event:
            return f"{name}: {event['error']}\n"
        return f"{name}\n"
