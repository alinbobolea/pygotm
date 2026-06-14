"""Aquirae process-boundary contract gate.

Aquirae never imports pygotm (forbidden_modules = ["pygotm"]); it consumes the
engine only across a process boundary via the ``pygotm`` CLI, the NetCDF
outputs, and the reference bundles. This gate snapshots the public CLI JSON
surface Aquirae ingests and lets a refactor prove that surface is unchanged.

Numeric NetCDF parity is covered separately by refactor_parity_gate.py; this
gate covers the schema/version/cite CLI contract.

Usage:
    conda run -n pygotm python scripts/aquirae_contract_gate.py capture baseline.json
    conda run -n pygotm python scripts/aquirae_contract_gate.py capture after.json
    conda run -n pygotm python scripts/aquirae_contract_gate.py diff baseline.json after.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Public CLI invocations Aquirae relies on, by stable key. Each must emit
# deterministic stdout (the schema/version/cite commands sort keys).
CONTRACT_COMMANDS: dict[str, list[str]] = {
    "version": ["pygotm", "version", "--json"],
    "schema.config": ["pygotm", "schema", "config", "--json"],
    "schema.output": ["pygotm", "schema", "output", "--json"],
    "schema.netcdf_attrs": ["pygotm", "schema", "netcdf-attrs", "--json"],
}


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(
        ["conda", "run", "-n", "pygotm", *cmd],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def capture(out_path: Path) -> None:
    manifest: dict[str, str] = {}
    for key, cmd in CONTRACT_COMMANDS.items():
        out = _run(cmd)
        manifest[key] = hashlib.sha256(out.encode()).hexdigest()
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {out_path} ({len(manifest)} CLI surfaces)")


def diff(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    mismatches = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if mismatches:
        print("AQUIRAE CONTRACT GATE FAILED — these CLI surfaces changed:")
        for m in mismatches:
            print(f"  {m}")
        return 1
    print("AQUIRAE CONTRACT GATE PASSED — public CLI surface byte-identical")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "capture":
        capture(Path(sys.argv[2]))
        return 0
    if cmd == "diff":
        return diff(Path(sys.argv[2]), Path(sys.argv[3]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
