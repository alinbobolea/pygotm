"""Behavior-preserving refactor gate.

Runs a fixed set of validation cases through the compiled runtime and writes a
JSON manifest mapping each output NetCDF variable to a SHA-256 of its raw
little-endian bytes. Compare two manifests to prove that a code-motion refactor
left every numeric result byte-identical.

Usage:
    # capture baseline before refactoring
    conda run -n pygotm python scripts/refactor_parity_gate.py capture baseline.json
    # after a refactor task, capture again and diff
    conda run -n pygotm python scripts/refactor_parity_gate.py capture after.json
    conda run -n pygotm python scripts/refactor_parity_gate.py \
        diff baseline.json after.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from pygotm.validation.runner import run_case

# Small, fast, physics-representative cases. couette/channel/entrainment are the
# repo default fast set; they exercise meanflow + turbulence on the compiled path.
GATE_CASES: tuple[str, ...] = ("couette", "channel", "entrainment")


def _hash_dataset(path: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    with xr.open_dataset(path) as ds:
        for name in sorted(ds.data_vars):
            arr = np.ascontiguousarray(ds[name].values)
            # Normalize byte order so the hash is platform-stable.
            if arr.dtype.byteorder not in ("=", "|", "<"):
                arr = arr.astype(arr.dtype.newbyteorder("<"))
            h = hashlib.sha256()
            h.update(str(arr.dtype.str).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
            digests[name] = h.hexdigest()
    return digests


def capture(out_path: Path) -> None:
    manifest: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="pygotm_parity_") as tmp:
        runs_dir = Path(tmp)
        for case_name in GATE_CASES:
            nc_path, _elapsed = run_case(case_name, runs_dir)
            manifest[case_name] = _hash_dataset(nc_path)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {out_path} ({sum(len(v) for v in manifest.values())} variables)")


def diff(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    mismatches: list[str] = []
    for case in sorted(set(a) | set(b)):
        va, vb = a.get(case, {}), b.get(case, {})
        for var in sorted(set(va) | set(vb)):
            if va.get(var) != vb.get(var):
                mismatches.append(f"{case}:{var}")
    if mismatches:
        print("PARITY GATE FAILED — these variables changed:")
        for m in mismatches:
            print(f"  {m}")
        return 1
    print("PARITY GATE PASSED — all variables byte-identical")
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
