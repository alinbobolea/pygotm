"""Generate pyGOTM's packaged GOTM SAAR grid data.

The source file is GOTM's ``extern/gsw/modules/gsw_mod_saar_data.f90``.
The generated ``.npz`` is consumed by ``pygotm.util.gsw.toolbox.gsw_saar`` so
runtime code does not depend on a local GOTM checkout.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")
_SAAR_BLOCK_PATTERN = re.compile(
    r"data\s+saar_ref\(:,(\d+),(\d+)\)\s*/\s*&(?P<body>.*?)\n\s*/",
    re.S,
)


def _parse_numbers(text: str) -> np.ndarray:
    clean = text.replace("_r8", "").replace("&", " ")
    return np.asarray(
        [
            float(token.replace("D", "E").replace("d", "e"))
            for token in _FLOAT_PATTERN.findall(clean)
        ],
        dtype=np.float64,
    )


def _parse_named_block(text: str, name: str) -> np.ndarray:
    pattern = re.compile(rf"data\s+{name}\s*/\s*&(?P<body>.*?)\n\s*/", re.S)
    match = pattern.search(text)
    if match is None:
        msg = f"could not find {name} in GOTM SAAR data"
        raise ValueError(msg)
    return _parse_numbers(match.group("body"))


def generate_saar_data(source: Path, output: Path) -> None:
    """Parse GOTM's Fortran SAAR data file and write compressed NumPy arrays."""

    text = source.read_text(encoding="utf-8")
    p_ref = _parse_named_block(text, "p_ref")
    lats_ref = _parse_named_block(text, "lats_ref")
    longs_ref = _parse_named_block(text, "longs_ref")
    nz = p_ref.size
    ny = lats_ref.size
    nx = longs_ref.size

    saar_ref = np.empty((nz, ny, nx), dtype=np.float64)
    saar_ref.fill(np.nan)
    count = 0
    for match in _SAAR_BLOCK_PATTERN.finditer(text):
        j = int(match.group(1)) - 1
        i = int(match.group(2)) - 1
        values = _parse_numbers(match.group("body"))
        if values.size != nz:
            msg = f"saar_ref(:,{j + 1},{i + 1}) has {values.size} values, expected {nz}"
            raise ValueError(msg)
        saar_ref[:, j, i] = values
        count += 1
    if count != nx * ny or np.isnan(saar_ref).any():
        msg = f"parsed {count} SAAR columns, expected {nx * ny}"
        raise ValueError(msg)

    ndepth_values = _parse_named_block(text, "ndepth_ref").astype(np.int64)
    ndepth_ref = np.reshape(ndepth_values, (ny, nx), order="F")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        p_ref=p_ref,
        lats_ref=lats_ref,
        longs_ref=longs_ref,
        saar_ref=saar_ref,
        ndepth_ref=ndepth_ref,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Path to gotm-model/code/extern/gsw/modules/gsw_mod_saar_data.f90",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to write saar_2011_gotm.npz",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line generator."""

    args = _parse_args()
    generate_saar_data(args.source, args.output)


if __name__ == "__main__":
    main()
