"""Refresh vendored intersphinx inventory files.

The ``docs/_intersphinx/*.inv`` files are committed so that ``sphinx-build -W``
works in restricted-network environments (CI sandboxes, offline mirrors). Run
this script when intersphinx-resolved references go stale.

Usage::

    conda run -n pygotm python docs/refresh_intersphinx.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

INVENTORIES: dict[str, str] = {
    "python.inv": "https://docs.python.org/3/objects.inv",
    "numpy.inv": "https://numpy.org/doc/stable/objects.inv",
    "scipy.inv": "https://docs.scipy.org/doc/scipy/objects.inv",
    "xarray.inv": "https://docs.xarray.dev/en/stable/objects.inv",
}


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "_intersphinx"
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (pygotm docs refresh)"}
    for name, url in INVENTORIES.items():
        print(f"fetching {url}")
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        (out_dir / name).write_bytes(data)
        print(f"  wrote {len(data)} bytes to {out_dir / name}")


if __name__ == "__main__":
    main()
