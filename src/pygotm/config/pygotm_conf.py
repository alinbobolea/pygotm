"""pygotm-conf.yaml loader — per-case pyGOTM configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["FABMConf", "PyGotmConf", "load_pygotm_conf"]


@dataclass
class FABMConf:
    chunk_size: int | None = None  # None → use default (one day in physics steps)


@dataclass
class PyGotmConf:
    fabm: FABMConf = field(default_factory=FABMConf)


def load_pygotm_conf(yaml_path: Path) -> PyGotmConf:
    """Load pygotm-conf.yaml from the same directory as gotm.yaml.

    Returns default PyGotmConf if the file does not exist.
    If the file exists but has no 'fabm' key, fabm defaults apply.
    """
    conf_path = yaml_path.parent / "pygotm-conf.yaml"
    if not conf_path.is_file():
        return PyGotmConf()
    with conf_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    fabm_raw = raw.get("fabm", {}) or {}
    chunk_size = fabm_raw.get("chunk_size", None)
    if chunk_size is not None:
        chunk_size = int(chunk_size)
        if chunk_size < 1:
            msg = f"fabm.chunk_size must be >= 1, got {chunk_size}"
            raise ValueError(msg)
    return PyGotmConf(fabm=FABMConf(chunk_size=chunk_size))
