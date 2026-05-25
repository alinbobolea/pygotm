"""Run provenance helpers for NetCDF global attributes and Studio manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from pygotm.config import GotmConfig
from pygotm.gotm.print_version import collect_version_info

yaml: Any = import_module("yaml")

PYGOTM_CONFIG_SCHEMA_VERSION = "gotm-6.x-pygotm-1"
PYGOTM_OUTPUT_SCHEMA_VERSION = "1"

REQUIRED_NETCDF_ATTRS: tuple[str, ...] = (
    "title",
    "source_yaml",
    "nlev",
    "dt",
    "runtime",
    "pygotm_version",
    "pygotm_git_commit",
    "pygotm_config_schema_version",
    "pygotm_output_schema_version",
    "python_version",
    "numpy_version",
    "numba_version",
    "xarray_version",
    "netcdf4_version",
    "gsw_version",
    "pyfabm_version",
    "platform",
    "source_yaml_sha256",
    "effective_yaml_sha256",
    "fabm_yaml_sha256",
    "started_at",
    "finished_at",
    "wall_clock_seconds",
    "turbulence_method",
    "tke_method",
    "len_scale_method",
    "turbulence_closure",
    "ice_model",
    "fabm_active",
    "fabm_models",
)

__all__ = [
    "PYGOTM_CONFIG_SCHEMA_VERSION",
    "PYGOTM_OUTPUT_SCHEMA_VERSION",
    "REQUIRED_NETCDF_ATTRS",
    "ConfigHashes",
    "build_run_attrs",
    "canonical_materialized_yaml_bytes",
    "collect_config_hashes",
    "derive_turbulence_closure",
    "file_sha256",
    "parse_fabm_models",
    "utc_timestamp",
]


@dataclass(frozen=True, slots=True)
class ConfigHashes:
    """Portable and source-file hashes for one GOTM configuration."""

    source_yaml_sha256: str
    effective_yaml_sha256: str


def utc_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp for run provenance."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path | None) -> str:
    """Return the SHA-256 of *path* bytes, or ``"unavailable"``."""

    if path is None:
        return "unavailable"
    resolved = Path(path)
    if not resolved.is_file():
        return "unavailable"
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_materialized_yaml_bytes(document: Mapping[str, Any]) -> bytes:
    """Serialize a materialized config document for portable hashing."""

    text: str = yaml.safe_dump(
        dict(document),
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    )
    return text.encode("utf-8")


def collect_config_hashes(config: GotmConfig) -> ConfigHashes:
    """Return raw-source and portable effective-YAML hashes for *config*."""

    effective_bytes = canonical_materialized_yaml_bytes(config.materialize_document())
    return ConfigHashes(
        source_yaml_sha256=file_sha256(config.source_path),
        effective_yaml_sha256=hashlib.sha256(effective_bytes).hexdigest(),
    )


def parse_fabm_models(path: str | Path | None) -> tuple[str, ...]:
    """Return FABM ``instances.<name>.model`` values from a FABM YAML file."""

    if path is None:
        return ()
    fabm_path = Path(path)
    if not fabm_path.is_file():
        return ()
    raw = yaml.safe_load(fabm_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return ()
    instances = raw.get("instances")
    if not isinstance(instances, dict):
        return ()

    models: list[str] = []
    for value in instances.values():
        if not isinstance(value, dict):
            continue
        model = value.get("model")
        if model is not None:
            models.append(str(model))
    return tuple(models)


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _token(value: object, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower().replace("-", "_")
    return text or default


def derive_turbulence_closure(
    *,
    turbulence_method: str,
    tke_method: str,
    len_scale_method: str,
) -> str:
    """Derive a stable display label from GOTM turbulence method tokens."""

    if turbulence_method in {"off", "no_model"}:
        return "off"
    if turbulence_method == "first_order":
        return "first-order"
    if tke_method == "mellor_yamada" or len_scale_method == "mellor_yamada":
        return "Mellor-Yamada"
    if len_scale_method == "omega":
        return "k-omega"
    if len_scale_method == "gls":
        return "GLS"
    if len_scale_method == "dissipation":
        return "k-epsilon"
    return f"{turbulence_method}/{tke_method}/{len_scale_method}"


def _ice_model(document: Mapping[str, Any]) -> str:
    surface = _mapping(document.get("surface"))
    surface_ice = _mapping(surface.get("ice"))
    root_ice = _mapping(document.get("ice"))
    model = _token(surface_ice.get("model", root_ice.get("model")), "off")
    return "off" if model in {"no_ice", "none", "false"} else model


def _turbulence_tokens(document: Mapping[str, Any]) -> tuple[str, str, str]:
    turbulence = _mapping(document.get("turbulence"))
    return (
        _token(turbulence.get("turb_method"), "second_order"),
        _token(turbulence.get("tke_method"), "tke"),
        _token(turbulence.get("len_scale_method"), "dissipation"),
    )


def _fabm_metadata(run: Any) -> tuple[str, str, str]:
    fabm_config = getattr(run, "fabm_config", None)
    fabm_active = bool(fabm_config is not None and getattr(fabm_config, "use", False))
    config_path = None if fabm_config is None else fabm_config.config_path
    return (
        "true" if fabm_active else "false",
        file_sha256(config_path) if fabm_active else "unavailable",
        json.dumps(list(parse_fabm_models(config_path)), separators=(",", ":")),
    )


def build_run_attrs(
    run: Any,
    *,
    source_yaml_sha256: str,
    effective_yaml_sha256: str,
    started_at: str,
    finished_at: str,
    wall_clock_seconds: float,
    runtime: str = "compiled",
) -> dict[str, str | int | float]:
    """Build the stable NetCDF global-attribute contract for one run."""

    document = _mapping(getattr(run, "document", {}))
    turbulence_method, tke_method, len_scale_method = _turbulence_tokens(document)
    fabm_active, fabm_yaml_sha256, fabm_models = _fabm_metadata(run)
    version_info = collect_version_info()
    attrs: dict[str, str | int | float] = {
        "title": str(run.settings.title),
        "source_yaml": str(run.yaml_path),
        "nlev": int(run.nlev),
        "dt": float(run.dt),
        "runtime": runtime,
        "pygotm_version": version_info["pygotm_version"],
        "pygotm_git_commit": version_info["pygotm_git_commit"],
        "pygotm_config_schema_version": PYGOTM_CONFIG_SCHEMA_VERSION,
        "pygotm_output_schema_version": PYGOTM_OUTPUT_SCHEMA_VERSION,
        "python_version": version_info["python_version"],
        "numpy_version": version_info["numpy_version"],
        "numba_version": version_info["numba_version"],
        "xarray_version": version_info["xarray_version"],
        "netcdf4_version": version_info["netcdf4_version"],
        "gsw_version": version_info["gsw_version"],
        "pyfabm_version": version_info["pyfabm_version"],
        "platform": version_info["platform"],
        "source_yaml_sha256": source_yaml_sha256,
        "effective_yaml_sha256": effective_yaml_sha256,
        "fabm_yaml_sha256": fabm_yaml_sha256,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_seconds": float(wall_clock_seconds),
        "turbulence_method": turbulence_method,
        "tke_method": tke_method,
        "len_scale_method": len_scale_method,
        "turbulence_closure": derive_turbulence_closure(
            turbulence_method=turbulence_method,
            tke_method=tke_method,
            len_scale_method=len_scale_method,
        ),
        "ice_model": _ice_model(document),
        "fabm_active": fabm_active,
        "fabm_models": fabm_models,
    }
    missing = set(REQUIRED_NETCDF_ATTRS).difference(attrs)
    if missing:
        names = ", ".join(sorted(missing))
        msg = f"run metadata missing required NetCDF attrs: {names}"
        raise RuntimeError(msg)
    return attrs
