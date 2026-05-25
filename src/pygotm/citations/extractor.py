"""Map active pyGOTM runtime features to curated bibliography entries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import xarray as xr

from pygotm.config import GotmConfig, load_config
from pygotm.gotm.run_metadata import derive_turbulence_closure, parse_fabm_models

__all__ = [
    "BIB_PATH",
    "citation_records",
    "citation_keys_for_document",
    "citations_for_config",
    "citations_for_output",
    "load_bibliography",
]

BIB_PATH = Path(__file__).with_name("pygotm.bib")


def load_bibliography(path: Path = BIB_PATH) -> Any:
    """Parse the curated BibTeX database with pybtex."""

    try:
        from pybtex.database.input import bibtex
    except ImportError as exc:  # pragma: no cover - environment guard.
        msg = "pybtex is required to parse pyGOTM citation data"
        raise RuntimeError(msg) from exc

    parser = bibtex.Parser()
    return parser.parse_file(str(path))


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _token(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip().lower().replace("-", "_") or default


def _add(keys: set[str], *values: str) -> None:
    keys.update(value for value in values if value)


def _fabm_config_path(config: GotmConfig, document: Mapping[str, Any]) -> Path:
    fabm = _mapping(document.get("fabm"))
    configured = (
        fabm.get("config")
        or fabm.get("config_file")
        or fabm.get("yaml")
        or fabm.get("file")
        or "fabm.yaml"
    )
    path = Path(str(configured))
    if not path.is_absolute() and config.source_dir is not None:
        path = config.source_dir / path
    return path


def _citation_keys_for_fabm_models(models: Iterable[str]) -> set[str]:
    keys = {"bruggeman2014"}
    for model in models:
        normalized = model.lower()
        if "npzd" in normalized:
            keys.add("fasham1990")
        if "jrc_med_ergom" in normalized or "ergom" in normalized:
            keys.add("neumann2002")
        if "bsem" in normalized:
            keys.add("bruggeman2005")
    return keys


def _citation_keys_for_document(
    document: Mapping[str, Any],
    *,
    fabm_models: Iterable[str] = (),
) -> set[str]:
    keys: set[str] = {"gotm", "pygotm"}

    surface = _mapping(document.get("surface"))
    fluxes = _mapping(surface.get("fluxes"))
    flux_method = _token(fluxes.get("method"), "off")
    if flux_method == "kondo":
        keys.add("kondo1975")
    if flux_method == "fairall":
        _add(keys, "fairall1996a", "fairall1996b", "liu1979")

    light = _mapping(document.get("light_extinction"))
    light_method = _token(light.get("method"), "jerlov_i")
    if light_method.startswith("jerlov"):
        keys.add("jerlov1976")

    turbulence = _mapping(document.get("turbulence"))
    turb_method = _token(turbulence.get("turb_method"), "second_order")
    tke_method = _token(turbulence.get("tke_method"), "tke")
    len_scale = _token(turbulence.get("len_scale_method"), "dissipation")
    closure = derive_turbulence_closure(
        turbulence_method=turb_method,
        tke_method=tke_method,
        len_scale_method=len_scale,
    )
    if closure == "k-epsilon":
        keys.add("rodi1987")
    if closure == "k-omega":
        keys.add("wilcox1988")
    if closure == "GLS":
        keys.add("umlauf2003")
    if closure == "Mellor-Yamada":
        keys.add("mellor1982")

    scnd = _mapping(turbulence.get("scnd"))
    scnd_coeff = _token(scnd.get("scnd_coeff"))
    if "canuto" in scnd_coeff:
        keys.add("canuto2001")
    if "cheng" in scnd_coeff:
        keys.add("cheng2002")
    if "kantha" in scnd_coeff or "clayson" in scnd_coeff:
        keys.add("kantha1994")

    bc = _mapping(turbulence.get("bc"))
    if _token(bc.get("ubc_type")) == "tke_injection":
        keys.add("craig1994")

    ice = _mapping(surface.get("ice"))
    ice_model = _token(ice.get("model"), "no_ice")
    if ice_model == "lebedev":
        keys.add("lebedev1938")
    if ice_model == "mylake":
        keys.add("saloranta2007")
    if ice_model == "winton":
        keys.add("winton2000")
    if ice_model == "basal_melt":
        keys.add("holland2008")

    fabm = _mapping(document.get("fabm"))
    if bool(fabm.get("use", False)) or tuple(fabm_models):
        keys.update(_citation_keys_for_fabm_models(fabm_models))

    return keys


def citation_keys_for_document(config: GotmConfig) -> tuple[str, ...]:
    """Return sorted citation keys for a GOTM configuration."""

    document = config.materialize_document()
    models: tuple[str, ...] = ()
    if bool(_mapping(document.get("fabm")).get("use", False)):
        models = parse_fabm_models(_fabm_config_path(config, document))
    return tuple(sorted(_citation_keys_for_document(document, fabm_models=models)))


def _records_for_keys(keys: Iterable[str]) -> list[dict[str, object]]:
    bibliography = load_bibliography()
    records: list[dict[str, object]] = []
    for key in sorted(set(keys)):
        entry = bibliography.entries.get(key)
        if entry is None:
            continue
        fields = entry.fields
        records.append(
            {
                "key": key,
                "type": entry.type,
                "title": fields.get("title", ""),
                "year": fields.get("year", ""),
                "doi": fields.get("doi", ""),
                "url": fields.get("url", ""),
            }
        )
    return records


def citation_records(keys: Iterable[str] | None = None) -> dict[str, object]:
    """Return JSON-serializable citation records."""

    bibliography = load_bibliography()
    selected = tuple(bibliography.entries) if keys is None else tuple(keys)
    return {
        "citation_keys": sorted(set(selected)),
        "citations": _records_for_keys(selected),
    }


def citations_for_config(config_path: str | Path) -> dict[str, object]:
    """Return citation records for the modules active in *config_path*."""

    config = load_config(config_path)
    return citation_records(citation_keys_for_document(config))


def _keys_from_attrs(attrs: Mapping[Any, Any]) -> set[str]:
    keys: set[str] = {"gotm", "pygotm"}

    flux_method = _token(
        attrs.get("surface_fluxes_method")
        or attrs.get("airsea_fluxes_method")
        or attrs.get("fluxes_method"),
        "off",
    )
    if flux_method == "kondo":
        keys.add("kondo1975")
    if flux_method == "fairall":
        _add(keys, "fairall1996a", "fairall1996b", "liu1979")

    light_method = _token(
        attrs.get("light_extinction_method") or attrs.get("light_method"),
    )
    if light_method.startswith("jerlov"):
        keys.add("jerlov1976")

    closure = _token(attrs.get("turbulence_closure"))
    if closure == "k_epsilon":
        keys.add("rodi1987")
    if closure == "k_omega":
        keys.add("wilcox1988")
    if closure == "gls":
        keys.add("umlauf2003")
    if closure == "mellor_yamada":
        keys.add("mellor1982")
    ice_model = _token(attrs.get("ice_model"), "off")
    if ice_model == "lebedev":
        keys.add("lebedev1938")
    if ice_model == "mylake":
        keys.add("saloranta2007")
    if ice_model == "winton":
        keys.add("winton2000")
    if ice_model == "basal_melt":
        keys.add("holland2008")
    try:
        models = json.loads(str(attrs.get("fabm_models", "[]")))
    except json.JSONDecodeError:
        models = []
    if isinstance(models, list) and models:
        keys.update(_citation_keys_for_fabm_models(str(model) for model in models))
    return keys


def citations_for_output(output_path: str | Path) -> dict[str, object]:
    """Return citation records for a pyGOTM NetCDF output."""

    with xr.open_dataset(output_path, engine="scipy") as dataset:
        attrs = dict(dataset.attrs)
    source_yaml = Path(str(attrs.get("source_yaml", "")))
    if source_yaml.is_file():
        config = load_config(source_yaml)
        return citation_records(citation_keys_for_document(config))
    return citation_records(_keys_from_attrs(attrs))
