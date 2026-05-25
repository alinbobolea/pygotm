"""Tests for runtime-to-citation mapping."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from pygotm.citations.extractor import citations_for_config, citations_for_output


def test_citations_for_config_maps_active_modules(tmp_path: Path) -> None:
    (tmp_path / "fabm.yaml").write_text(
        "instances:\n  phy:\n    model: gotm/npzd\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "gotm.yaml"
    config_path.write_text(
        "version: 7\n"
        "surface:\n"
        "  fluxes:\n"
        "    method: kondo\n"
        "  ice:\n"
        "    model: winton\n"
        "light_extinction:\n"
        "  method: jerlov_i\n"
        "turbulence:\n"
        "  turb_method: second_order\n"
        "  tke_method: tke\n"
        "  len_scale_method: omega\n"
        "fabm:\n"
        "  use: true\n"
        "  config_file: fabm.yaml\n",
        encoding="utf-8",
    )

    keys = set(citations_for_config(config_path)["citation_keys"])

    assert {"gotm", "pygotm", "kondo1975", "jerlov1976"}.issubset(keys)
    assert {"wilcox1988", "winton2000", "bruggeman2014", "fasham1990"}.issubset(keys)


def test_citations_for_output_uses_attrs_when_source_yaml_is_absent(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "out.nc"
    dataset = xr.Dataset(
        attrs={
            "source_yaml": str(tmp_path / "missing.yaml"),
            "surface_fluxes_method": "kondo",
            "light_extinction_method": "jerlov_i",
            "turbulence_closure": "k-epsilon",
            "ice_model": "off",
            "fabm_models": "[]",
        }
    )
    dataset.to_netcdf(output_path, engine="scipy")

    keys = set(citations_for_output(output_path)["citation_keys"])

    assert {"gotm", "pygotm", "kondo1975", "jerlov1976", "rodi1987"}.issubset(keys)
