"""Tests for config schema overlays."""

from __future__ import annotations

from pygotm.gotm.run_metadata import PYGOTM_CONFIG_SCHEMA_VERSION
from pygotm.schema import config_schema


def test_config_schema_includes_free_form_section_overlays() -> None:
    schema = config_schema()
    properties = schema["properties"]

    assert schema["x-pygotm-schema-version"] == PYGOTM_CONFIG_SCHEMA_VERSION
    assert "fluxes" in properties["surface"]["properties"]
    assert "ice" in properties["surface"]["properties"]
    assert "load" in properties["restart"]["properties"]
    assert "time_method" in properties["output"]["additionalProperties"]["properties"]
    assert "linear" in properties["equation_of_state"]["properties"]
    assert "feedbacks" in properties["fabm"]["properties"]
    assert "turb_method" in properties["turbulence"]["properties"]
    assert "tke_method" in properties["turbulence"]["properties"]
    assert "len_scale_method" in properties["turbulence"]["properties"]
    assert "scnd" in properties["turbulence"]["properties"]
    assert "bc" in properties["turbulence"]["properties"]
