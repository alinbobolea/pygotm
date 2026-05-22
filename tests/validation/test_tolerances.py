"""Tests for validation/tolerances.py Frechet configuration."""

from __future__ import annotations

import pytest

from pygotm.validation.tolerances import (
    DEFAULT_FRECHET_CONFIG,
    PYGOTM_VARIABLES,
    VARIABLE_MAGNITUDE_FLOORS,
    FrechetConfig,
    classify_section,
)


def test_frechet_config_defaults_match_validation_policy() -> None:
    cfg = DEFAULT_FRECHET_CONFIG
    assert cfg.pass_tol == pytest.approx(0.01)
    assert cfg.marginal_tol == pytest.approx(0.05)
    assert cfg.discrepant_tol == pytest.approx(0.20)
    assert cfg.frechet_abs_tol == pytest.approx(1.0e-12)
    assert cfg.frechet_k == 200
    assert cfg.robust is False
    assert cfg.q_low == pytest.approx(0.1)
    assert cfg.q_high == pytest.approx(99.9)
    assert cfg.pyfabm_robust is True
    assert cfg.pyfabm_q_low == pytest.approx(0.1)
    assert cfg.pyfabm_q_high == pytest.approx(99.9)
    assert cfg.peak_frechet_k == 400
    assert cfg.switch_oom == pytest.approx(2.0)
    assert cfg.eps_floor == pytest.approx(1.0e-12)
    assert cfg.default_magnitude_floor == pytest.approx(1.0e-6)


def test_frechet_config_is_frozen() -> None:
    cfg = FrechetConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.pass_tol = 1.0  # type: ignore[misc]


def test_frechet_config_validates_threshold_order() -> None:
    with pytest.raises(ValueError, match="pass < marginal < discrepant"):
        FrechetConfig(pass_tol=0.05, marginal_tol=0.01)


def test_frechet_config_validates_pyfabm_quantiles() -> None:
    with pytest.raises(ValueError, match="pyfabm normalization quantiles"):
        FrechetConfig(pyfabm_q_low=99.9, pyfabm_q_high=0.1)


def test_normalization_settings_are_section_specific() -> None:
    cfg = FrechetConfig()

    assert cfg.normalization_settings("temp") == (False, 0.1, 99.9)
    assert cfg.normalization_settings("oxygen_some_fabm_model") == (
        True,
        0.1,
        99.9,
    )


def test_known_gotm_variable_returns_pygotm_section() -> None:
    for name in ("temp", "salt", "u", "v", "tke", "eps", "num", "nuh"):
        assert classify_section(name) == "pygotm", f"expected pygotm for {name!r}"


def test_known_gotm_diagnostic_variable_returns_pygotm_section() -> None:
    for name in ("mld_surf", "uu", "vv", "ww", "taux", "tauy", "Eturb"):
        assert classify_section(name) == "pygotm", f"expected pygotm for {name!r}"


def test_unknown_variable_returns_pyfabm_section() -> None:
    assert classify_section("oxygen_some_fabm_model") == "pyfabm"


def test_pygotm_variable_set_contains_registered_physics_names() -> None:
    assert {"temp", "salt", "tke", "eps", "u", "v"} <= PYGOTM_VARIABLES


def test_no_per_variable_tolerance_fields_on_config() -> None:
    cfg = FrechetConfig()
    assert not hasattr(cfg, "atol")
    assert not hasattr(cfg, "rtol")
    assert not hasattr(cfg, "scale_floor")


def test_frechet_rel_tol_default() -> None:
    cfg = FrechetConfig()
    assert cfg.frechet_rel_tol == pytest.approx(1.0e-6)


def test_frechet_rel_tol_negative_raises() -> None:
    with pytest.raises(ValueError, match="frechet_rel_tol must be non-negative"):
        FrechetConfig(frechet_rel_tol=-1.0e-7)


def test_variable_magnitude_floors_cover_pygotm_variables() -> None:
    assert PYGOTM_VARIABLES <= set(VARIABLE_MAGNITUDE_FLOORS)
    assert all(floor > 0.0 for floor in VARIABLE_MAGNITUDE_FLOORS.values())


def test_effective_score_uses_dnorm_above_variable_floor() -> None:
    cfg = FrechetConfig()
    score, metric_mode = cfg.effective_score(
        "NN",
        d_raw=3.0e-7,
        d_norm=0.02,
        signal_scale=1.0e-3,
    )
    assert score == pytest.approx(0.02)
    assert metric_mode == "d_norm"


def test_effective_score_uses_drel_below_variable_floor() -> None:
    cfg = FrechetConfig()
    score, metric_mode = cfg.effective_score(
        "NN",
        d_raw=3.0e-9,
        d_norm=0.02,
        signal_scale=1.0e-6,
    )
    assert score == pytest.approx(0.003)
    assert metric_mode == "d_rel"


def test_effective_score_zero_signal_uses_dnorm() -> None:
    cfg = FrechetConfig()
    score, metric_mode = cfg.effective_score(
        "NN",
        d_raw=0.0,
        d_norm=0.0,
        signal_scale=0.0,
    )
    assert score == pytest.approx(0.0)
    assert metric_mode == "d_norm"


def test_effective_score_uses_default_floor_for_missing_variable() -> None:
    cfg = FrechetConfig(default_magnitude_floor=1.0e-5)
    score, metric_mode = cfg.effective_score(
        "oxygen_some_fabm_model",
        d_raw=2.0e-9,
        d_norm=0.02,
        signal_scale=1.0e-6,
    )
    assert score == pytest.approx(0.002)
    assert metric_mode == "d_rel"


def test_default_magnitude_floor_validation() -> None:
    with pytest.raises(ValueError, match="default_magnitude_floor must be positive"):
        FrechetConfig(default_magnitude_floor=0.0)
