"""Tests for GOTM-bundled GSW salinity conversions."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.util.gsw import gsw_sa_from_sp, gsw_saar, gsw_sp_from_sa


def test_gsw_saar_matches_bundled_flex_reference_point() -> None:
    """SAAR uses GOTM's bundled 2011 grid, not the installed gsw package grid."""

    saar = gsw_saar(76.5, 0.32, 58.91666)

    assert saar == pytest.approx(3.652974094910443e-05, rel=0.0, abs=1.0e-16)


def test_gsw_salinity_conversion_is_vectorized_and_reversible() -> None:
    pressure = np.asarray([0.0, 76.5, 101.5], dtype=np.float64)
    practical = np.asarray([35.0, 35.11033248901367, 35.2], dtype=np.float64)

    absolute = gsw_sa_from_sp(practical, pressure, 0.32, 58.91666)

    assert absolute[1] == pytest.approx(35.27718137320878, rel=0.0, abs=1.0e-14)
    np.testing.assert_allclose(
        gsw_sp_from_sa(absolute, pressure, 0.32, 58.91666),
        practical,
        rtol=0.0,
        atol=1.0e-14,
    )
