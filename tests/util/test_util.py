"""Tests for pygotm.util.util — GOTM named parameter constants."""

from pygotm.util import util
from pygotm.util.util import (
    CENTRAL,
    Dirichlet,
    MUSCL,
    Neumann,
    P1,
    P2,
    P2_PDM,
    SPLMAX13,
    Superbee,
    UPSTREAM,
    flux,
    oneSided,
    value,
    zeroDivergence,
)


# ---------------------------------------------------------------------------
# Import / smoke
# ---------------------------------------------------------------------------


def test_import() -> None:
    assert util is not None


def test_all_exported() -> None:
    expected = {
        "CENTRAL",
        "UPSTREAM",
        "P1",
        "P2",
        "Superbee",
        "MUSCL",
        "P2_PDM",
        "SPLMAX13",
        "Dirichlet",
        "Neumann",
        "flux",
        "value",
        "oneSided",
        "zeroDivergence",
    }
    assert set(util.__all__) == expected


# ---------------------------------------------------------------------------
# Advection scheme constants — exact values from util.F90
# ---------------------------------------------------------------------------


def test_central() -> None:
    assert CENTRAL == -1


def test_upstream() -> None:
    assert UPSTREAM == 1


def test_p1() -> None:
    assert P1 == 2


def test_p2() -> None:
    assert P2 == 3


def test_superbee() -> None:
    assert Superbee == 4


def test_muscl() -> None:
    assert MUSCL == 5


def test_p2_pdm() -> None:
    assert P2_PDM == 6


def test_splmax13() -> None:
    assert SPLMAX13 == 13


# ---------------------------------------------------------------------------
# Diffusion boundary condition constants
# ---------------------------------------------------------------------------


def test_dirichlet() -> None:
    assert Dirichlet == 0


def test_neumann() -> None:
    assert Neumann == 1


# ---------------------------------------------------------------------------
# Advection boundary condition constants
# ---------------------------------------------------------------------------


def test_flux() -> None:
    assert flux == 1


def test_value() -> None:
    assert value == 2


def test_one_sided() -> None:
    assert oneSided == 3


def test_zero_divergence() -> None:
    assert zeroDivergence == 4


# ---------------------------------------------------------------------------
# Uniqueness — no two different scheme types share a value
# ---------------------------------------------------------------------------


def test_advection_schemes_unique() -> None:
    schemes = [CENTRAL, UPSTREAM, P1, P2, Superbee, MUSCL, P2_PDM, SPLMAX13]
    assert len(schemes) == len(set(schemes))


def test_diffusion_bc_unique() -> None:
    assert Dirichlet != Neumann


def test_advection_bc_unique() -> None:
    bc = [flux, value, oneSided, zeroDivergence]
    assert len(bc) == len(set(bc))


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------


def test_all_constants_are_int() -> None:
    constants = [
        CENTRAL,
        UPSTREAM,
        P1,
        P2,
        Superbee,
        MUSCL,
        P2_PDM,
        SPLMAX13,
        Dirichlet,
        Neumann,
        flux,
        value,
        oneSided,
        zeroDivergence,
    ]
    for c in constants:
        assert isinstance(c, int), f"{c} is not an int"
