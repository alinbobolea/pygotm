"""Validation configuration and section classification for Frechet comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DEFAULT_FRECHET_CONFIG",
    "PYGOTM_VARIABLES",
    "VARIABLE_MAGNITUDE_FLOORS",
    "FrechetConfig",
    "classify_section",
]


@dataclass(frozen=True)
class FrechetConfig:
    """Thresholds and normalization controls for validation Frechet distance."""

    pass_tol: float = 0.01
    marginal_tol: float = 0.05
    discrepant_tol: float = 0.20
    frechet_abs_tol: float = 1.0e-12
    frechet_rel_tol: float = 1.0e-6
    frechet_k: int = 400
    robust: bool = True
    q_low: float = 1.0
    q_high: float = 99.0
    switch_oom: float = 2.0
    eps_floor: float = 1.0e-12
    default_magnitude_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        if not (0.0 < self.pass_tol < self.marginal_tol < self.discrepant_tol):
            msg = "Frechet status thresholds must satisfy pass < marginal < discrepant"
            raise ValueError(msg)
        if self.frechet_abs_tol < 0.0:
            msg = "frechet_abs_tol must be non-negative"
            raise ValueError(msg)
        if self.frechet_rel_tol < 0.0:
            msg = "frechet_rel_tol must be non-negative"
            raise ValueError(msg)
        if self.frechet_k <= 0:
            msg = "frechet_k must be positive"
            raise ValueError(msg)
        if not (0.0 <= self.q_low < self.q_high <= 100.0):
            msg = "normalization quantiles must satisfy 0 <= q_low < q_high <= 100"
            raise ValueError(msg)
        if self.switch_oom < 0.0:
            msg = "switch_oom must be non-negative"
            raise ValueError(msg)
        if self.eps_floor <= 0.0:
            msg = "eps_floor must be positive"
            raise ValueError(msg)
        if self.default_magnitude_floor <= 0.0:
            msg = "default_magnitude_floor must be positive"
            raise ValueError(msg)

    def effective_score(
        self,
        name: str,
        d_raw: float,
        d_norm: float,
        signal_scale: float,
    ) -> tuple[float, Literal["d_norm", "d_rel"]]:
        """Return the Frechet score and metric mode used for classification."""

        floor = VARIABLE_MAGNITUDE_FLOORS.get(name, self.default_magnitude_floor)
        if 0.0 < signal_scale < floor:
            return d_raw / signal_scale, "d_rel"
        return d_norm, "d_norm"


DEFAULT_FRECHET_CONFIG = FrechetConfig()


PYGOTM_VARIABLES: frozenset[str] = frozenset(
    {
        "temp",
        "salt",
        "u",
        "v",
        "h",
        "rho",
        "buoy",
        "NN",
        "SS",
        "ga",
        "tke",
        "eps",
        "num",
        "nuh",
        "nus",
        "nucl",
        "L",
        "P",
        "G",
        "Pb",
        "kb",
        "epsb",
        "an",
        "cmue1",
        "cmue2",
        "as",
        "at",
        "avh",
        "xP",
        "fric",
        "drag",
        "taub",
        "I_0",
        "bioshade",
        "PSTK",
        "idpdy",
        "idpdx",
        "w",
    }
)


VARIABLE_MAGNITUDE_FLOORS: dict[str, float] = {
    "temp": 1.0e-2,
    "salt": 1.0e-2,
    "u": 1.0e-4,
    "v": 1.0e-4,
    "h": 1.0e-3,
    "rho": 1.0e-2,
    "buoy": 1.0e-5,
    "NN": 1.0e-4,
    "SS": 1.0e-4,
    "ga": 1.0e-8,
    "tke": 1.0e-8,
    "eps": 1.0e-12,
    "num": 1.0e-7,
    "nuh": 1.0e-7,
    "nus": 1.0e-7,
    "nucl": 1.0e-7,
    "avh": 1.0e-7,
    "L": 1.0e-4,
    "P": 1.0e-12,
    "G": 1.0e-12,
    "Pb": 1.0e-12,
    "kb": 1.0e-8,
    "epsb": 1.0e-12,
    "an": 1.0e-6,
    "cmue1": 1.0e-6,
    "cmue2": 1.0e-6,
    "as": 1.0e-6,
    "at": 1.0e-6,
    "xP": 1.0e-12,
    "fric": 1.0e-6,
    "drag": 1.0e-6,
    "taub": 1.0e-6,
    "I_0": 1.0e-2,
    "bioshade": 1.0e-4,
    "PSTK": 1.0e-8,
    "idpdy": 1.0e-8,
    "idpdx": 1.0e-8,
    "w": 1.0e-8,
}


def classify_section(name: str) -> Literal["pygotm", "pyfabm"]:
    """Return 'pygotm' for known GOTM physics variables, 'pyfabm' otherwise."""

    return "pygotm" if name in PYGOTM_VARIABLES else "pyfabm"
