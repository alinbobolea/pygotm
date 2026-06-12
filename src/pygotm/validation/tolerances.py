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
    frechet_k: int = 200
    # Core PyGOTM variables use full-range normalization by default. The
    # turbulence fields are often floor-dominated; clipping their active tails
    # with robust percentiles can turn small floor differences into unstable
    # normalized distances.
    robust: bool = False
    q_low: float = 0.1
    q_high: float = 99.9
    # Non-PyGOTM variables, primarily PyFABM, use a wide robust range. This
    # suppresses isolated biogeochemical outliers without hiding broad shape
    # differences.
    pyfabm_robust: bool = True
    pyfabm_q_low: float = 0.1
    pyfabm_q_high: float = 99.9
    peak_frechet_k: int = 400
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
        if not (0.0 <= self.pyfabm_q_low < self.pyfabm_q_high <= 100.0):
            msg = (
                "pyfabm normalization quantiles must satisfy 0 <= q_low < q_high <= 100"
            )
            raise ValueError(msg)
        if self.peak_frechet_k <= 0:
            msg = "peak_frechet_k must be positive"
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

    def normalization_settings(self, name: str) -> tuple[bool, float, float]:
        """Return robust normalization controls for a variable.

        Full-range normalization is reserved for the floor-dominated mean-flow
        and turbulence fields in ``_FULL_RANGE_NORM_VARS``; every other variable
        (native air-sea / ice / lake-hydro fields as well as FABM variables)
        uses the robust quantile range. This is intentionally independent of the
        report-section ownership split so that completing the native ownership
        list does not change any variable's d_norm.
        """

        if name in _FULL_RANGE_NORM_VARS:
            return self.robust, self.q_low, self.q_high
        return self.pyfabm_robust, self.pyfabm_q_low, self.pyfabm_q_high


DEFAULT_FRECHET_CONFIG = FrechetConfig()


# Authoritative set of variables produced natively by pyGOTM (not by the FABM
# coupling). It mirrors every field registered in
# ``pygotm/gotm/register_all_variables.py`` plus the lake hypsography /
# water-balance and air-sea fields emitted by the runtime output writer. Any
# numeric output variable NOT in this set is treated as a FABM-owned variable
# (``classify_section`` returns ``"pyfabm"``). Keep this list complete: a native
# field missing here is mis-reported under the PyFABM section and is normalised
# with the wrong (robust) range. FABM state/diagnostic names (e.g.
# ``selmaprotbas_*``, ``cyanobacteria_*``, ``*_calculator_result``) must never
# appear here.
PYGOTM_VARIABLES: frozenset[str] = frozenset(
    {
        # --- meanflow: mean state, grid, density, stratification ---
        "temp",
        "salt",
        "u",
        "v",
        "w",
        "h",
        "ho",
        "zeta",
        "depth",
        "cori",
        "rho",
        "rho_p",
        "alpha",
        "beta",
        "buoy",
        "NN",
        "NNT",
        "NNS",
        "SS",
        "SSU",
        "SSV",
        "ga",
        "fric",
        "drag",
        "avh",
        "bioshade",
        "rad",
        "xP",
        "idpdx",
        "idpdy",
        # --- bottom/surface friction ---
        "u_taus",
        "u_taub",
        "u_taubo",
        "taub",
        "taux",
        "tauy",
        # --- turbulence (k-eps / second-order closure fields) ---
        "num",
        "nuh",
        "nus",
        "nucl",
        "tke",
        "tkeo",
        "eps",
        "L",
        "kb",
        "epsb",
        "P",
        "G",
        "Pb",
        "PSTK",
        "cmue1",
        "cmue2",
        "cmue3",
        "gam",
        "gamu",
        "gamv",
        "gamb",
        "gamh",
        "gams",
        "an",
        "as",
        "at",
        "r",
        "Rig",
        "xRf",
        "uu",
        "vv",
        "ww",
        # --- Stokes drift ---
        "us",
        "vs",
        "dusdz",
        "dvsdz",
        "us0",
        "vs0",
        "ds",
        # --- ice model ---
        "Hice",
        "Hfrazil",
        "Hsnow",
        "T1",
        "T2",
        "Tf",
        "Tice_surface",
        "albedo_ice",
        "transmissivity",
        "ocean_ice_flux",
        "ocean_ice_heat_flux",
        "ocean_ice_salt_flux",
        "surface_ice_energy",
        "bottom_ice_energy",
        "melt_rate",
        "T_melt",
        "S_melt",
        "dHib",
        "dHis",
        # --- air-sea forcing inputs and computed surface fluxes ---
        "u10",
        "v10",
        "airt",
        "airp",
        "hum",
        "cloud",
        "precip",
        "evap",
        "es",
        "ea",
        "qs",
        "qa",
        "rhoa",
        "shortwave",
        "heat",
        "qh",
        "qe",
        "ql",
        "qlobs",
        "tx",
        "ty",
        "sst",
        "sss",
        "albedo",
        "I_0",
        # --- observation profiles / prescribed inputs echoed to output ---
        "temp_obs",
        "salt_obs",
        "u_obs",
        "v_obs",
        "eps_obs",
        "sst_obs",
        "zeta_obs",
        "dpdx",
        "dpdy",
        # --- energetics and integrated surface diagnostics ---
        "Ekin",
        "Epot",
        "Eturb",
        "mld_surf",
        "mld_bott",
        "int_swr",
        "int_heat",
        "int_total",
        "int_precip",
        "int_evap",
        "int_fwf",
        # --- lake hypsography / water balance (native lake feature) ---
        "Af",
        "Qlayer",
        "Qs",
        "wq",
        "FQ",
        "Qres",
        "Qt",
        "int_flow",
        "int_inflow",
        "int_outflow",
        "int_water_balance",
        # Lake inflow streams are written with user-defined names from the run
        # config (here, the lake_erken tributaries). A fully general classifier
        # would source these names from the active stream list; they are listed
        # explicitly so the reference suite reports them as native.
        "Q_Kristine",
        "Q_Stensta",
        "Q_Unguaged",
        "T_Kristine",
        "T_Unguaged",
    }
)


# Variables that use full-range (non-robust) normalization. These are the
# floor-dominated mean-flow / turbulence fields for which clipping the active
# tail with robust percentiles turns small floor differences into unstable
# normalized distances (see ``FrechetConfig.robust``). Reporting ownership
# (``classify_section``) is deliberately decoupled from this normalization
# choice: native air-sea / ice / lake-hydro fields are reported under the
# PyGOTM section but keep the robust range that suits their statistics, so the
# ownership correction does not alter any variable's d_norm or status.
_FULL_RANGE_NORM_VARS: frozenset[str] = frozenset(
    {
        "temp",
        "salt",
        "u",
        "v",
        "h",
        "rho",
        "rho_p",
        "buoy",
        "NN",
        "NNT",
        "NNS",
        "SS",
        "Rig",
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
        "uu",
        "vv",
        "ww",
        "xP",
        "fric",
        "drag",
        "taub",
        "taux",
        "tauy",
        "I_0",
        "bioshade",
        "PSTK",
        "idpdy",
        "idpdx",
        "w",
        "mld_surf",
        "mld_bott",
        "Ekin",
        "Epot",
        "Eturb",
        "int_swr",
        "int_heat",
        "int_total",
        "int_precip",
        "int_evap",
    }
)


VARIABLE_MAGNITUDE_FLOORS: dict[str, float] = {
    "temp": 1.0e-2,
    "salt": 1.0e-2,
    "u": 1.0e-4,
    "v": 1.0e-4,
    "h": 1.0e-3,
    "rho": 1.0e-2,
    "rho_p": 1.0e-2,
    "buoy": 1.0e-5,
    "NN": 1.0e-4,
    "NNT": 1.0e-4,
    "NNS": 1.0e-4,
    "SS": 1.0e-4,
    "Rig": 1.0e-6,
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
    "uu": 1.0e-8,
    "vv": 1.0e-8,
    "ww": 1.0e-8,
    "xP": 1.0e-12,
    "fric": 1.0e-6,
    "drag": 1.0e-6,
    "taub": 1.0e-6,
    "taux": 1.0e-6,
    "tauy": 1.0e-6,
    "I_0": 1.0e-2,
    "bioshade": 1.0e-4,
    "PSTK": 1.0e-8,
    "idpdy": 1.0e-8,
    "idpdx": 1.0e-8,
    "w": 1.0e-8,
    "mld_surf": 1.0e-3,
    "mld_bott": 1.0e-3,
    "Ekin": 1.0e-6,
    "Epot": 1.0e-6,
    "Eturb": 1.0e-6,
    "int_swr": 1.0e-6,
    "int_heat": 1.0e-6,
    "int_total": 1.0e-6,
    "int_precip": 1.0e-9,
    "int_evap": 1.0e-9,
}


def classify_section(name: str) -> Literal["pygotm", "pyfabm"]:
    """Return 'pygotm' for known GOTM physics variables, 'pyfabm' otherwise."""

    return "pygotm" if name in PYGOTM_VARIABLES else "pyfabm"
