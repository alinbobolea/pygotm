"""
Equation of state (density) — translation of ``density.F90``.

Computes density :math:`\\bar{\\rho}(S, \\Theta, P)`, potential density, and
buoyancy expansion coefficients from salinity, temperature, and pressure.
Three methods are supported, selected via ``DensityState.density_method``:

* ``METHOD_TEOS10`` (1) — Full TEOS-10 equation of state via the ``gsw``
  package: ``gsw_rho``, ``gsw_sigma0``, ``gsw_alpha``, ``gsw_beta``.
* ``METHOD_LINEAR_TEOS10`` (2) — Linearised EOS; thermal expansion
  :math:`\\alpha_0` and haline contraction :math:`\\beta_0` are computed from
  TEOS-10 at the user reference point :math:`(S_0, T_0, p_0)`.
* ``METHOD_LINEAR_USER`` (3) — Linearised EOS with user-supplied
  :math:`\\rho_0`, :math:`\\alpha_0`, :math:`\\beta_0`.

The GOTM Fortran source (``density.F90``) uses the TEOS-10 Gibbs SeaWater
toolbox (``gsw``) rather than the original UNESCO 1983 polynomial described in
the historical module header.  This Python translation wraps the same TEOS-10
library via the ``gsw`` Python package (McDougall & Barker 2011).

TEOS-10 reference: IOC, SCOR and IAPSO (2010). *The International
Thermodynamic Equation of Seawater 2010.* Intergovernmental Oceanographic
Commission, Manuals and Guides No. 56.

Public interface: :func:`init_density`, :func:`do_density`, :func:`get_rho`,
:func:`get_alpha`, :func:`get_beta`, :func:`clean_density`,
:class:`DensityState`.

Original FORTRAN authors: Hans Burchard, Karsten Bolding.
"""

from __future__ import annotations

import math

import gsw as _gsw
import numpy as np

# TEOS-10 standard specific heat of seawater at SA=0, CT=0, p=0.
# Source: IOC, SCOR and IAPSO (2010), TEOS-10 monograph, Appendix K.
# Matches Fortran gsw_mod_teos10_constants::gsw_cp0.
CP0: float = 3991.86795711963

__all__ = [
    "CP0",
    "DensityState",
    "METHOD_JACKETT_FULL",
    "METHOD_JACKETT_POTENTIAL",
    "METHOD_LINEAR_TEOS10",
    "METHOD_LINEAR_USER",
    "METHOD_TEOS10",
    "METHOD_UNESCO_FULL",
    "METHOD_UNESCO_POTENTIAL",
    "clean_density",
    "do_density",
    "get_alpha",
    "get_beta",
    "get_rho",
    "init_density",
    "jackett_density",
    "unesco_density",
]

# Density method codes matching GOTM Fortran density_method variable
METHOD_TEOS10 = 1  # Full TEOS-10 equation of state (gsw_rho / gsw_sigma0)
METHOD_LINEAR_TEOS10 = 2  # Linearised EOS; α, β computed from TEOS-10 at (S0,T0,p0)
METHOD_LINEAR_USER = 3  # Linearised EOS with user-supplied ρ₀, α₀, β₀
METHOD_UNESCO_FULL = 101
METHOD_UNESCO_POTENTIAL = 102
METHOD_JACKETT_FULL = 201
METHOD_JACKETT_POTENTIAL = 202


def unesco_density(S: float, T: float, p: float, use_pressure: bool) -> float:
    """Compute GOTM-Lake's UNESCO density polynomial.

    This is a direct translation of ``gotm-lake/src/util/eqstate.F90``'s
    ``unesco`` routine.  Pressure ``p`` is in bar, matching that Fortran path.
    """

    T2 = T * T
    T3 = T * T2
    T4 = T2 * T2
    T5 = T * T4
    S2 = S * S
    S3 = S * S2
    S15 = math.sqrt(S3) if S3 >= 0.0 else math.nan

    x = 999.842594 + 6.793952e-02 * T - 9.09529e-03 * T2 + 1.001685e-04 * T3
    x = x - 1.120083e-06 * T4 + 6.536332e-09 * T5
    x = x + S * (0.824493 - 4.0899e-03 * T + 7.6438e-05 * T2 - 8.2467e-07 * T3)
    x = x + S * 5.3875e-09 * T4
    x = x + S15 * (-5.72466e-03 + 1.0227e-04 * T - 1.6546e-06 * T2)
    x = x + 4.8314e-04 * S2

    if use_pressure and p > 0.0:
        p2 = p * p
        k = (
            19652.21
            + 148.4206 * T
            - 2.327105 * T2
            + 1.360477e-2 * T3
            - 5.155288e-5 * T4
            + 3.239908 * p
            + 1.43713e-3 * T * p
            + 1.16092e-4 * T2 * p
            - 5.77905e-7 * T3 * p
            + 8.50935e-5 * p2
            - 6.12293e-6 * T * p2
            + 5.2787e-8 * T2 * p2
            + 54.6746 * S
            - 0.603459 * T * S
            + 1.09987e-2 * T2 * S
            - 6.1670e-5 * T3 * S
            + 7.944e-2 * S15
            + 1.6483e-2 * T * S15
            - 5.3009e-4 * T2 * S15
            + 2.2838e-3 * p * S
            - 1.0981e-5 * T * p * S
            - 1.6078e-6 * T2 * p * S
            + 1.91075e-4 * p * S15
            - 9.9348e-7 * p2 * S
            + 2.0816e-8 * T * p2 * S
            + 9.1697e-10 * T2 * p2 * S
        )
        x = x / (1.0 - p / k)

    return float(x)


def jackett_density(S: float, T: float, p: float, use_pressure: bool) -> float:
    """Compute GOTM-Lake's Jackett/Feistel density polynomial.

    This translates ``rho_feistel`` from ``gotm-lake/src/util/eqstate.F90``.
    Pressure ``p`` is in dbar, matching that routine.
    """

    th2 = T * T
    sqrts = math.sqrt(S) if S >= 0.0 else math.nan

    anum = (
        9.9984085444849347e02
        + T
        * (
            7.3471625860981584e00
            + T * (-5.3211231792841769e-02 + T * 3.6492439109814549e-04)
        )
        + S
        * (
            2.5880571023991390e00
            - T * 6.7168282786692355e-03
            + S * 1.9203202055760151e-03
        )
    )
    aden = (
        1.0000000000000000e00
        + T
        * (
            7.2815210113327091e-03
            + T
            * (
                -4.4787265461983921e-05
                + T * (3.3851002965802430e-07 + T * 1.3651202389758572e-10)
            )
        )
        + S
        * (
            1.7632126669040377e-03
            - T * (8.8066583251206474e-06 + th2 * 1.8832689434804897e-10)
            + sqrts * (5.7463776745432097e-06 + th2 * 1.4716275472242334e-09)
        )
    )

    if use_pressure and p > 0.0:
        pth = p * T
        anum = anum + p * (
            1.1798263740430364e-02
            + th2 * 9.8920219266399117e-08
            + S * 4.6996642771754730e-06
            - p * (2.5862187075154352e-08 + th2 * 3.2921414007960662e-12)
        )
        aden = aden + p * (
            6.7103246285651894e-06
            - pth * (th2 * 2.4461698007024582e-17 + p * 9.1534417604289062e-18)
        )

    return float(anum / aden)


def _is_legacy_lake_method(method: int) -> bool:
    return method in {
        METHOD_UNESCO_FULL,
        METHOD_UNESCO_POTENTIAL,
        METHOD_JACKETT_FULL,
        METHOD_JACKETT_POTENTIAL,
    }


def _legacy_uses_pressure(method: int) -> bool:
    return method in {METHOD_UNESCO_FULL, METHOD_JACKETT_FULL}


def _legacy_density(method: int, S: float, T: float, p_bar: float) -> float:
    use_pressure = _legacy_uses_pressure(method)
    if method in {METHOD_UNESCO_FULL, METHOD_UNESCO_POTENTIAL}:
        return unesco_density(S, T, p_bar, use_pressure)
    return jackett_density(S, T, p_bar * 10.0, use_pressure)


def _legacy_alpha(method: int, S: float, T: float, p_bar: float, rho0: float) -> float:
    delta = 0.01
    rho_a = _legacy_density(method, S, T + 0.5 * delta, p_bar)
    rho_b = _legacy_density(method, S, T - 0.5 * delta, p_bar)
    return -(rho_a - rho_b) / (rho0 * delta)


def _legacy_beta(method: int, S: float, T: float, p_bar: float, rho0: float) -> float:
    delta = 0.01
    rho_a = _legacy_density(method, S + 0.5 * delta, T, p_bar)
    rho_b = _legacy_density(method, S - 0.5 * delta, T, p_bar)
    return (rho_a - rho_b) / (rho0 * delta)


class DensityState:
    """State for the GOTM density / EOS module.

    Mirrors the module-level public and private variables in density.F90.
    Set configuration attributes, call :func:`init_density`, then call
    :func:`do_density` each time step.

    Attributes
    ----------
    density_method : int
        EOS method: METHOD_TEOS10 (1), METHOD_LINEAR_TEOS10 (2), or
        METHOD_LINEAR_USER (3).
    T0, S0, p0 : float
        Reference temperature [°C], salinity [g/kg], pressure [dbar].
        Used by methods 2 and 3 for the linearisation reference point.
    rho0 : float
        Reference density [kg/m³], default 1027.  Used by method 3.
    alpha0, beta0 : float
        Thermal expansion coefficient [1/K] and haline contraction coefficient
        [kg/g].  For method 2, computed from TEOS-10 at (S0, T0, p0).
        For method 3, supplied by the caller.
    cp : float
        Specific heat capacity of seawater [J/(kg·K)]; set by init_density.
    alpha, beta : np.ndarray or None, shape (nlev+1,)
        Per-interface thermal expansion and haline contraction.
    rho, rho_p : np.ndarray or None, shape (nlev+1,)
        In-situ and potential density at cell centres.  Index 0 is seabed
        (GOTM convention); index nlev is surface.
    """

    def __init__(self) -> None:
        self.density_method: int = METHOD_TEOS10
        self.T0: float = 10.0
        self.S0: float = 35.0
        self.p0: float = 0.0
        self.rho0: float = 1027.0
        self.alpha0: float = 0.0
        self.beta0: float = 0.0
        self.cp: float = 0.0

        # Internal reference density used in the linear EOS
        self._rhob: float = 1027.0

        # Arrays allocated by init_density
        self.alpha: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.rho: np.ndarray | None = None
        self.rho_p: np.ndarray | None = None


def init_density(state: DensityState, nlev: int) -> None:
    """Initialise the density module state for a column of *nlev* layers.

    Method-specific initialisation:

    * ``METHOD_TEOS10`` (1): sets ``cp = CP0``; all other coefficients are
      computed per call in :func:`do_density`.
    * ``METHOD_LINEAR_TEOS10`` (2): computes ``_rhob = gsw_sigma0(S0,T0) + 1000``,
      ``alpha0 = gsw_alpha(S0,T0,p0)``, ``beta0 = gsw_beta(S0,T0,p0)``,
      and ``cp = CP0`` from TEOS-10 at the user reference point.
    * ``METHOD_LINEAR_USER`` (3): sets ``_rhob = rho0`` from the caller-supplied
      reference density; ``alpha0`` and ``beta0`` must be set by the caller
      before calling this function.

    After all methods, allocates ``alpha``, ``beta`` (shape ``nlev+1``, filled
    with ``alpha0`` / ``beta0``), ``rho``, and ``rho_p`` (shape ``nlev+1``,
    zero-initialised).

    Parameters
    ----------
    state : DensityState
        Pre-configured state (set density_method and reference values first).
    nlev : int
        Number of vertical layers.
    """
    if state.density_method == METHOD_TEOS10:
        state.cp = CP0
    elif state.density_method == METHOD_LINEAR_TEOS10:
        state._rhob = float(_gsw.sigma0(state.S0, state.T0)) + 1000.0
        state.alpha0 = float(_gsw.alpha(state.S0, state.T0, state.p0))
        state.beta0 = float(_gsw.beta(state.S0, state.T0, state.p0))
        state.cp = CP0
    elif state.density_method == METHOD_LINEAR_USER:
        state._rhob = state.rho0
    elif _is_legacy_lake_method(state.density_method):
        state.cp = CP0

    state.alpha = np.full(nlev + 1, state.alpha0)
    state.beta = np.full(nlev + 1, state.beta0)
    state.rho = np.zeros(nlev + 1)
    state.rho_p = np.zeros(nlev + 1)


def do_density(
    state: DensityState,
    nlev: int,
    S: np.ndarray,
    T: np.ndarray,
    p: np.ndarray,
    pi: np.ndarray,
) -> None:
    """Compute density, potential density, and expansion coefficients.

    Interface salinity and temperature are computed as arithmetic means of
    adjacent cell-centre values.  Boundary faces (seabed index 0 and surface
    index nlev) use the single adjacent cell value.

    Method-specific computation:

    * ``METHOD_TEOS10`` (1): computes ``rho[1:nlev+1]`` via ``gsw_rho``,
      ``rho_p[1:nlev+1]`` via ``gsw_sigma0 + 1000``, and the full
      interface arrays ``alpha`` and ``beta`` via ``gsw_alpha`` / ``gsw_beta``
      evaluated at the interface salinity, temperature, and pressure.
    * ``METHOD_LINEAR_TEOS10`` / ``METHOD_LINEAR_USER`` (2, 3): applies the
      linear EOS :math:`\\rho_p = \\rho_b(1 - \\alpha_0(T-T_0) + \\beta_0(S-S_0))`
      at all cell centres.  In-situ density is set equal to potential density
      (no pressure dependence implemented, following the Fortran source).

    Parameters
    ----------
    state : DensityState
        Initialised density state.
    nlev : int
        Number of vertical layers.
    S : np.ndarray, shape (nlev+1,)
        Absolute salinity at cell centres [g/kg].
    T : np.ndarray, shape (nlev+1,)
        Conservative temperature at cell centres [°C].
    p : np.ndarray, shape (nlev+1,)
        In-situ pressure at cell centres [dbar].
    pi : np.ndarray, shape (nlev+1,)
        In-situ pressure at cell interfaces [dbar].
    """
    assert state.rho is not None, "call init_density before do_density"
    assert state.rho_p is not None, "call init_density before do_density"
    assert state.alpha is not None, "call init_density before do_density"
    assert state.beta is not None, "call init_density before do_density"

    # Interface S and T: arithmetic mean of adjacent cell centres.
    # Boundary faces (k=0 seabed, k=nlev surface) use the single adjacent cell.
    si = np.empty(nlev + 1)
    ti = np.empty(nlev + 1)
    if nlev > 1:
        si[1:nlev] = 0.5 * (S[1:nlev] + S[2 : nlev + 1])
        ti[1:nlev] = 0.5 * (T[1:nlev] + T[2 : nlev + 1])
    si[0] = S[0]
    si[nlev] = S[nlev]
    ti[0] = T[0]
    ti[nlev] = T[nlev]

    if state.density_method == METHOD_TEOS10:
        state.rho[1 : nlev + 1] = _gsw.rho(
            S[1 : nlev + 1], T[1 : nlev + 1], p[1 : nlev + 1]
        )
        state.rho_p[1 : nlev + 1] = (
            _gsw.sigma0(S[1 : nlev + 1], T[1 : nlev + 1]) + 1000.0
        )
        state.alpha[:] = _gsw.alpha(si, ti, pi)
        state.beta[:] = _gsw.beta(si, ti, pi)
    elif _is_legacy_lake_method(state.density_method):
        for k in range(1, nlev + 1):
            p_bar = float(p[k]) / 10.0
            state.rho[k] = _legacy_density(
                state.density_method,
                float(S[k]),
                float(T[k]),
                p_bar,
            )
            state.rho_p[k] = state.rho[k]
        for k in range(1, nlev):
            p_bar = float(pi[k]) / 10.0
            state.alpha[k] = _legacy_alpha(
                state.density_method,
                float(si[k]),
                float(ti[k]),
                p_bar,
                state.rho0,
            )
            state.beta[k] = _legacy_beta(
                state.density_method,
                float(si[k]),
                float(ti[k]),
                p_bar,
                state.rho0,
            )
        state.alpha[0] = _legacy_alpha(
            state.density_method,
            float(S[0]),
            float(T[0]),
            float(pi[0]) / 10.0,
            state.rho0,
        )
        state.beta[0] = _legacy_beta(
            state.density_method,
            float(S[0]),
            float(T[0]),
            float(pi[0]) / 10.0,
            state.rho0,
        )
        state.alpha[nlev] = _legacy_alpha(
            state.density_method,
            float(S[nlev]),
            float(T[nlev]),
            float(pi[nlev]) / 10.0,
            state.rho0,
        )
        state.beta[nlev] = _legacy_beta(
            state.density_method,
            float(S[nlev]),
            float(T[nlev]),
            float(pi[nlev]) / 10.0,
            state.rho0,
        )
    else:
        # Linear EOS: ρ = ρ_b · (1 − α₀(T − T₀) + β₀(S − S₀))
        # Lars: here, we should implement some sort of pressure dependency
        state.rho_p[1 : nlev + 1] = state._rhob * (
            1.0
            - state.alpha0 * (T[1 : nlev + 1] - state.T0)
            + state.beta0 * (S[1 : nlev + 1] - state.S0)
        )
        state.rho[:] = state.rho_p


def get_rho(state: DensityState, S: float, T: float, p: float | None = None) -> float:
    """Compute density for a single water parcel.

    * ``METHOD_TEOS10`` (1): returns ``gsw_rho(S, T, p)`` when ``p`` is given,
      or ``gsw_sigma0(S, T) + 1000`` (potential density) when ``p`` is omitted.
    * ``METHOD_LINEAR_TEOS10`` / ``METHOD_LINEAR_USER`` (2, 3): returns
      :math:`\\rho_b(1 - \\alpha_0(T-T_0) + \\beta_0(S-S_0))`.  No pressure
      dependence is applied (following the Fortran source).

    Parameters
    ----------
    state : DensityState
        Initialised density state.
    S : float
        Absolute salinity [g/kg].
    T : float
        Conservative temperature [°C].
    p : float, optional
        In-situ pressure [dbar].  If omitted under method 1, returns potential
        density (referenced to 0 dbar).
    """
    if state.density_method == METHOD_TEOS10:
        if p is not None:
            return float(_gsw.rho(S, T, p))
        return float(_gsw.sigma0(S, T)) + 1000.0
    if _is_legacy_lake_method(state.density_method):
        return _legacy_density(state.density_method, S, T, 0.0 if p is None else p)
    return float(
        state._rhob
        * (1.0 - state.alpha0 * (T - state.T0) + state.beta0 * (S - state.S0))
    )


def get_alpha(state: DensityState, S: float, T: float, p: float) -> float:
    """Compute thermal expansion coefficient for a single water parcel.

    * ``METHOD_TEOS10`` (1): returns ``gsw_alpha(S, T, p)`` from TEOS-10.
    * ``METHOD_LINEAR_TEOS10`` / ``METHOD_LINEAR_USER`` (2, 3): returns the
      constant reference value ``alpha0`` set during initialisation.

    Parameters
    ----------
    state : DensityState
        Initialised density state.
    S : float
        Absolute salinity [g/kg].
    T : float
        Conservative temperature [°C].
    p : float
        In-situ pressure [dbar].
    """
    if state.density_method == METHOD_TEOS10:
        return float(_gsw.alpha(S, T, p))
    if _is_legacy_lake_method(state.density_method):
        return _legacy_alpha(state.density_method, S, T, p, state.rho0)
    return state.alpha0


def get_beta(state: DensityState, S: float, T: float, p: float) -> float:
    """Compute haline contraction coefficient for a single water parcel.

    * ``METHOD_TEOS10`` (1): returns ``gsw_beta(S, T, p)`` from TEOS-10.
    * ``METHOD_LINEAR_TEOS10`` / ``METHOD_LINEAR_USER`` (2, 3): returns the
      constant reference value ``beta0`` set during initialisation.

    Parameters
    ----------
    state : DensityState
        Initialised density state.
    S : float
        Absolute salinity [g/kg].
    T : float
        Conservative temperature [°C].
    p : float
        In-situ pressure [dbar].
    """
    if state.density_method == METHOD_TEOS10:
        return float(_gsw.beta(S, T, p))
    if _is_legacy_lake_method(state.density_method):
        return _legacy_beta(state.density_method, S, T, p, state.rho0)
    return state.beta0


def clean_density(state: DensityState) -> None:
    """Release the density arrays held by *state*.

    Sets ``alpha``, ``beta``, ``rho``, and ``rho_p`` to ``None``, mirroring
    Fortran ``deallocate``.  Call :func:`init_density` again before the next
    :func:`do_density` call.
    """
    state.alpha = None
    state.beta = None
    state.rho = None
    state.rho_p = None
