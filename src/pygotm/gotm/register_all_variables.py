"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: register_all_variables
!
! !INTERFACE:
!   module register_all_variables
!
! !DESCRIPTION:
!
! !USES:
!   use field_manager
!   IMPLICIT NONE
!
! !PUBLIC MEMBER FUNCTIONS:
!   public :: do_register_all_variables
!
! !PUBLIC DATA MEMBERS:
!   type (type_field_manager), public, target :: fm
!
! !REVISION HISTORY:
!  Original FORTRAN author(s): Karsten Bolding & Jorn Bruggeman
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "FieldDimension",
    "FieldRecord",
    "FieldRegistry",
    "do_register_all_variables",
    "fm",
    "snapshot_registry",
]


Provider = Callable[[], float | np.ndarray]


def _float_attr_provider(obj: object, attr: str) -> Provider:
    def provider() -> float:
        return float(getattr(obj, attr))

    return provider


def _float_array0_provider(obj: object, attr: str) -> Provider:
    def provider() -> float:
        values = getattr(obj, attr)
        assert isinstance(values, np.ndarray)
        return float(values[0])

    return provider


def _array_attr_provider(obj: object, attr: str) -> Provider:
    def provider() -> np.ndarray:
        values = getattr(obj, attr)
        assert isinstance(values, np.ndarray)
        return values

    return provider


def _input_value_provider(obj: object, attr: str) -> Provider:
    def provider() -> float:
        return float(getattr(obj, attr).value)

    return provider


def _scalar_input_provider(input_: Any) -> Provider:
    def provider() -> float:
        return float(input_.value)

    return provider


def _profile_data_provider(obj: object, attr: str, nlev: int) -> Provider:
    def provider() -> np.ndarray:
        values = getattr(obj, attr).data
        assert isinstance(values, np.ndarray)
        return values[1 : nlev + 1]

    return provider


def _constant_array_provider(values: np.ndarray) -> Provider:
    def provider() -> np.ndarray:
        return values

    return provider


def _constant_float_provider(value: float) -> Provider:
    def provider() -> float:
        return value

    return provider


def _array_slice_provider(obj: object, attr: str, start: int, stop: int) -> Provider:
    def provider() -> np.ndarray:
        values = getattr(obj, attr)
        assert isinstance(values, np.ndarray)
        return values[start:stop]

    return provider


def _callable_float_provider(func: Callable[[], float]) -> Provider:
    def provider() -> float:
        return float(func())

    return provider


@dataclass
class FieldDimension:
    name: str
    size: int | None


@dataclass
class FieldRecord:
    name: str
    units: str
    long_name: str
    provider: Provider
    standard_name: str = ""
    dimensions: tuple[str, ...] = ()
    category: str = ""
    coordinate_dimension: str | None = None
    state: bool = False

    def snapshot(self) -> float | np.ndarray:
        value = self.provider()
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        return float(value)


@dataclass
class FieldRegistry:
    dimensions: dict[str, FieldDimension] = field(default_factory=dict)
    fields: dict[str, FieldRecord] = field(default_factory=dict)

    def clear(self) -> None:
        self.dimensions.clear()
        self.fields.clear()

    def register_dimension(self, name: str, size: int | None) -> None:
        self.dimensions[name] = FieldDimension(name=name, size=size)

    def register(
        self,
        name: str,
        units: str,
        long_name: str,
        *,
        data0d: float | None = None,
        data1d: np.ndarray | None = None,
        provider: Provider | None = None,
        standard_name: str = "",
        dimensions: tuple[str, ...] = (),
        category: str = "",
        coordinate_dimension: str | None = None,
        state: bool = False,
    ) -> None:
        if provider is None:
            if data1d is not None:
                provider = _constant_array_provider(data1d)
            elif data0d is not None:
                provider = _constant_float_provider(float(data0d))
            else:
                msg = f"field {name!r} requires data or a provider"
                raise ValueError(msg)
        self.fields[name] = FieldRecord(
            name=name,
            units=units,
            long_name=long_name,
            provider=provider,
            standard_name=standard_name,
            dimensions=dimensions,
            category=category,
            coordinate_dimension=coordinate_dimension,
            state=state,
        )

    def list(self) -> tuple[str, ...]:
        return tuple(self.fields)

    def state_fields(self) -> tuple[FieldRecord, ...]:
        return tuple(field for field in self.fields.values() if field.state)

    def finalize(self) -> None:
        self.clear()


fm = FieldRegistry()


def _register_coordinate_variables(lat: float, lon: float) -> None:
    fm.register(
        "lon",
        "degrees_east",
        "longitude",
        data0d=lon,
        dimensions=("lon",),
        coordinate_dimension="lon",
    )
    fm.register(
        "lat",
        "degrees_north",
        "latitude",
        data0d=lat,
        dimensions=("lat",),
        coordinate_dimension="lat",
    )


def _register_meanflow_variables(nlev: int, meanflow: Any | None) -> None:
    if meanflow is None:
        return
    array_fields = (
        ("z", "m", "cell-centre depth", ("z",)),
        ("zi", "m", "interface depth", ("zi",)),
        ("h", "m", "layer thickness", ("z",)),
        ("u", "m/s", "x-velocity", ("z",)),
        ("v", "m/s", "y-velocity", ("z",)),
        ("w", "m/s", "vertical velocity", ("z",)),
        ("T", "Celsius", "conservative temperature", ("z",)),
        ("S", "g/kg", "absolute salinity", ("z",)),
        ("NN", "1/s2", "buoyancy frequency squared", ("zi",)),
        (
            "NNT",
            "1/s2",
            "temperature contribution to buoyancy frequency squared",
            ("zi",),
        ),
        ("NNS", "1/s2", "salinity contribution to buoyancy frequency squared", ("zi",)),
        ("SS", "1/s2", "shear frequency squared", ("zi",)),
        ("SSU", "1/s2", "x contribution to shear frequency squared", ("zi",)),
        ("SSV", "1/s2", "y contribution to shear frequency squared", ("zi",)),
        ("buoy", "m/s2", "buoyancy", ("z",)),
        ("rad", "W/m2", "shortwave radiation profile", ("zi",)),
        ("bioshade", "", "biological shading factor", ("z",)),
        ("xP", "m2/s3", "extra turbulence production", ("z",)),
        ("avh", "m2/s", "eddy diffusivity", ("z",)),
        ("ga", "", "coordinate scaling", ("z",)),
        ("fric", "", "extra friction coefficient in water column", ("z",)),
        ("drag", "", "drag coefficient in water column", ("z",)),
        ("ho", "m", "previous layer thickness", ("z",)),
    )
    for name, units, long_name, dimensions in array_fields:
        values = getattr(meanflow, name, None)
        if isinstance(values, np.ndarray):
            fm.register(
                name,
                units,
                long_name,
                provider=_array_attr_provider(meanflow, name),
                dimensions=dimensions,
                category="meanflow",
                state=True,
            )
    alias_fields = (
        ("temp", "Celsius", "temperature (conservative)", "T"),
        ("salt", "g/kg", "salinity (absolute)", "S"),
    )
    for name, units, long_name, source in alias_fields:
        values = getattr(meanflow, source, None)
        if isinstance(values, np.ndarray):
            fm.register(
                name,
                units,
                long_name,
                provider=_array_attr_provider(meanflow, source),
                dimensions=("z",),
                category="temperature_and_salinity",
                state=True,
            )
    for name, units, long_name in (
        ("zeta", "m", "surface elevation"),
        ("depth", "m", "water depth"),
        ("cori", "1/s", "Coriolis parameter"),
        ("u_taus", "m/s", "surface friction velocity"),
        ("u_taub", "m/s", "bottom friction velocity"),
        ("u_taubo", "m/s", "previous bottom friction velocity"),
        ("taub", "Pa", "bottom stress"),
        ("Hice", "m", "fake ice thickness"),
    ):
        if hasattr(meanflow, name):
            fm.register(
                name,
                units,
                long_name,
                provider=_float_attr_provider(meanflow, name),
                category="meanflow",
                state=True,
            )
    del nlev


def _register_ice_variables(ice_state: Any | None) -> None:
    if ice_state is None:
        return
    for name, units, long_name in (
        ("Hfrazil", "m", "frazil ice thickness"),
        ("Hice", "m", "ice thickness"),
        ("Hsnow", "m", "snow thickness on ice"),
        ("T1", "Celsius", "upper ice-layer temperature"),
        ("T2", "Celsius", "lower ice-layer temperature"),
        ("Tf", "Celsius", "freezing temperature"),
        ("Tice_surface", "Celsius", "ice surface temperature"),
        ("albedo_ice", "", "ice albedo"),
        ("transmissivity", "", "ice shortwave transmissivity"),
        ("ocean_ice_flux", "W/m2", "ocean-to-ice heat flux"),
        ("ocean_ice_heat_flux", "W/m2", "diagnosed ocean-ice heat flux"),
        ("ocean_ice_salt_flux", "g/kg m/s", "diagnosed ocean-ice salt flux"),
        ("surface_ice_energy", "J/m2", "surface ice energy residual"),
        ("bottom_ice_energy", "J/m2", "bottom ice energy residual"),
        ("melt_rate", "m/s", "ice melt rate"),
        ("T_melt", "Celsius", "ice-ocean interface temperature"),
        ("S_melt", "g/kg", "ice-ocean interface salinity"),
    ):
        if hasattr(ice_state, name):
            fm.register(
                name,
                units,
                long_name,
                provider=_float_array0_provider(ice_state, name),
                category="ice",
                state=True,
            )


def _register_density_variables(density: Any | None) -> None:
    if density is None:
        return
    for name, units, long_name, dimensions in (
        ("rho_p", "kg/m3", "density (potential)", ("z",)),
        ("rho", "kg/m3", "density (in-situ)", ("z",)),
        ("alpha", "1/K", "thermal expansion coefficient", ("zi",)),
        ("beta", "kg/g", "saline contraction coefficient", ("zi",)),
    ):
        values = getattr(density, name, None)
        if isinstance(values, np.ndarray):
            fm.register(
                name,
                units,
                long_name,
                provider=_array_attr_provider(density, name),
                dimensions=dimensions,
                category="density",
            )


def _register_observation_variables(nlev: int, observations: Any | None) -> None:
    if observations is None:
        return
    if observations.tprof_input.data is not None:
        fm.register(
            "temp_obs",
            "Celsius",
            "temperature (observed)",
            provider=_profile_data_provider(observations, "tprof_input", nlev),
            dimensions=("z",),
            category="temperature_and_salinity",
        )
    if observations.sprof_input.data is not None:
        fm.register(
            "salt_obs",
            "psu",
            "salinity (observed)",
            provider=_profile_data_provider(observations, "sprof_input", nlev),
            dimensions=("z",),
            category="temperature_and_salinity",
        )
    if observations.uprof_input.data is not None:
        fm.register(
            "u_obs",
            "m/s",
            "x-velocity (observed)",
            provider=_profile_data_provider(observations, "uprof_input", nlev),
            dimensions=("z",),
            category="velocities",
        )
    if observations.vprof_input.data is not None:
        fm.register(
            "v_obs",
            "m/s",
            "y-velocity (observed)",
            provider=_profile_data_provider(observations, "vprof_input", nlev),
            dimensions=("z",),
            category="velocities",
        )
    if (
        observations.epsprof_input.method != 0
        and observations.epsprof_input.data is not None
    ):
        fm.register(
            "eps_obs",
            "m2/s3",
            "observed dissipation",
            provider=_profile_data_provider(observations, "epsprof_input", nlev),
            dimensions=("z",),
            category="turbulence",
        )
    for name, units, long_name, attr in (
        ("zeta_obs", "m", "sea surface elevation", "zeta_input"),
        ("dpdx", "", "pressure in West-East direction", "dpdx_input"),
        ("dpdy", "", "pressure in South-North direction", "dpdy_input"),
    ):
        fm.register(
            name,
            units,
            long_name,
            provider=_input_value_provider(observations, attr),
            category="observations",
        )
    for name, source in (("idpdx", "idpdx"), ("idpdy", "idpdy")):
        values = getattr(observations, source, None)
        if isinstance(values, np.ndarray):
            fm.register(
                name,
                "",
                f"internal pressure gradient ({source[-1]})",
                provider=_array_attr_provider(observations, source),
                dimensions=("z",),
                category="mimic_3d",
            )


def _register_diagnostic_variables(diagnostics: Any | None) -> None:
    if diagnostics is None:
        return
    for name, units, long_name in (
        ("ekin", "J/m2", "kinetic energy"),
        ("epot", "J/m2", "potential energy"),
        ("eturb", "J/m2", "turbulent energy"),
        ("mld_surf", "m", "surface mixed-layer depth"),
        ("mld_bott", "m", "bottom mixed-layer depth"),
    ):
        fm.register(
            name,
            units,
            long_name,
            provider=_float_attr_provider(diagnostics, name),
            category="diagnostics",
        )
    for name, source in (
        ("Ekin", "ekin"),
        ("Epot", "epot"),
        ("Eturb", "eturb"),
    ):
        fm.register(
            name,
            "J/m2",
            name,
            provider=_float_attr_provider(diagnostics, source),
            category="diagnostics",
        )
    if diagnostics.taux is not None:
        fm.register(
            "taux",
            "m2/s2",
            "turbulent momentum flux (x)",
            provider=_array_attr_provider(diagnostics, "taux"),
            dimensions=("zi",),
            category="diagnostics",
        )
    if diagnostics.tauy is not None:
        fm.register(
            "tauy",
            "m2/s2",
            "turbulent momentum flux (y)",
            provider=_array_attr_provider(diagnostics, "tauy"),
            dimensions=("zi",),
            category="diagnostics",
        )
    if diagnostics.Rig is not None:
        fm.register(
            "Rig",
            "",
            "gradient Richardson number",
            provider=_array_attr_provider(diagnostics, "Rig"),
            dimensions=("zi",),
            category="diagnostics",
        )


def _register_stokes_variables(nlev: int, stokes_drift: Any | None = None) -> None:
    if stokes_drift is None:
        zero_z = np.zeros(nlev, dtype=np.float64)
        zero_zi = np.zeros(nlev + 1, dtype=np.float64)
        us_provider = _constant_array_provider(zero_z)
        vs_provider = _constant_array_provider(zero_z)
        dusdz_provider = _constant_array_provider(zero_zi)
        dvsdz_provider = _constant_array_provider(zero_zi)
        us0_provider = _constant_float_provider(0.0)
        vs0_provider = _constant_float_provider(0.0)
        ds_provider = _constant_float_provider(0.0)
    else:
        us_provider = _array_slice_provider(stokes_drift, "usprof", 1, nlev + 1)
        vs_provider = _array_slice_provider(stokes_drift, "vsprof", 1, nlev + 1)
        dusdz_provider = _array_attr_provider(stokes_drift, "dusdz")
        dvsdz_provider = _array_attr_provider(stokes_drift, "dvsdz")
        us0_provider = _float_attr_provider(stokes_drift, "us0")
        vs0_provider = _float_attr_provider(stokes_drift, "vs0")
        ds_provider = _float_attr_provider(stokes_drift, "ds")

    for name in ("us", "vs"):
        fm.register(
            name,
            "m/s",
            name,
            provider=us_provider if name == "us" else vs_provider,
            dimensions=("z",),
            category="stokes_drift",
        )
    for name in ("dusdz", "dvsdz"):
        fm.register(
            name,
            "1/s",
            name,
            provider=dusdz_provider if name == "dusdz" else dvsdz_provider,
            dimensions=("zi",),
            category="stokes_drift",
        )
    for name, provider in (
        ("us0", us0_provider),
        ("vs0", vs0_provider),
        ("ds", ds_provider),
    ):
        fm.register(
            name,
            "m/s" if name != "ds" else "m",
            name,
            provider=provider,
            category="stokes_drift",
        )


def _register_airsea_variables(
    airsea: Any | None,
    surface_inputs: Any | None,
    i0_provider: Callable[[], float] | None,
) -> None:
    if airsea is None:
        return
    if surface_inputs is not None:
        for name, units, long_name, attr in (
            ("u10", "m/s", "10m wind (x)", "u10"),
            ("v10", "m/s", "10m wind (y)", "v10"),
            ("airt", "Celsius", "2m air temperature", "airt"),
            ("airp", "Pa", "air pressure", "airp"),
            ("hum", "", "humidity input", "hum"),
            ("cloud", "", "cloud cover", "cloud"),
            ("precip", "m/s", "precipitation", "precip"),
            ("sst_obs", "Celsius", "observed sea surface temperature", "sst_obs"),
            ("sss", "1e-3", "sea surface salinity", "sss_obs"),
        ):
            input_ = getattr(surface_inputs, attr, None)
            provider: Provider
            if input_ is not None:
                provider = _scalar_input_provider(input_)
            else:
                provider = _constant_float_provider(0.0)
            fm.register(
                name,
                units,
                long_name,
                provider=provider,
                category="surface",
            )
    scalar_fields = (
        ("es", "Pa", "saturation water vapour pressure"),
        ("ea", "Pa", "actual water vapour pressure"),
        ("qs", "kg/kg", "saturation specific humidity"),
        ("qa", "kg/kg", "specific humidity"),
        ("rhoa", "kg/m3", "air density"),
        ("shortwave", "W/m2", "incoming shortwave radiation"),
        ("heat", "W/m2", "net surface heat flux"),
        ("qh", "W/m2", "sensible heat flux"),
        ("qe", "W/m2", "latent heat flux"),
        ("ql", "W/m2", "net longwave radiation"),
        ("tx", "m2/s2", "wind stress (x)"),
        ("ty", "m2/s2", "wind stress (y)"),
        ("sst", "Celsius", "sea surface temperature"),
        ("sss", "1e-3", "sea surface salinity"),
        ("albedo", "", "surface albedo"),
    )
    for name, units, long_name in scalar_fields:
        if hasattr(airsea, name):
            fm.register(
                name,
                units,
                long_name,
                provider=_float_attr_provider(airsea, name),
                category="surface",
            )
    if hasattr(airsea, "evap"):
        fm.register(
            "evap",
            "m/s",
            "evaporation",
            provider=_float_attr_provider(airsea, "evap"),
            category="surface",
        )
    for name, units, long_name in (
        ("int_precip", "m", "integrated precipitation"),
        ("int_evap", "m", "integrated evaporation"),
        ("int_fwf", "m", "integrated freshwater flux"),
        ("int_swr", "J/m2", "integrated shortwave radiation"),
        ("int_heat", "J/m2", "integrated surface heat flux"),
        ("int_total", "J/m2", "integrated total surface heat exchange"),
    ):
        if hasattr(airsea, name):
            fm.register(
                name,
                units,
                long_name,
                provider=_float_attr_provider(airsea, name),
                category="surface",
            )
    if i0_provider is not None:
        fm.register(
            "I_0",
            "W/m2",
            "incoming shortwave radiation",
            provider=_callable_float_provider(i0_provider),
            category="surface",
        )


def _register_turbulence_variables(turbulence: Any | None) -> None:
    if turbulence is None:
        return
    fields = (
        ("num", "m2/s", "turbulent diffusivity of momentum"),
        ("nuh", "m2/s", "turbulent diffusivity of heat"),
        ("nus", "m2/s", "turbulent diffusivity of salt"),
        ("nucl", "m2/s", "turbulent diffusivity down Stokes gradient"),
        ("gamu", "m2/s2", "non-local flux of u-momentum"),
        ("gamv", "m2/s2", "non-local flux of v-momentum"),
        ("gamb", "m2/s3", "non-local buoyancy flux"),
        ("gamh", "K m/s", "non-local heat flux"),
        ("gams", "g/kg m/s", "non-local salinity flux"),
        ("Rig", "", "gradient Richardson number"),
        ("tke", "m2/s2", "turbulent kinetic energy"),
        ("tkeo", "m2/s2", "previous turbulent kinetic energy"),
        ("eps", "m2/s3", "energy dissipation rate"),
        ("L", "m", "turbulence length scale"),
        ("kb", "m2/s4", "buoyancy variance"),
        ("epsb", "m2/s5", "destruction of buoyancy variance"),
        ("P", "m2/s3", "shear production"),
        ("B", "m2/s3", "buoyancy production"),
        ("Pb", "m2/s5", "buoyancy-variance production"),
        ("PSTK", "m2/s3", "Stokes production"),
        ("cmue1", "", "stability function for momentum diffusivity"),
        ("cmue2", "", "stability function for scalar diffusivity"),
        ("gam", "", "non-dimensional non-local buoyancy flux"),
        ("an", "", "non-dimensional buoyancy time scale"),
        ("as_", "", "non-dimensional shear time scale"),
        ("at", "", "non-dimensional buoyancy variance"),
        ("r", "", "turbulent time scale ratio"),
        ("uu", "m2/s2", "variance of u fluctuations"),
        ("vv", "m2/s2", "variance of v fluctuations"),
        ("ww", "m2/s2", "variance of w fluctuations"),
    )
    for source, units, long_name in fields:
        values = getattr(turbulence, source, None)
        if isinstance(values, np.ndarray):
            name = "G" if source == "B" else ("as" if source == "as_" else source)
            fm.register(
                name,
                units,
                long_name,
                provider=_array_attr_provider(turbulence, source),
                dimensions=("zi",),
                category="turbulence",
            )


def do_register_all_variables(
    lat: float,
    lon: float,
    nlev: int,
    *,
    observations: Any | None = None,
    diagnostics: Any | None = None,
    meanflow: Any | None = None,
    airsea: Any | None = None,
    density: Any | None = None,
    turbulence: Any | None = None,
    stokes_drift: Any | None = None,
    ice_state: Any | None = None,
    surface_inputs: Any | None = None,
    i0_provider: Callable[[], float] | None = None,
) -> FieldRegistry:
    """Populate the local field registry with currently available model fields."""

    fm.clear()
    fm.register_dimension("lon", 1)
    fm.register_dimension("lat", 1)
    fm.register_dimension("z", nlev)
    fm.register_dimension("zi", nlev + 1)
    fm.register_dimension("time", None)

    _register_coordinate_variables(lat, lon)
    _register_density_variables(density)
    _register_meanflow_variables(nlev, meanflow)
    _register_ice_variables(ice_state)
    _register_airsea_variables(airsea, surface_inputs, i0_provider)
    _register_observation_variables(nlev, observations)
    _register_turbulence_variables(turbulence)
    _register_diagnostic_variables(diagnostics)
    _register_stokes_variables(nlev, stokes_drift)
    return fm


def snapshot_registry(registry: FieldRegistry) -> dict[str, float | np.ndarray]:
    """Return a copy of every registered field value."""

    return {name: record.snapshot() for name, record in registry.fields.items()}
