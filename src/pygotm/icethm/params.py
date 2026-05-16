"""Configuration parameters for ice thermodynamics models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class IceModelEnum(IntEnum):
    """Stable integer identifiers for Numba dispatch."""

    NO_ICE = 0
    SIMPLE = 1
    BASAL_MELT = 2
    LEBEDEV = 3
    MYLAKE = 4
    WINTON = 5


_ICE_MODEL_MAP: dict[str, IceModelEnum] = {
    "no_ice": IceModelEnum.NO_ICE,
    "none": IceModelEnum.NO_ICE,
    "off": IceModelEnum.NO_ICE,
    "simple": IceModelEnum.SIMPLE,
    "basal_melt": IceModelEnum.BASAL_MELT,
    "basalmelt": IceModelEnum.BASAL_MELT,
    "lebedev": IceModelEnum.LEBEDEV,
    "mylake": IceModelEnum.MYLAKE,
    "winton": IceModelEnum.WINTON,
}


@dataclass(frozen=True)
class IceParams:
    """Immutable scalar parameters used to initialize and dispatch ice models."""

    model: IceModelEnum
    Hice_init: float = 0.0
    Hsnow_init: float = 0.0
    ocean_ice_flux_init: float = 0.0


def canonical_ice_model(value: object, default: str = "simple") -> IceModelEnum:
    """Return the configured ice model enum.

    Tokens accept the same hyphen/underscore normalization used by the GOTM YAML
    parser.
    """

    token = default if value is None else str(value)
    key = token.strip().lower().replace("-", "_")
    try:
        return _ICE_MODEL_MAP[key]
    except KeyError as exc:
        msg = f"unsupported surface.ice.model {key!r}"
        raise NotImplementedError(msg) from exc


def make_ice_params(
    *,
    model: IceModelEnum | int | str = IceModelEnum.SIMPLE,
    Hice_init: float = 0.0,
    Hsnow_init: float = 0.0,
    ocean_ice_flux_init: float = 0.0,
) -> IceParams:
    """Build validated scalar ice parameters."""

    if isinstance(model, str):
        model_enum = canonical_ice_model(model)
    else:
        model_enum = IceModelEnum(int(model))
    return IceParams(
        model=model_enum,
        Hice_init=max(0.0, float(Hice_init)),
        Hsnow_init=max(0.0, float(Hsnow_init)),
        ocean_ice_flux_init=float(ocean_ice_flux_init),
    )


def make_ice_params_from_mapping(mapping: dict[str, Any]) -> IceParams:
    """Create :class:`IceParams` from a ``surface.ice`` YAML mapping."""

    return make_ice_params(
        model=canonical_ice_model(mapping.get("model"), "simple"),
        Hice_init=float(mapping.get("H", mapping.get("Hice", 0.0))),
        Hsnow_init=float(mapping.get("Hsnow", mapping.get("snow", 0.0))),
        ocean_ice_flux_init=float(mapping.get("ocean_ice_flux", 0.0)),
    )
