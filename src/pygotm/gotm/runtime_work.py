"""Persistent work arrays for compiled single-column GOTM integration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from pygotm.arrays import make_column_array
from pygotm.gotm.runtime_state import FloatArray

__all__ = ["RuntimeWork", "allocate_runtime_work"]

_WORK_FIELDS = (
    "au",
    "bu",
    "cu",
    "du",
    "ru",
    "qu",
    "avh",
    "q_sour",
    "l_sour",
    "sig_eff",
    "adv_cu",
    "idpdx",
    "idpdy",
    "dusdz",
    "dvsdz",
    "vel_relax_tau",
    "vel_relax_tau_eff",
    "s_relax_tau",
    "t_relax_tau",
    "uprof",
    "vprof",
    "q2l",
    "seagrass_z",
    "seagrass_exc",
    "seagrass_vfric",
    "seagrass_xx",
    "seagrass_yy",
    "seagrass_xxP",
    "seagrass_excur",
    "seagrass_grassfric",
)


def _new_work_array(nlev: int) -> FloatArray:
    return make_column_array(nlev)


@dataclass(slots=True)
class RuntimeWork:
    """Reusable scratch arrays passed explicitly into compiled routines."""

    nlev: int
    au: FloatArray
    bu: FloatArray
    cu: FloatArray
    du: FloatArray
    ru: FloatArray
    qu: FloatArray
    avh: FloatArray
    q_sour: FloatArray
    l_sour: FloatArray
    sig_eff: FloatArray
    adv_cu: FloatArray
    idpdx: FloatArray
    idpdy: FloatArray
    dusdz: FloatArray
    dvsdz: FloatArray
    vel_relax_tau: FloatArray
    vel_relax_tau_eff: FloatArray
    s_relax_tau: FloatArray
    t_relax_tau: FloatArray
    uprof: FloatArray
    vprof: FloatArray
    q2l: FloatArray
    seagrass_z: FloatArray
    seagrass_exc: FloatArray
    seagrass_vfric: FloatArray
    seagrass_xx: FloatArray
    seagrass_yy: FloatArray
    seagrass_xxP: FloatArray
    seagrass_excur: FloatArray
    seagrass_grassfric: FloatArray

    def iter_arrays(self) -> Iterator[tuple[str, FloatArray]]:
        """Yield every persistent work array."""

        for name in _WORK_FIELDS:
            yield name, getattr(self, name)

    def validate(self) -> None:
        """Raise if work arrays violate compiled-runner assumptions."""

        if self.nlev < 1:
            msg = f"nlev must be at least 1, got {self.nlev}"
            raise ValueError(msg)
        expected_shape = (self.nlev + 1,)
        for name, array in self.iter_arrays():
            if array.dtype != np.float64:
                msg = f"{name} must have dtype float64, got {array.dtype}"
                raise TypeError(msg)
            if array.shape != expected_shape:
                msg = f"{name} must have shape {expected_shape}, got {array.shape}"
                raise ValueError(msg)
            if not array.flags.c_contiguous:
                msg = f"{name} must be C-contiguous"
                raise ValueError(msg)


def allocate_runtime_work(nlev: int) -> RuntimeWork:
    """Allocate persistent work arrays once for a single-column run."""

    work = RuntimeWork(
        nlev=nlev,
        au=_new_work_array(nlev),
        bu=_new_work_array(nlev),
        cu=_new_work_array(nlev),
        du=_new_work_array(nlev),
        ru=_new_work_array(nlev),
        qu=_new_work_array(nlev),
        avh=_new_work_array(nlev),
        q_sour=_new_work_array(nlev),
        l_sour=_new_work_array(nlev),
        sig_eff=_new_work_array(nlev),
        adv_cu=_new_work_array(nlev),
        idpdx=_new_work_array(nlev),
        idpdy=_new_work_array(nlev),
        dusdz=_new_work_array(nlev),
        dvsdz=_new_work_array(nlev),
        vel_relax_tau=np.full(nlev + 1, 1.0e15, dtype=np.float64),
        vel_relax_tau_eff=np.full(nlev + 1, 1.0e15, dtype=np.float64),
        s_relax_tau=np.full(nlev + 1, 1.0e15, dtype=np.float64),
        t_relax_tau=np.full(nlev + 1, 1.0e15, dtype=np.float64),
        uprof=_new_work_array(nlev),
        vprof=_new_work_array(nlev),
        q2l=_new_work_array(nlev),
        seagrass_z=_new_work_array(nlev),
        seagrass_exc=_new_work_array(nlev),
        seagrass_vfric=_new_work_array(nlev),
        seagrass_xx=_new_work_array(nlev),
        seagrass_yy=_new_work_array(nlev),
        seagrass_xxP=_new_work_array(nlev),
        seagrass_excur=_new_work_array(nlev),
        seagrass_grassfric=_new_work_array(nlev),
    )
    work.validate()
    return work
