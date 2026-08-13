"""TRACE-ESUS cohort simulators loaded without eager kernel dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "NullCohortSimulator": ".null_cohort",
    "TwoMechanismSimulator": ".two_mechanism",
    "MultiHospitalCohort": ".multi_hospital",
    "MultiHospitalSimulator": ".multi_hospital",
    "SourceHospitalPool": ".multi_hospital",
    "TransportSimulationTruth": ".multi_hospital",
    "pool_source_hospitals": ".multi_hospital",
}


def __getattr__(name: str) -> Any:
    """Resolve one simulator without importing unrelated experiment kernels."""

    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
