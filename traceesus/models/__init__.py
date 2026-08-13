"""Model implementations sharing the uniform TRACE-ESUS interface.

Exports are loaded lazily because individual model adapters depend on exact
experiment kernels; eager imports would turn those dependencies into package
initialization cycles.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "AdjustedLatentClassModel": ".adjusted_lcm",
    "AssociativeLatentClassModel": ".associative_lcm",
    "BiologicallyConstrainedCausalSCM": ".causal_scm",
    "SupervisedStructuralCausalModel": ".causal_scm",
    "CounterfactualCausalSCM": ".counterfactual_causal_scm",
    "DataGeneratingOracle": ".oracle",
    "BiomarkersOnlyLogisticModel": ".logistic",
    "KidneyAdjustedLogisticModel": ".logistic",
    "FrozenCausalSCM": ".modular_causal_scm",
    "ModularCausalSCM": ".modular_causal_scm",
    "PooledAssociativeTransportModel": ".modular_causal_scm",
    "TargetAdjustedAssociativeModel": ".modular_causal_scm",
    "TargetTransportOracle": ".modular_causal_scm",
    "KnownKidneyBlindPosteriorModel": ".known_scm",
    "KnownStructuralCausalModel": ".known_scm",
}


def __getattr__(name: str) -> Any:
    """Resolve one public model without importing unrelated experiment kernels."""

    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "AdjustedLatentClassModel",
    "AssociativeLatentClassModel",
    "BiologicallyConstrainedCausalSCM",
    "SupervisedStructuralCausalModel",
    "CounterfactualCausalSCM",
    "DataGeneratingOracle",
    "BiomarkersOnlyLogisticModel",
    "KidneyAdjustedLogisticModel",
    "FrozenCausalSCM",
    "ModularCausalSCM",
    "PooledAssociativeTransportModel",
    "TargetAdjustedAssociativeModel",
    "TargetTransportOracle",
    "KnownKidneyBlindPosteriorModel",
    "KnownStructuralCausalModel",
]
