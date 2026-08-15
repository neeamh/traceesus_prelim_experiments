"""Known-SCM posterior and counterfactual query experiment.

The causal specification is the prespecified data-generating model. No SCM is
fitted in this experiment; this boundary is distinct from unlabeled endotype
discovery.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Load the facade lazily while keeping the exact kernel independently usable."""

    if name != "CounterfactualExperiment":
        raise AttributeError(name)
    from .experiment import CounterfactualExperiment

    globals()[name] = CounterfactualExperiment
    return CounterfactualExperiment


__all__ = ["CounterfactualExperiment"]
