"""Unsupervised latent endotype discovery experiment."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Load an experiment facade only when requested, leaving kernels importable."""

    if name == "EndotypeDiscoveryExperiment":
        from .experiment import EndotypeDiscoveryExperiment

        globals()[name] = EndotypeDiscoveryExperiment
        return EndotypeDiscoveryExperiment
    if name == "RedundancySweepExperiment":
        from .redundancy import RedundancySweepExperiment

        globals()[name] = RedundancySweepExperiment
        return RedundancySweepExperiment
    raise AttributeError(name)

__all__ = ["EndotypeDiscoveryExperiment", "RedundancySweepExperiment"]
