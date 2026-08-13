"""Unlabeled cross-hospital transportability and shift controls."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Load the facade lazily so the exact numerical kernel stays importable."""

    if name != "TransportabilityExperiment":
        raise AttributeError(name)
    from .experiment import TransportabilityExperiment

    globals()[name] = TransportabilityExperiment
    return TransportabilityExperiment


__all__ = ["TransportabilityExperiment"]
