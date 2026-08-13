"""Supervised associative-versus-SCM model comparison.

Every fitted method in this experiment receives known synthetic mechanism
labels.  This package is therefore deliberately separate from the unlabeled
endotype-discovery experiment.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Load the experiment facade only when requested by the root runner."""

    if name != "ModelComparisonExperiment":
        raise AttributeError(name)
    from .experiment import ModelComparisonExperiment

    globals()[name] = ModelComparisonExperiment
    return ModelComparisonExperiment

__all__ = ["ModelComparisonExperiment"]
