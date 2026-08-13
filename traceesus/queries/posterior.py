"""Uniform posterior query dispatch."""

from __future__ import annotations

import numpy as np

from traceesus.core.model import FittedModel
from traceesus.core.simulator import Cohort


def anchor_order(
    class_means: np.ndarray,
    biomarker_variance: np.ndarray,
    *,
    atrial_electrical_index: int,
    competing_specific_index: int,
) -> tuple[np.ndarray, float]:
    """Orient K=2 labels by the prespecified biology-only anchor contrast.

    The standardized atrial-electrical minus competing-specific contrast is
    deliberately independent of simulator truth and of the renal-distorted
    NT-proBNP-like marker. The arithmetic order is shared verbatim by latent
    discovery and transport because even a harmless-looking reformulation
    could perturb the recorded anchor margin.
    """

    noise_sd = np.sqrt(biomarker_variance)
    score = (
        class_means[:, atrial_electrical_index]
        / noise_sd[atrial_electrical_index]
        - class_means[:, competing_specific_index]
        / noise_sd[competing_specific_index]
    )
    atrial_component = int(np.argmax(score))
    competing_component = 1 - atrial_component
    order = np.asarray((atrial_component, competing_component), dtype=int)
    return order, float(score[atrial_component] - score[competing_component])


def posterior(model: FittedModel, data: Cohort) -> np.ndarray:
    """Dispatch posterior evaluation without giving the model simulator truth."""

    return model.posterior(data)


__all__ = ["anchor_order", "posterior"]
