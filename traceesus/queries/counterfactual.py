"""Known-SCM sufficiency and disablement query dispatch.

The preliminary experiment uses the prespecified data-generating SCM; it does
not estimate or fit that SCM.  These functions retain the exact legacy
abduction-action-prediction arithmetic while making the scientific boundary
explicit at the query interface.
"""

from __future__ import annotations

import numpy as np

from traceesus.experiments.counterfactual import kernel


def kidney_blind_posterior(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: kernel.ExperimentConfig,
) -> np.ndarray:
    """Return resemblance scores that deliberately omit the direct renal path."""

    return kernel.kidney_blind_posterior(
        biomarkers, renal, renal_effect_sd, config
    )


def kidney_aware_posterior(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: kernel.ExperimentConfig,
) -> np.ndarray:
    """Return the Bayes posterior under the same known renal-aware SCM."""

    return kernel.kidney_aware_posterior(
        biomarkers, renal, renal_effect_sd, config
    )


def sufficiency_disablement_scores(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: kernel.ExperimentConfig,
) -> dict[str, np.ndarray]:
    """Compute posterior-integrated sufficiency and disablement exactly.

    For every patient, both latent-mechanism branches are enumerated.  Within
    each branch, the branch-specific exogenous biomarker residual is abducted,
    reused under intervention, and integrated over the kidney-aware posterior;
    this avoids abducting once as if each candidate were already factual.

    In the deliberately symmetric K=2 model, normalized sufficiency and
    disablement are monotone transformations of the correctly specified
    posterior.  Therefore a gain over kidney-blind resemblance identifies the
    value of representing the renal path, not an intrinsic advantage of causal
    queries over the Bayes classifier.
    """

    return kernel.posterior_integrated_counterfactual_scores(
        biomarkers, renal, renal_effect_sd, config
    )


__all__ = [
    "kidney_aware_posterior",
    "kidney_blind_posterior",
    "sufficiency_disablement_scores",
]
