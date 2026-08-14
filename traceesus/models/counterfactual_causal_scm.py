"""Counterfactual query on the *fitted* biology-constrained latent SCM.

Why this exists
---------------
The locked counterfactual experiment scores sufficiency and disablement on a
*known* SCM whose parameters are supplied by the simulator.  That design cannot
answer the incremental-value question, because the query and the posterior are
computed from the same hand-supplied truth.

This model fits the identical conditional EM used by the causal SCM row, then
answers with posterior-integrated sufficiency and disablement instead of the
posterior.  Registering it alongside ``BiologicallyConstrainedCausalSCM``
produces the decisive comparison: one fitted model, two queries, identical
cohorts and paired seeds, so any difference is attributable to the query.

Scope of the returned score
---------------------------
``posterior`` returns the row-normalized combined attribution score so that
ranking metrics — top-rank accuracy and false atrial attribution — are directly
comparable with the posterior row.  Those normalized values are *not*
calibrated probabilities, so Brier score and expected calibration error should
not be interpreted for this model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from traceesus.core.model import FitDiagnostics, FittedModel, Model
from traceesus.core.markers import Biomarker
from traceesus.core.simulator import Cohort
from traceesus.models.adjusted_lcm import (
    ConditionalLatentFit,
    conditional_posterior,
    fit_conditional_latent_model,
)
from configs.endotype_discovery import FittingConfig

COUNTERFACTUAL_CAUSAL_SCM = "Biologically constrained latent SCM (counterfactual query)"

_DISABLEMENT_WEIGHT = 0.50
_SUFFICIENCY_WEIGHT = 0.50


def _fitted_candidate_scores(
    candidate: int,
    biomarkers: np.ndarray,
    posterior: np.ndarray,
    effects: np.ndarray,
    noise_sd: np.ndarray,
    nuisance: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one fitted candidate under both abducted mechanism branches."""

    n, marker_count = biomarkers.shape
    disablement_by_branch = np.empty((n, 2), dtype=float)
    sufficiency_by_branch = np.empty((n, 2), dtype=float)
    for branch in (0, 1):
        disabled_gate = np.zeros(marker_count) if branch == candidate else effects[branch]
        disabled = nuisance + disabled_gate + residual[:, branch, :]
        disablement_by_branch[:, branch] = np.sum(
            ((biomarkers - disabled) / noise_sd) ** 2, axis=1
        )
        candidate_only = nuisance + effects[candidate] + residual[:, branch, :]
        sufficiency_by_branch[:, branch] = np.exp(
            -0.5 * np.sum(((biomarkers - candidate_only) / noise_sd) ** 2, axis=1)
        )
    maximum = np.sum((effects[candidate] / noise_sd) ** 2)
    disablement = np.divide(
        np.sum(posterior * disablement_by_branch, axis=1),
        maximum,
        out=np.zeros(n),
        where=maximum > 0.0,
    )
    other = 1 if candidate == 0 else 0
    mismatch = float(np.exp(-0.5 * np.sum(((effects[candidate] - effects[other]) / noise_sd) ** 2)))
    expected = np.sum(posterior * sufficiency_by_branch, axis=1)
    sufficiency = (expected - mismatch) / (1.0 - mismatch) if abs(1.0 - mismatch) > 1e-12 else np.zeros(n)
    return disablement, sufficiency


def fitted_counterfactual_scores(
    fit_result: ConditionalLatentFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
    *,
    disablement_weight: float = _DISABLEMENT_WEIGHT,
    sufficiency_weight: float = _SUFFICIENCY_WEIGHT,
) -> dict[str, np.ndarray]:
    """Apply abduction-action-prediction to one fitted conditional SCM."""

    posterior = conditional_posterior(fit_result, biomarkers, renal)
    effects = np.asarray(fit_result.class_means_at_renal_normal, dtype=float)
    noise_sd = np.sqrt(np.asarray(fit_result.biomarker_variance, dtype=float))
    nuisance = renal[:, None] * np.asarray(fit_result.renal_effect, dtype=float)
    residual = np.empty((len(biomarkers), 2, biomarkers.shape[1]), dtype=float)
    for branch in (0, 1):
        residual[:, branch, :] = biomarkers - (effects[branch] + nuisance)
    disablement = np.empty((len(biomarkers), 2), dtype=float)
    sufficiency = np.empty((len(biomarkers), 2), dtype=float)
    for candidate in (0, 1):
        disablement[:, candidate], sufficiency[:, candidate] = _fitted_candidate_scores(
            candidate, biomarkers, posterior, effects, noise_sd, nuisance, residual
        )

    combined = (
        disablement_weight * disablement + sufficiency_weight * sufficiency
    )
    return {
        "combined": combined,
        "disablement": disablement,
        "sufficiency": sufficiency,
        "posterior": posterior,
    }


def _row_normalize(scores: np.ndarray) -> np.ndarray:
    """Rescale rows to sum to one so argmax-based metrics stay comparable.

    Rows whose scores are non-positive or degenerate fall back to a uniform
    split, which registers as maximal ambiguity rather than a silent preference
    for the first mechanism.
    """

    shifted = scores - np.minimum(np.min(scores, axis=1, keepdims=True), 0.0)
    totals = np.sum(shifted, axis=1, keepdims=True)
    uniform = np.full_like(shifted, 1.0 / shifted.shape[1])
    return np.where(totals > 1e-12, shifted / np.maximum(totals, 1e-12), uniform)


@dataclass(frozen=True)
class FittedCounterfactualCausalSCM(FittedModel):
    """Same fitted conditional SCM, answered by sufficiency and disablement."""

    fit_result: ConditionalLatentFit

    def posterior(self, data: Cohort) -> np.ndarray:
        """Return normalized counterfactual attribution for ranking metrics."""

        scores = fitted_counterfactual_scores(
            self.fit_result, data.biomarkers, data.covariate("renal_dysfunction")
        )
        return _row_normalize(scores["combined"])

    def counterfactual_scores(self, data: Cohort) -> dict[str, np.ndarray]:
        """Expose raw sufficiency, disablement, and the model's own posterior."""

        return fitted_counterfactual_scores(
            self.fit_result, data.biomarkers, data.covariate("renal_dysfunction")
        )

    def fit_diagnostics(self) -> FitDiagnostics:
        """Report the same EM audit fields as the posterior-query row."""

        result = self.fit_result
        return FitDiagnostics(
            converged=result.converged,
            iterations=result.iterations,
            best_start=result.best_start,
            log_likelihood=result.log_likelihood,
            effective_class_fraction=result.effective_class_fraction,
            anchor_margin=result.anchor_margin,
        )

    @property
    def n_parameters(self) -> int:
        """Identical to the posterior row: the query adds no free parameters."""

        return 12


@dataclass(frozen=True)
class CounterfactualCausalSCM(Model):
    """Fit the constrained latent SCM and query it counterfactually."""

    name: str = COUNTERFACTUAL_CAUSAL_SCM

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: FittingConfig,
    ) -> FittedCounterfactualCausalSCM:
        """Reuse the exact conditional EM sequence used by the posterior row."""

        mask = np.zeros(data.biomarkers.shape[1], dtype=bool)
        mask[Biomarker.NT_PROBNP] = True
        result = fit_conditional_latent_model(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            mask,
            rng,
            config,
        )
        return FittedCounterfactualCausalSCM(result)
