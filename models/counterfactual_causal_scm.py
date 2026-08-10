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
from traceesus.core.simulator import Cohort
from traceesus.experiments.endotype_discovery import kernel

COUNTERFACTUAL_CAUSAL_SCM = "Biologically constrained latent SCM (counterfactual query)"

_DISABLEMENT_WEIGHT = 0.50
_SUFFICIENCY_WEIGHT = 0.50


def fitted_counterfactual_scores(
    fit_result: kernel.ConditionalLatentFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
    *,
    disablement_weight: float = _DISABLEMENT_WEIGHT,
    sufficiency_weight: float = _SUFFICIENCY_WEIGHT,
) -> dict[str, np.ndarray]:
    """Abduction-action-prediction on estimated rather than supplied parameters.

    The arithmetic mirrors the locked known-SCM routine exactly — per-branch
    residual abduction, reuse of that residual under intervention, and
    integration over the model's own posterior — with every structural quantity
    read from the fit instead of the simulator.  Abducting within each branch,
    rather than once as if a candidate were already factual, is what keeps the
    query a counterfactual rather than a relabeled likelihood.
    """

    posterior = kernel.conditional_posterior(fit_result, biomarkers, renal)
    effects = np.asarray(fit_result.class_means_at_renal_normal, dtype=float)
    noise_sd = np.sqrt(np.asarray(fit_result.biomarker_variance, dtype=float))
    marker_count = biomarkers.shape[1]
    n = biomarkers.shape[0]

    nuisance_contribution = renal[:, None] * np.asarray(
        fit_result.renal_effect, dtype=float
    )

    # residual[i, z, :] is the exogenous residual abducted under branch Z = z.
    residual = np.empty((n, 2, marker_count), dtype=float)
    for branch in (kernel.Mechanism.ATRIAL, kernel.Mechanism.COMPETING):
        residual[:, branch, :] = biomarkers - (effects[branch] + nuisance_contribution)

    normalized_disablement = np.empty((n, 2), dtype=float)
    normalized_sufficiency = np.empty((n, 2), dtype=float)

    for candidate in (kernel.Mechanism.ATRIAL, kernel.Mechanism.COMPETING):
        disablement_by_branch = np.empty((n, 2), dtype=float)
        sufficiency_by_branch = np.empty((n, 2), dtype=float)

        for branch in (kernel.Mechanism.ATRIAL, kernel.Mechanism.COMPETING):
            disabled_gate = (
                np.zeros(marker_count, dtype=float)
                if branch == candidate
                else effects[branch]
            )
            if_disabled = nuisance_contribution + disabled_gate + residual[:, branch, :]
            disablement_by_branch[:, branch] = np.sum(
                ((biomarkers - if_disabled) / noise_sd) ** 2, axis=1
            )

            if_candidate_only = (
                nuisance_contribution + effects[candidate] + residual[:, branch, :]
            )
            sufficiency_by_branch[:, branch] = np.exp(
                -0.5
                * np.sum(((biomarkers - if_candidate_only) / noise_sd) ** 2, axis=1)
            )

        maximum_disablement = np.sum((effects[candidate] / noise_sd) ** 2)
        normalized_disablement[:, candidate] = np.divide(
            np.sum(posterior * disablement_by_branch, axis=1),
            maximum_disablement,
            out=np.zeros(n, dtype=float),
            where=maximum_disablement > 0.0,
        )

        other = (
            kernel.Mechanism.COMPETING
            if candidate == kernel.Mechanism.ATRIAL
            else kernel.Mechanism.ATRIAL
        )
        mismatch_fit = float(
            np.exp(-0.5 * np.sum(((effects[candidate] - effects[other]) / noise_sd) ** 2))
        )
        denominator = 1.0 - mismatch_fit
        expected_sufficiency = np.sum(posterior * sufficiency_by_branch, axis=1)
        normalized_sufficiency[:, candidate] = (
            (expected_sufficiency - mismatch_fit) / denominator
            if abs(denominator) > 1e-12
            else np.zeros(n, dtype=float)
        )

    combined = (
        disablement_weight * normalized_disablement
        + sufficiency_weight * normalized_sufficiency
    )
    return {
        "combined": combined,
        "disablement": normalized_disablement,
        "sufficiency": normalized_sufficiency,
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

    fit_result: kernel.ConditionalLatentFit

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


class CounterfactualCausalSCM(Model):
    """Fit the constrained latent SCM and query it counterfactually."""

    name = COUNTERFACTUAL_CAUSAL_SCM

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: kernel.FittingConfig,
    ) -> FittedCounterfactualCausalSCM:
        """Reuse the exact conditional EM sequence used by the posterior row."""

        mask = np.zeros(data.biomarkers.shape[1], dtype=bool)
        mask[kernel.Biomarker.NT_PROBNP] = True
        result = kernel.fit_conditional_latent_model(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            mask,
            rng,
            config,
        )
        return FittedCounterfactualCausalSCM(result)
