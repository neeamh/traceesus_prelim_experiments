"""Two-nuisance latent models: renal on NT-proBNP, heart failure on PTFV1.

Purpose
-------
The proposal-locked conditional EM represents exactly one nuisance path
(renal -> NT-proBNP).  Under the heart-failure generator that model is
misspecified by construction, and — because its nuisance structure stays
one-path — sufficiency/disablement remain monotone in its posterior, so the
query comparison is decided by algebra rather than data.

This module generalizes the *model*, not the locked code: a nuisance design
matrix ``N`` (n x q) with a boolean path mask (q x p).  With the biology mask
(renal -> NT-proBNP, HF -> PTFV1) the fitted model can represent the redundant
explanation, which is the precondition for posterior and counterfactual
queries to separate.

Nothing here touches the locked kernel: the EM below mirrors its arithmetic
(same floors, same stopping rule, same anchor) but lives in its own module and
is only ever invoked by the heart-failure grid.

Design choice retained from the locked model: the class prior is conditioned
on renal status only.  In the generator neither nuisance influences the true
mechanism, and conditioning the prior on the sparse HF stratum (7%) would add
an empty-stratum failure mode without representing any additional biology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from configs.endotype_discovery import FittingConfig
from traceesus.core.em import e_step, m_step
from traceesus.core.markers import Biomarker
from traceesus.core.model import FitDiagnostics, FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.models.adjusted_lcm import conditional_class_probability
from traceesus.models.associative_lcm import initial_responsibilities
from traceesus.queries.posterior import anchor_order

from . import kernel

ADJUSTED_TWO_NUISANCE = "Two-nuisance adjusted associative latent model"
CAUSAL_TWO_NUISANCE = "Two-path biologically constrained latent SCM"
COUNTERFACTUAL_TWO_NUISANCE = "Two-path latent SCM (counterfactual query)"

#: Biology mask, rows = (renal, heart failure), columns = biomarkers.
BIOLOGY_MASK = ((True, False, False), (False, True, False))

#: Permissive mask: both nuisances free on every marker (adjusted comparator).
PERMISSIVE_MASK = ((True, True, True), (True, True, True))


@dataclass(frozen=True)
class MultiNuisanceLatentFit:
    """Fit of p(Z|R) p(B|Z,N) with a prespecified nuisance path mask."""

    class_probability_by_renal: np.ndarray
    class_means_at_reference: np.ndarray
    nuisance_effects: np.ndarray            # (q, p), masked entries fixed at 0
    nuisance_path_mask: np.ndarray
    biomarker_variance: np.ndarray
    log_likelihood: float
    converged: bool
    iterations: int
    best_start: int
    effective_class_fraction: np.ndarray
    anchor_margin: float


def _nuisance_matrix(data: Cohort) -> np.ndarray:
    """Stack the two observed nuisance covariates in a fixed column order."""

    return np.column_stack(
        (
            data.covariate("renal_dysfunction").astype(float),
            data.covariate("heart_failure").astype(float),
        )
    )


def _run_multi_start(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisances: np.ndarray,
    mask: np.ndarray,
    responsibility: np.ndarray,
    config: FittingConfig,
) -> dict[str, object]:
    """Run one two-nuisance EM start through the shared numerical primitives."""

    converged = False
    previous = -np.inf
    for iteration in range(1, config.maximum_em_iterations + 1):
        class_probability = conditional_class_probability(responsibility, renal, config)
        emission = m_step(
            biomarkers,
            responsibility,
            config.variance_floor,
            config.minimum_effective_class_fraction,
            nuisance_design=nuisances,
            path_mask=mask,
        )
        responsibility, log_likelihood = e_step(
            biomarkers,
            np.log(class_probability[renal]),
            emission.class_means,
            emission.variance,
            nuisance_contribution=nuisances @ emission.nuisance_effects,
        )
        if np.isfinite(previous):
            denominator = max(abs(previous), 1.0)
            if (log_likelihood - previous) / denominator < config.relative_log_likelihood_tolerance:
                converged = True
                break
        previous = log_likelihood
    return {
        "class_probability": class_probability,
        "emission": emission,
        "log_likelihood": log_likelihood,
        "converged": converged,
        "iterations": iteration,
        "responsibility": responsibility,
    }


def fit_multi_nuisance_latent_model(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisances: np.ndarray,
    mask: np.ndarray,
    rng: Generator,
    config: FittingConfig,
) -> MultiNuisanceLatentFit:
    """Fit all starts; RNG draws occur only in ordered responsibility initialization."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (nuisances.shape[1], biomarkers.shape[1]):
        raise ValueError("mask must be (nuisance_count, biomarker_count).")
    best: dict[str, object] | None = None
    for start_index in range(config.random_starts):
        initial = initial_responsibilities(biomarkers, renal, rng, start_index, config)
        try:
            candidate = _run_multi_start(biomarkers, renal, nuisances, mask, initial, config)
        except FloatingPointError:
            continue
        candidate["best_start"] = start_index
        if best is None or candidate["log_likelihood"] > best["log_likelihood"]:  # type: ignore[operator]
            best = candidate

    if best is None:
        raise FloatingPointError("Every EM start collapsed for the two-nuisance model.")
    emission = best["emission"]
    order, margin = anchor_order(
        emission.class_means,
        emission.variance,
        atrial_electrical_index=Biomarker.PTFV1,
        competing_specific_index=Biomarker.COMPETING_VASCULAR,
    )
    responsibility = best["responsibility"][:, order]
    return MultiNuisanceLatentFit(
        class_probability_by_renal=best["class_probability"][:, order],
        class_means_at_reference=emission.class_means[order],
        nuisance_effects=emission.nuisance_effects,
        nuisance_path_mask=mask,
        biomarker_variance=emission.variance,
        log_likelihood=float(best["log_likelihood"]),
        converged=bool(best["converged"]),
        iterations=int(best["iterations"]),
        best_start=int(best["best_start"]),
        effective_class_fraction=np.mean(responsibility, axis=0),
        anchor_margin=margin,
    )


def multi_nuisance_posterior(
    fit: MultiNuisanceLatentFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisances: np.ndarray,
) -> np.ndarray:
    """Posterior over mechanisms under the fitted two-nuisance model."""

    responsibility, _ = e_step(
        biomarkers,
        np.log(fit.class_probability_by_renal[renal]),
        fit.class_means_at_reference,
        fit.biomarker_variance,
        nuisance_contribution=nuisances @ fit.nuisance_effects,
    )
    return responsibility


def _evidence_fit(
    biomarkers: np.ndarray,
    world_mean: np.ndarray,
    noise_sd: np.ndarray,
) -> np.ndarray:
    return np.exp(-0.5 * np.sum(((biomarkers - world_mean) / noise_sd) ** 2, axis=1))


def _candidate_scores(
    candidate: int,
    posterior: np.ndarray,
    effects: np.ndarray,
    noise_sd: np.ndarray,
    world_fit: dict[int, np.ndarray],
    background_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one candidate's normalized disablement and sufficiency."""

    other = 1 if candidate == 0 else 0
    mismatch = float(
        np.exp(-0.5 * np.sum(((effects[candidate] - effects[other]) / noise_sd) ** 2))
    )
    denominator = 1.0 - mismatch
    sufficiency = (
        (world_fit[candidate] - mismatch) / denominator
        if abs(denominator) > 1e-12
        else np.zeros(len(posterior))
    )
    persists = np.minimum(
        background_fit / np.maximum(world_fit[candidate], 1e-300), 1.0
    )
    disablement = posterior[:, candidate] * (1.0 - persists)
    return disablement, sufficiency


def multi_nuisance_counterfactual_scores(
    fit: MultiNuisanceLatentFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisances: np.ndarray,
    *,
    disablement_weight: float = 0.50,
    sufficiency_weight: float = 0.50,
) -> dict[str, np.ndarray]:
    """Evaluate fitted two-path sufficiency and disablement scores."""

    posterior = multi_nuisance_posterior(fit, biomarkers, renal, nuisances)
    effects = np.asarray(fit.class_means_at_reference, dtype=float)
    noise_sd = np.sqrt(np.asarray(fit.biomarker_variance, dtype=float))
    nuisance = nuisances @ np.asarray(fit.nuisance_effects, dtype=float)
    world_fit = {
        candidate: _evidence_fit(biomarkers, effects[candidate] + nuisance, noise_sd)
        for candidate in (0, 1)
    }
    background_fit = _evidence_fit(biomarkers, nuisance, noise_sd)
    disablement = np.empty((len(biomarkers), 2), dtype=float)
    sufficiency = np.empty((len(biomarkers), 2), dtype=float)
    for candidate in (0, 1):
        disablement[:, candidate], sufficiency[:, candidate] = _candidate_scores(
            candidate, posterior, effects, noise_sd, world_fit, background_fit
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
    shifted = scores - np.minimum(np.min(scores, axis=1, keepdims=True), 0.0)
    totals = np.sum(shifted, axis=1, keepdims=True)
    uniform = np.full_like(shifted, 1.0 / shifted.shape[1])
    return np.where(totals > 1e-12, shifted / np.maximum(totals, 1e-12), uniform)


def _diagnostics(fit: MultiNuisanceLatentFit) -> FitDiagnostics:
    return FitDiagnostics(
        converged=fit.converged,
        iterations=fit.iterations,
        best_start=fit.best_start,
        log_likelihood=fit.log_likelihood,
        effective_class_fraction=fit.effective_class_fraction,
        anchor_margin=fit.anchor_margin,
    )


@dataclass(frozen=True)
class _FittedTwoNuisance(FittedModel):
    fit_result: MultiNuisanceLatentFit
    query: str  # "posterior" | "counterfactual"

    def posterior(self, data: Cohort) -> np.ndarray:
        nuisances = _nuisance_matrix(data)
        renal = data.covariate("renal_dysfunction")
        if self.query == "posterior":
            return multi_nuisance_posterior(
                self.fit_result, data.biomarkers, renal, nuisances
            )
        scores = multi_nuisance_counterfactual_scores(
            self.fit_result, data.biomarkers, renal, nuisances
        )
        return _row_normalize(scores["combined"])

    def counterfactual_scores(self, data: Cohort) -> dict[str, np.ndarray]:
        return multi_nuisance_counterfactual_scores(
            self.fit_result,
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            _nuisance_matrix(data),
        )

    def fit_diagnostics(self) -> FitDiagnostics:
        return _diagnostics(self.fit_result)

    @property
    def n_parameters(self) -> int:
        return 12 + int(np.sum(self.fit_result.nuisance_path_mask))


def _fit(data: Cohort, rng: Generator, config, mask, query: str) -> _FittedTwoNuisance:
    result = fit_multi_nuisance_latent_model(
        data.biomarkers,
        data.covariate("renal_dysfunction"),
        _nuisance_matrix(data),
        np.asarray(mask, dtype=bool),
        rng,
        config,
    )
    return _FittedTwoNuisance(result, query)


@dataclass(frozen=True)
class TwoNuisanceAdjustedLCM(Model):
    """Both nuisances free on every marker: the strongest adjusted comparator."""

    name: str = ADJUSTED_TWO_NUISANCE

    def fit(self, data: Cohort, rng: Generator, config) -> _FittedTwoNuisance:
        return _fit(data, rng, config, PERMISSIVE_MASK, "posterior")


@dataclass(frozen=True)
class TwoNuisanceCausalSCM(Model):
    """Biology mask (renal -> NT-proBNP, HF -> PTFV1), posterior query."""

    name: str = CAUSAL_TWO_NUISANCE

    def fit(self, data: Cohort, rng: Generator, config) -> _FittedTwoNuisance:
        return _fit(data, rng, config, BIOLOGY_MASK, "posterior")


@dataclass(frozen=True)
class TwoNuisanceCounterfactualSCM(Model):
    """Identical fit to the causal row, answered by sufficiency/disablement."""

    name: str = COUNTERFACTUAL_TWO_NUISANCE

    def fit(self, data: Cohort, rng: Generator, config) -> _FittedTwoNuisance:
        return _fit(data, rng, config, BIOLOGY_MASK, "counterfactual")
