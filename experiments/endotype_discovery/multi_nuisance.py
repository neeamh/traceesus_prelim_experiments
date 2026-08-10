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
from scipy.special import logsumexp

from traceesus.core.model import FitDiagnostics, FittedModel, Model
from traceesus.core.simulator import Cohort

from . import kernel

ADJUSTED_TWO_NUISANCE = "Two-nuisance adjusted associative latent model"
CAUSAL_TWO_NUISANCE = "Two-path biologically constrained latent SCM"
COUNTERFACTUAL_TWO_NUISANCE = "Two-path latent SCM (counterfactual query)"

#: Biology mask, rows = (renal, heart failure), columns = biomarkers.
BIOLOGY_MASK = np.array(
    [
        [True, False, False],   # renal -> NT-proBNP only
        [False, True, False],   # heart failure -> PTFV1 only
    ]
)

#: Permissive mask: both nuisances free on every marker (adjusted comparator).
PERMISSIVE_MASK = np.ones((2, 3), dtype=bool)


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


def _weighted_multi_regression(
    response: np.ndarray,
    nuisances: np.ndarray,
    responsibility: np.ndarray,
    include: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted M-step for two component intercepts and masked shared slopes.

    Direct generalization of the locked single-slope regression: the design is
    ``[class indicators | permitted nuisance columns]`` and estimates are
    scattered back into a full-length coefficient vector with zeros on
    prohibited paths.
    """

    patient_count = response.shape[0]
    component_count = responsibility.shape[1]
    included_columns = np.flatnonzero(include)
    parameter_count = component_count + included_columns.size
    design = np.zeros((patient_count * component_count, parameter_count), dtype=float)
    tiled_response = np.tile(response, component_count)
    weights = np.concatenate(
        [responsibility[:, component] for component in range(component_count)]
    )
    for component in range(component_count):
        row_slice = slice(component * patient_count, (component + 1) * patient_count)
        design[row_slice, component] = 1.0
        for position, column in enumerate(included_columns):
            design[row_slice, component_count + position] = nuisances[:, column]

    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_response = tiled_response * np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_response, rcond=None)
    effects = np.zeros(nuisances.shape[1], dtype=float)
    effects[included_columns] = coefficients[component_count:]
    return coefficients[:component_count], effects


def _m_step(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisances: np.ndarray,
    responsibility: np.ndarray,
    mask: np.ndarray,
    config: kernel.FittingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    effective_count = np.sum(responsibility, axis=0)
    minimum_count = config.minimum_effective_class_fraction * biomarkers.shape[0]
    if np.any(effective_count < minimum_count):
        raise FloatingPointError("A latent class collapsed below the minimum size.")

    smoothing = config.beta_prior_pseudocount
    class_probability_by_renal = np.zeros((2, 2), dtype=float)
    for renal_value in (0, 1):
        stratum = renal == renal_value
        stratum_count = int(np.sum(stratum))
        if stratum_count == 0:
            raise FloatingPointError("A renal stratum was empty.")
        weighted_atrial = float(np.sum(responsibility[stratum, kernel.Mechanism.ATRIAL]))
        atrial_probability = (weighted_atrial + smoothing) / (stratum_count + 2.0 * smoothing)
        atrial_probability = float(np.clip(
            atrial_probability, config.probability_floor, 1.0 - config.probability_floor
        ))
        class_probability_by_renal[renal_value] = (atrial_probability, 1.0 - atrial_probability)

    marker_count = biomarkers.shape[1]
    class_means = np.zeros((2, marker_count), dtype=float)
    nuisance_effects = np.zeros((nuisances.shape[1], marker_count), dtype=float)
    for marker in range(marker_count):
        class_means[:, marker], nuisance_effects[:, marker] = _weighted_multi_regression(
            biomarkers[:, marker], nuisances, responsibility, mask[:, marker]
        )

    conditional_means = (
        class_means[None, :, :] + (nuisances @ nuisance_effects)[:, None, :]
    )
    residual = biomarkers[:, None, :] - conditional_means
    variance = np.sum(responsibility[:, :, None] * residual**2, axis=(0, 1)) / biomarkers.shape[0]
    variance = np.maximum(variance, config.variance_floor)
    return class_probability_by_renal, class_means, nuisance_effects, variance


def _e_step(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisances: np.ndarray,
    class_probability_by_renal: np.ndarray,
    class_means: np.ndarray,
    nuisance_effects: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, float]:
    patient_means = class_means[None, :, :] + (nuisances @ nuisance_effects)[:, None, :]
    residual = biomarkers[:, None, :] - patient_means
    log_density = -0.5 * (
        np.sum(residual**2 / variance[None, None, :], axis=2)
        + np.sum(np.log(2.0 * np.pi * variance))
    )
    log_joint = np.log(class_probability_by_renal[renal]) + log_density
    log_normalizer = logsumexp(log_joint, axis=1)
    responsibility = np.exp(log_joint - log_normalizer[:, None])
    return responsibility, float(np.sum(log_normalizer))


def fit_multi_nuisance_latent_model(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisances: np.ndarray,
    mask: np.ndarray,
    rng: Generator,
    config: kernel.FittingConfig,
) -> MultiNuisanceLatentFit:
    """Multi-start EM mirroring the locked fitting protocol on a wider design."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (nuisances.shape[1], biomarkers.shape[1]):
        raise ValueError("mask must be (nuisance_count, biomarker_count).")

    best: dict[str, object] | None = None
    for start_index in range(config.random_starts):
        responsibility = kernel._initial_responsibilities(
            biomarkers, renal, rng, start_index, config
        )
        converged = False
        previous = -np.inf
        try:
            for iteration in range(1, config.maximum_em_iterations + 1):
                parameters = _m_step(
                    biomarkers, renal, nuisances, responsibility, mask, config
                )
                responsibility, log_likelihood = _e_step(
                    biomarkers, renal, nuisances, *parameters
                )
                if np.isfinite(previous):
                    denominator = max(abs(previous), 1.0)
                    if (log_likelihood - previous) / denominator < config.relative_log_likelihood_tolerance:
                        converged = True
                        break
                previous = log_likelihood
        except FloatingPointError:
            continue

        if best is None or log_likelihood > best["log_likelihood"]:  # type: ignore[index]
            class_probability_by_renal, class_means, nuisance_effects, variance = parameters
            order, margin = kernel._anchor_order(class_means, variance)
            best = {
                "class_probability_by_renal": class_probability_by_renal[:, order],
                "class_means": class_means[order, :],
                "nuisance_effects": nuisance_effects,
                "variance": variance,
                "log_likelihood": log_likelihood,
                "converged": converged,
                "iterations": iteration,
                "best_start": start_index,
                "responsibility": responsibility[:, order],
                "anchor_margin": margin,
            }

    if best is None:
        raise FloatingPointError("Every EM start collapsed for the two-nuisance model.")

    responsibility = best["responsibility"]  # type: ignore[assignment]
    return MultiNuisanceLatentFit(
        class_probability_by_renal=best["class_probability_by_renal"],
        class_means_at_reference=best["class_means"],
        nuisance_effects=best["nuisance_effects"],
        nuisance_path_mask=mask,
        biomarker_variance=best["variance"],
        log_likelihood=float(best["log_likelihood"]),
        converged=bool(best["converged"]),
        iterations=int(best["iterations"]),
        best_start=int(best["best_start"]),
        effective_class_fraction=np.mean(responsibility, axis=0),
        anchor_margin=float(best["anchor_margin"]),
    )


def multi_nuisance_posterior(
    fit: MultiNuisanceLatentFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisances: np.ndarray,
) -> np.ndarray:
    """Posterior over mechanisms under the fitted two-nuisance model."""

    responsibility, _ = _e_step(
        biomarkers,
        renal,
        nuisances,
        fit.class_probability_by_renal,
        fit.class_means_at_reference,
        fit.nuisance_effects,
        fit.biomarker_variance,
    )
    return responsibility


def multi_nuisance_counterfactual_scores(
    fit: MultiNuisanceLatentFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisances: np.ndarray,
    *,
    disablement_weight: float = 0.50,
    sufficiency_weight: float = 0.50,
) -> dict[str, np.ndarray]:
    """Sufficiency and disablement on the fitted two-path model.

    Identical abduction-action-prediction arithmetic to the locked query, with
    one substantive difference: the nuisance contribution now contains the
    estimated heart-failure path on PTFV1.  When the atrial gate is disabled,
    that path remains available to explain an elevated PTFV1 — which is exactly
    the redundancy the one-path model could not represent, and the reason the
    collapse theorem's premises no longer hold here.
    """

    posterior = multi_nuisance_posterior(fit, biomarkers, renal, nuisances)
    effects = np.asarray(fit.class_means_at_reference, dtype=float)
    noise_sd = np.sqrt(np.asarray(fit.biomarker_variance, dtype=float))
    n = biomarkers.shape[0]
    nuisance_contribution = nuisances @ np.asarray(fit.nuisance_effects, dtype=float)

    # Why the reference world below is nuisance-only, not the rival mechanism.
    #
    # A point-abducted residual reused additively cancels any disabled gate:
    # the disablement distance reduces to a patient-independent constant times
    # the posterior, for every path mask — which is exactly the collapse this
    # experiment exists to escape.  Richens' definition removes the candidate
    # cause while *preserving the patient's background*.  Kidneys and heart
    # failure are background, so the disabled world keeps the estimated
    # nuisance contribution and drops the mechanism gate.  That world's density
    # is not one of the K=2 mixture components, so disablement is no longer a
    # function of the posterior — and for a renal+HF patient it can remain
    # close to the evidence (both markers explained by background), which is
    # the redundancy signature.

    # Evidence compatibility with each candidate world, in shared units.
    def _fit_to(world_mean: np.ndarray) -> np.ndarray:
        return np.exp(
            -0.5 * np.sum(((biomarkers - world_mean) / noise_sd) ** 2, axis=1)
        )

    world_fit = {
        candidate: _fit_to(effects[candidate] + nuisance_contribution)
        for candidate in (kernel.Mechanism.ATRIAL, kernel.Mechanism.COMPETING)
    }
    background_only_fit = _fit_to(nuisance_contribution)

    normalized_disablement = np.empty((n, 2), dtype=float)
    normalized_sufficiency = np.empty((n, 2), dtype=float)
    for candidate in (kernel.Mechanism.ATRIAL, kernel.Mechanism.COMPETING):
        other = (
            kernel.Mechanism.COMPETING
            if candidate == kernel.Mechanism.ATRIAL
            else kernel.Mechanism.ATRIAL
        )
        # Sufficiency: could the candidate, acting with the patient's observed
        # background, reproduce the evidence?  Normalized against the fit a
        # pure signature mismatch would produce, as in the locked query.
        mismatch_fit = float(
            np.exp(-0.5 * np.sum(((effects[candidate] - effects[other]) / noise_sd) ** 2))
        )
        denominator = 1.0 - mismatch_fit
        normalized_sufficiency[:, candidate] = (
            (world_fit[candidate] - mismatch_fit) / denominator
            if abs(denominator) > 1e-12
            else np.zeros(n)
        )

        # Disablement: if the candidate were the cause and is removed, does the
        # evidence survive on background alone?  Integrated over the posterior:
        # removing the candidate changes nothing when the rival was the cause.
        evidence_persists = background_only_fit / np.maximum(
            world_fit[candidate], 1e-300
        )
        evidence_persists = np.minimum(evidence_persists, 1.0)
        normalized_disablement[:, candidate] = posterior[:, candidate] * (
            1.0 - evidence_persists
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


def _fit(data: Cohort, rng: Generator, config, mask: np.ndarray, query: str) -> _FittedTwoNuisance:
    result = fit_multi_nuisance_latent_model(
        data.biomarkers,
        data.covariate("renal_dysfunction"),
        _nuisance_matrix(data),
        mask,
        rng,
        config,
    )
    return _FittedTwoNuisance(result, query)


class TwoNuisanceAdjustedLCM(Model):
    """Both nuisances free on every marker: the strongest adjusted comparator."""

    name = ADJUSTED_TWO_NUISANCE

    def fit(self, data: Cohort, rng: Generator, config) -> _FittedTwoNuisance:
        return _fit(data, rng, config, PERMISSIVE_MASK, "posterior")


class TwoNuisanceCausalSCM(Model):
    """Biology mask (renal -> NT-proBNP, HF -> PTFV1), posterior query."""

    name = CAUSAL_TWO_NUISANCE

    def fit(self, data: Cohort, rng: Generator, config) -> _FittedTwoNuisance:
        return _fit(data, rng, config, BIOLOGY_MASK, "posterior")


class TwoNuisanceCounterfactualSCM(Model):
    """Identical fit to the causal row, answered by sufficiency/disablement."""

    name = COUNTERFACTUAL_TWO_NUISANCE

    def fit(self, data: Cohort, rng: Generator, config) -> _FittedTwoNuisance:
        return _fit(data, rng, config, BIOLOGY_MASK, "counterfactual")
