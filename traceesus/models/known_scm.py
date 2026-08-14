"""Evaluate prespecified SCM queries and fit the counterfactual null mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator
from scipy.special import expit

from configs.counterfactual import ExperimentConfig
from traceesus.core.em import e_step, m_step
from traceesus.core.model import FittedModel, Model
from traceesus.core.simulator import Cohort


ATRIAL = 0
COMPETING = 1
METHOD_POSTERIOR_BLIND = "Posterior matching (kidney-blind)"


def atrial_prior_given_renal(renal: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    log_odds = config.atrial_log_odds_when_renal_normal + config.renal_to_atrial_log_odds * renal
    return expit(log_odds)


def _class_prior(
    renal: np.ndarray, config: ExperimentConfig, include_renal_path: bool
) -> np.ndarray:
    """Return either conditional or historically marginalized class priors."""

    if include_renal_path:
        atrial = atrial_prior_given_renal(renal, config)
    else:
        normal = atrial_prior_given_renal(np.zeros(1, dtype=int), config)[0]
        impaired = atrial_prior_given_renal(np.ones(1, dtype=int), config)[0]
        marginal = (1.0 - config.renal_prevalence) * normal + config.renal_prevalence * impaired
        atrial = np.full(len(renal), marginal, dtype=float)
    return np.column_stack((atrial, 1.0 - atrial))


def posterior_from_model(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: ExperimentConfig,
    include_renal_path: bool,
) -> np.ndarray:
    """Evaluate a two-class known-SCM posterior through the shared E-step."""

    nuisance = np.zeros_like(biomarkers)
    if include_renal_path:
        nuisance[:, 0] = renal_effect_sd * renal
    posterior, _ = e_step(
        biomarkers,
        np.log(_class_prior(renal, config, include_renal_path)),
        np.asarray(config.mechanism_effects, dtype=float),
        np.asarray(config.biomarker_noise_sd, dtype=float) ** 2,
        nuisance_contribution=nuisance,
    )
    return posterior


def kidney_blind_posterior(
    biomarkers: np.ndarray, renal: np.ndarray, renal_effect_sd: float, config: ExperimentConfig
) -> np.ndarray:
    return posterior_from_model(biomarkers, renal, renal_effect_sd, config, False)


def kidney_aware_posterior(
    biomarkers: np.ndarray, renal: np.ndarray, renal_effect_sd: float, config: ExperimentConfig
) -> np.ndarray:
    return posterior_from_model(biomarkers, renal, renal_effect_sd, config, True)


def _candidate_counterfactual(
    candidate: int,
    biomarkers: np.ndarray,
    posterior: np.ndarray,
    effects: np.ndarray,
    noise_sd: np.ndarray,
    nuisance: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate disablement and sufficiency for one candidate mechanism."""

    n = len(biomarkers)
    disablement = np.empty((n, 2), dtype=float)
    sufficiency = np.empty((n, 2), dtype=float)
    for branch in (ATRIAL, COMPETING):
        disabled_gate = np.zeros(3) if branch == candidate else effects[branch]
        disabled = nuisance + disabled_gate + residual[:, branch, :]
        disablement[:, branch] = np.sum(((biomarkers - disabled) / noise_sd) ** 2, axis=1)
        candidate_only = nuisance + effects[candidate] + residual[:, branch, :]
        sufficiency[:, branch] = np.exp(
            -0.5 * np.sum(((biomarkers - candidate_only) / noise_sd) ** 2, axis=1)
        )
    maximum = np.sum((effects[candidate] / noise_sd) ** 2)
    normalized_disablement = np.sum(posterior * disablement, axis=1) / maximum
    other = COMPETING if candidate == ATRIAL else ATRIAL
    mismatch = np.exp(-0.5 * np.sum(((effects[candidate] - effects[other]) / noise_sd) ** 2))
    expected_sufficiency = np.sum(posterior * sufficiency, axis=1)
    return normalized_disablement, (expected_sufficiency - mismatch) / (1.0 - mismatch)


def posterior_integrated_counterfactual_scores(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: ExperimentConfig,
) -> dict[str, np.ndarray]:
    """Evaluate posterior-integrated sufficiency and disablement deterministically."""

    posterior = kidney_aware_posterior(biomarkers, renal, renal_effect_sd, config)
    effects = np.asarray(config.mechanism_effects, dtype=float)
    noise_sd = np.asarray(config.biomarker_noise_sd, dtype=float)
    nuisance = np.zeros_like(biomarkers)
    nuisance[:, 0] = renal_effect_sd * renal
    residual = biomarkers[:, None, :] - effects[None, :, :] - nuisance[:, None, :]
    disablement = np.empty((len(biomarkers), 2), dtype=float)
    sufficiency = np.empty((len(biomarkers), 2), dtype=float)
    for candidate in (ATRIAL, COMPETING):
        disablement[:, candidate], sufficiency[:, candidate] = _candidate_counterfactual(
            candidate, biomarkers, posterior, effects, noise_sd, nuisance, residual
        )
    combined = (
        config.counterfactual_disablement_weight * disablement
        + config.counterfactual_sufficiency_weight * sufficiency
    )
    return {
        "combined": combined,
        "disablement": disablement,
        "sufficiency": sufficiency,
        "posterior": posterior,
    }


def fit_one_diagonal_gaussian(x: np.ndarray, variance_floor: float) -> dict[str, Any]:
    """Fit the null K=1 model through the shared all-observed M-step."""

    responsibility = np.ones((len(x), 1), dtype=float)
    emission = m_step(x, responsibility, variance_floor, 0.0, shared_variance=False)
    _, log_likelihood = e_step(
        x, np.zeros((1, 1)), emission.class_means, emission.variance
    )
    return {
        "log_likelihood": log_likelihood,
        "weights": np.asarray((1.0,)),
        "means": emission.class_means,
        "variances": emission.variance,
        "converged": True,
        "iterations": 1,
    }


def _initial_two_component_parameters(
    x: np.ndarray, rng: Generator, start: int, variance_floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize K=2; random direction is the sole draw and occurs for start >0."""

    dimension = x.shape[1]
    global_variance = np.maximum(np.var(x, axis=0), variance_floor)
    if start == 0:
        direction = np.zeros(dimension)
        direction[np.argmax(global_variance)] = 1.0
    else:
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
    projection = x @ direction
    lower, upper = np.quantile(projection, (0.30, 0.70))
    means = np.vstack((x[projection <= lower].mean(axis=0), x[projection >= upper].mean(axis=0)))
    return np.asarray((0.5, 0.5)), means, np.tile(global_variance, (2, 1))


def _fit_two_component_start(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    max_iter: int,
    variance_floor: float,
    tolerance: float,
) -> dict[str, Any]:
    """Run one K=2 start with the shared E-step and component-variance M-step."""

    previous = -np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        responsibility, log_likelihood = e_step(
            x, np.log(np.clip(weights, 1e-12, None))[None, :], means, variances
        )
        effective = np.maximum(np.sum(responsibility, axis=0), 1e-8)
        weights = effective / len(x)
        emission = m_step(x, responsibility, variance_floor, 0.0, shared_variance=False)
        means, variances = emission.class_means, emission.variance
        if np.isfinite(previous) and abs(log_likelihood - previous) <= tolerance * (1.0 + abs(previous)):
            converged = True
            break
        previous = log_likelihood
    _, final = e_step(x, np.log(np.clip(weights, 1e-12, None))[None, :], means, variances)
    return {
        "log_likelihood": final,
        "weights": weights,
        "means": means,
        "variances": variances,
        "converged": converged,
        "iterations": iteration,
    }


def fit_two_diagonal_gaussians(
    x: np.ndarray,
    rng: Generator,
    starts: int,
    max_iter: int,
    variance_floor: float,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Fit K=2 starts; RNG draws occur once per start after deterministic start 0."""

    best: dict[str, Any] | None = None
    for start in range(starts):
        parameters = _initial_two_component_parameters(x, rng, start, variance_floor)
        candidate = _fit_two_component_start(
            x, *parameters, max_iter, variance_floor, tolerance
        )
        if best is None or candidate["log_likelihood"] > best["log_likelihood"]:
            best = candidate
    if best is None:
        raise RuntimeError("No two-component GMM fit was produced.")
    return best


@dataclass(frozen=True)
class FittedKnownKidneyBlindPosterior(FittedModel):
    renal_effect_sd: float
    config: ExperimentConfig

    def posterior(self, data: Cohort) -> np.ndarray:
        return kidney_blind_posterior(
            data.biomarkers, data.covariate("renal_dysfunction"), self.renal_effect_sd, self.config
        )

    @property
    def n_parameters(self) -> int:
        return 0


@dataclass(frozen=True)
class KnownKidneyBlindPosteriorModel(Model):
    renal_effect_sd: float
    name: str = METHOD_POSTERIOR_BLIND

    def fit(self, data: Cohort, rng: Generator, config: ExperimentConfig) -> FittedKnownKidneyBlindPosterior:
        return FittedKnownKidneyBlindPosterior(self.renal_effect_sd, config)


@dataclass(frozen=True)
class FittedKnownStructuralCausalModel(FittedModel):
    renal_effect_sd: float
    config: ExperimentConfig

    def posterior(self, data: Cohort) -> np.ndarray:
        return kidney_aware_posterior(
            data.biomarkers, data.covariate("renal_dysfunction"), self.renal_effect_sd, self.config
        )

    def counterfactual_scores(self, data: Cohort) -> dict[str, np.ndarray]:
        return posterior_integrated_counterfactual_scores(
            data.biomarkers, data.covariate("renal_dysfunction"), self.renal_effect_sd, self.config
        )

    @property
    def n_parameters(self) -> int:
        return 0


@dataclass(frozen=True)
class KnownStructuralCausalModel(Model):
    renal_effect_sd: float
    name: str = "Known structural causal model"

    def fit(self, data: Cohort, rng: Generator, config: ExperimentConfig) -> FittedKnownStructuralCausalModel:
        return FittedKnownStructuralCausalModel(self.renal_effect_sd, config)


__all__ = [
    "FittedKnownKidneyBlindPosterior",
    "FittedKnownStructuralCausalModel",
    "KnownKidneyBlindPosteriorModel",
    "KnownStructuralCausalModel",
    "fit_one_diagonal_gaussian",
    "fit_two_diagonal_gaussians",
    "kidney_aware_posterior",
    "kidney_blind_posterior",
    "posterior_integrated_counterfactual_scores",
]
