"""Fit the pooled associative latent-class model with the shared EM primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from scipy.special import expit

from configs.endotype_discovery import FittingConfig
from traceesus.core.em import e_step, has_converged, m_step
from traceesus.core.markers import Biomarker
from traceesus.core.model import FitDiagnostics, FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.queries.posterior import anchor_order


ASSOCIATIVE_LCA = "Associative latent class model"


@dataclass(frozen=True)
class AssociativeLatentClassFit:
    """Parameters and diagnostics for p(Z) p(R|Z) p(B|Z)."""

    class_probability: np.ndarray
    renal_probability_by_class: np.ndarray
    class_means: np.ndarray
    biomarker_variance: np.ndarray
    log_likelihood: float
    converged: bool
    iterations: int
    best_start: int
    effective_class_fraction: np.ndarray
    anchor_margin: float


def _clip_probability(values: np.ndarray | float, config: FittingConfig) -> np.ndarray:
    return np.clip(values, config.probability_floor, 1.0 - config.probability_floor)


def initial_responsibilities(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    rng: Generator,
    start_index: int,
    config: FittingConfig,
) -> np.ndarray:
    """Initialize classes; draws occur only for start >=3, then fallback if flat."""

    biomarker_sd = np.std(biomarkers, axis=0)
    biomarker_sd = np.where(biomarker_sd > 1e-8, biomarker_sd, 1.0)
    standardized = (biomarkers - np.mean(biomarkers, axis=0)) / biomarker_sd
    if start_index == 0:
        projection = standardized[:, Biomarker.PTFV1] - standardized[:, Biomarker.COMPETING_VASCULAR]
    elif start_index == 1:
        projection = standardized[:, Biomarker.NT_PROBNP]
    elif start_index == 2:
        projection = 2.0 * (renal - np.mean(renal))
    else:
        direction = rng.normal(size=standardized.shape[1] + 1)
        direction /= np.linalg.norm(direction)
        projection = np.column_stack((standardized, renal)) @ direction
    projection_sd = float(np.std(projection))
    if projection_sd < 1e-8:
        projection = rng.normal(size=biomarkers.shape[0])
        projection_sd = float(np.std(projection))
    atrial = expit(1.5 * (projection - np.median(projection)) / projection_sd)
    atrial = _clip_probability(atrial, config)
    return np.column_stack((atrial, 1.0 - atrial))


def _prior_parameters(
    responsibility: np.ndarray,
    renal: np.ndarray,
    config: FittingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Update the retained smoothed p(Z) and p(R|Z) terms."""

    effective = np.sum(responsibility, axis=0)
    smoothing = config.beta_prior_pseudocount
    class_probability = (effective + smoothing) / (len(renal) + 2.0 * smoothing)
    renal_probability = (
        np.sum(responsibility * renal[:, None], axis=0) + smoothing
    ) / (effective + 2.0 * smoothing)
    return class_probability, _clip_probability(renal_probability, config)


def _log_prior(
    renal: np.ndarray,
    class_probability: np.ndarray,
    renal_probability: np.ndarray,
) -> np.ndarray:
    return (
        np.log(class_probability)[None, :]
        + renal[:, None] * np.log(renal_probability)[None, :]
        + (1 - renal[:, None]) * np.log(1.0 - renal_probability)[None, :]
    )


def _fit_start(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    responsibility: np.ndarray,
    config: FittingConfig,
) -> dict[str, object]:
    """Run one associative EM start using the shared emission update."""

    previous = -np.inf
    converged = False
    for iteration in range(1, config.maximum_em_iterations + 1):
        class_probability, renal_probability = _prior_parameters(responsibility, renal, config)
        emission = m_step(
            biomarkers,
            responsibility,
            config.variance_floor,
            config.minimum_effective_class_fraction,
        )
        responsibility, log_likelihood = e_step(
            biomarkers,
            _log_prior(renal, class_probability, renal_probability),
            emission.class_means,
            emission.variance,
        )
        if has_converged(previous, log_likelihood, config.relative_log_likelihood_tolerance):
            converged = True
            break
        previous = log_likelihood
    return {
        "class_probability": class_probability,
        "renal_probability": renal_probability,
        "emission": emission,
        "responsibility": responsibility,
        "log_likelihood": log_likelihood,
        "converged": converged,
        "iterations": iteration,
    }


def fit_associative_latent_class_model(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    rng: Generator,
    config: FittingConfig,
) -> AssociativeLatentClassFit:
    """Fit all starts; RNG draws follow start order and occur only in initialization."""

    best: dict[str, object] | None = None
    for start_index in range(config.random_starts):
        responsibility = initial_responsibilities(biomarkers, renal, rng, start_index, config)
        try:
            candidate = _fit_start(biomarkers, renal, responsibility, config)
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        candidate["best_start"] = start_index
        if best is None or float(candidate["log_likelihood"]) > float(best["log_likelihood"]):
            best = candidate
    if best is None:
        raise RuntimeError("All associative latent-class EM starts failed.")
    emission = best["emission"]
    order, margin = anchor_order(
        emission.class_means,
        emission.variance,
        atrial_electrical_index=Biomarker.PTFV1,
        competing_specific_index=Biomarker.COMPETING_VASCULAR,
    )
    responsibility = best["responsibility"][:, order]
    return AssociativeLatentClassFit(
        best["class_probability"][order],
        best["renal_probability"][order],
        emission.class_means[order],
        emission.variance,
        float(best["log_likelihood"]),
        bool(best["converged"]),
        int(best["iterations"]),
        int(best["best_start"]),
        np.mean(responsibility, axis=0),
        margin,
    )


def associative_posterior(
    fit: AssociativeLatentClassFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
) -> np.ndarray:
    """Evaluate the oriented associative posterior deterministically."""

    responsibility, _ = e_step(
        biomarkers,
        _log_prior(renal, fit.class_probability, fit.renal_probability_by_class),
        fit.class_means,
        fit.biomarker_variance,
    )
    return responsibility


@dataclass(frozen=True)
class FittedAssociativeLatentClassModel(FittedModel):
    """Expose an immutable associative fit through the common posterior API."""

    fit_result: AssociativeLatentClassFit

    def posterior(self, data: Cohort) -> np.ndarray:
        return associative_posterior(
            self.fit_result, data.biomarkers, data.covariate("renal_dysfunction")
        )

    def fit_diagnostics(self) -> FitDiagnostics:
        result = self.fit_result
        return FitDiagnostics(
            result.converged,
            result.iterations,
            result.best_start,
            result.log_likelihood,
            result.effective_class_fraction,
            result.anchor_margin,
        )

    @property
    def n_parameters(self) -> int:
        return 12


@dataclass(frozen=True)
class AssociativeLatentClassModel(Model):
    """Fit p(Z)p(R|Z)prod_j p(B_j|Z) without mechanism labels."""

    name: str = ASSOCIATIVE_LCA

    def fit(
        self, data: Cohort, rng: Generator, config: FittingConfig
    ) -> FittedAssociativeLatentClassModel:
        """Fit using only observed fields and the explicitly supplied RNG."""

        result = fit_associative_latent_class_model(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            rng,
            config,
        )
        return FittedAssociativeLatentClassModel(result)


__all__ = [
    "ASSOCIATIVE_LCA",
    "AssociativeLatentClassFit",
    "AssociativeLatentClassModel",
    "FittedAssociativeLatentClassModel",
    "associative_posterior",
    "fit_associative_latent_class_model",
    "initial_responsibilities",
]
