"""Fit renal-adjusted latent mixtures with prespecified nuisance paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from configs.endotype_discovery import FittingConfig
from traceesus.core.em import e_step, has_converged, m_step
from traceesus.core.markers import Biomarker
from traceesus.core.model import FitDiagnostics, FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.models.associative_lcm import initial_responsibilities
from traceesus.queries.posterior import anchor_order


ASSOCIATIVE_ADJUSTED = "Renal-adjusted associative latent class model"


@dataclass(frozen=True)
class ConditionalLatentFit:
    """Fit of p(Z|R) p(B|Z,R) under a fixed renal path mask."""

    class_probability_by_renal: np.ndarray
    class_means_at_renal_normal: np.ndarray
    renal_effect: np.ndarray
    renal_path_mask: np.ndarray
    biomarker_variance: np.ndarray
    log_likelihood: float
    converged: bool
    iterations: int
    best_start: int
    effective_class_fraction: np.ndarray
    anchor_margin: float


def conditional_class_probability(
    responsibility: np.ndarray,
    renal: np.ndarray,
    config: FittingConfig,
) -> np.ndarray:
    """Update the retained smoothed p(Z|R) term for both renal strata."""

    result = np.zeros((2, responsibility.shape[1]), dtype=float)
    smoothing = config.beta_prior_pseudocount
    for renal_value in (0, 1):
        stratum = renal == renal_value
        stratum_count = int(np.sum(stratum))
        if stratum_count == 0:
            raise FloatingPointError("A renal stratum was empty.")
        atrial = (float(np.sum(responsibility[stratum, 0])) + smoothing) / (
            stratum_count + 2.0 * smoothing
        )
        atrial = float(np.clip(
            atrial, config.probability_floor, 1.0 - config.probability_floor
        ))
        result[renal_value] = (atrial, 1.0 - atrial)
    return result


def _fit_start(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_path_mask: np.ndarray,
    responsibility: np.ndarray,
    config: FittingConfig,
) -> dict[str, object]:
    """Run one conditional EM start using the shared masked emission update."""

    previous = -np.inf
    converged = False
    design = renal[:, None].astype(float)
    paths = renal_path_mask[None, :]
    for iteration in range(1, config.maximum_em_iterations + 1):
        class_probability = conditional_class_probability(responsibility, renal, config)
        emission = m_step(
            biomarkers,
            responsibility,
            config.variance_floor,
            config.minimum_effective_class_fraction,
            nuisance_design=design,
            path_mask=paths,
        )
        responsibility, log_likelihood = e_step(
            biomarkers,
            np.log(class_probability[renal]),
            emission.class_means,
            emission.variance,
            nuisance_contribution=design @ emission.nuisance_effects,
        )
        if has_converged(previous, log_likelihood, config.relative_log_likelihood_tolerance):
            converged = True
            break
        previous = log_likelihood
    return {
        "class_probability": class_probability,
        "emission": emission,
        "responsibility": responsibility,
        "log_likelihood": log_likelihood,
        "converged": converged,
        "iterations": iteration,
    }


def fit_conditional_latent_model(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_path_mask: np.ndarray,
    rng: Generator,
    config: FittingConfig,
) -> ConditionalLatentFit:
    """Fit all starts; RNG draws follow start order and occur only in initialization."""

    mask = np.asarray(renal_path_mask, dtype=bool)
    if mask.shape != (biomarkers.shape[1],):
        raise ValueError("renal_path_mask must contain one Boolean per biomarker.")
    best: dict[str, object] | None = None
    for start_index in range(config.random_starts):
        responsibility = initial_responsibilities(biomarkers, renal, rng, start_index, config)
        try:
            candidate = _fit_start(biomarkers, renal, mask, responsibility, config)
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        candidate["best_start"] = start_index
        if best is None or float(candidate["log_likelihood"]) > float(best["log_likelihood"]):
            best = candidate
    if best is None:
        raise RuntimeError("All conditional latent-model EM starts failed.")
    emission = best["emission"]
    order, margin = anchor_order(
        emission.class_means,
        emission.variance,
        atrial_electrical_index=Biomarker.PTFV1,
        competing_specific_index=Biomarker.COMPETING_VASCULAR,
    )
    responsibility = best["responsibility"][:, order]
    return ConditionalLatentFit(
        best["class_probability"][:, order],
        emission.class_means[order],
        emission.nuisance_effects[0],
        mask,
        emission.variance,
        float(best["log_likelihood"]),
        bool(best["converged"]),
        int(best["iterations"]),
        int(best["best_start"]),
        np.mean(responsibility, axis=0),
        margin,
    )


def conditional_posterior(
    fit: ConditionalLatentFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
) -> np.ndarray:
    """Evaluate the fitted renal-conditional posterior deterministically."""

    responsibility, _ = e_step(
        biomarkers,
        np.log(fit.class_probability_by_renal[renal]),
        fit.class_means_at_renal_normal,
        fit.biomarker_variance,
        nuisance_contribution=renal[:, None] * fit.renal_effect[None, :],
    )
    return responsibility


@dataclass(frozen=True)
class FittedAdjustedLatentClassModel(FittedModel):
    """Conditional mixture with freely estimated renal slopes for every marker."""

    fit_result: ConditionalLatentFit

    def posterior(self, data: Cohort) -> np.ndarray:
        return conditional_posterior(
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
        return 14


@dataclass(frozen=True)
class AdjustedLatentClassModel(Model):
    """Fit p(Z|R)p(B|Z,R) with a renal association for every biomarker."""

    name: str = ASSOCIATIVE_ADJUSTED

    def fit(
        self, data: Cohort, rng: Generator, config: FittingConfig
    ) -> FittedAdjustedLatentClassModel:
        mask = np.ones(data.biomarkers.shape[1], dtype=bool)
        result = fit_conditional_latent_model(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            mask,
            rng,
            config,
        )
        return FittedAdjustedLatentClassModel(result)


__all__ = [
    "ASSOCIATIVE_ADJUSTED",
    "AdjustedLatentClassModel",
    "ConditionalLatentFit",
    "FittedAdjustedLatentClassModel",
    "conditional_class_probability",
    "conditional_posterior",
    "fit_conditional_latent_model",
]
