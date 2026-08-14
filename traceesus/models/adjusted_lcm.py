"""Fit renal-adjusted latent mixtures with prespecified nuisance paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from configs.endotype_discovery import FittingConfig
from traceesus.core.em import (
    ConditionalParameters,
    EmissionParameters,
    conditional_e_step,
    run_conditional_em_start,
)
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
    design = renal[:, None].astype(float)
    paths = mask[None, :]
    best = None
    best_start = -1
    for start_index in range(config.random_starts):
        responsibility = initial_responsibilities(biomarkers, renal, rng, start_index, config)
        try:
            candidate = run_conditional_em_start(
                biomarkers,
                renal,
                design,
                paths,
                responsibility,
                config.maximum_em_iterations,
                config.relative_log_likelihood_tolerance,
                config.variance_floor,
                config.minimum_effective_class_fraction,
                config.beta_prior_pseudocount,
                config.probability_floor,
            )
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        if best is None or candidate.log_likelihood > best.log_likelihood:
            best = candidate
            best_start = start_index
    if best is None:
        raise RuntimeError("All conditional latent-model EM starts failed.")
    emission = best.parameters.emission
    order, margin = anchor_order(
        emission.class_means,
        emission.variance,
        atrial_electrical_index=Biomarker.PTFV1,
        competing_specific_index=Biomarker.COMPETING_VASCULAR,
    )
    responsibility = best.responsibility[:, order]
    return ConditionalLatentFit(
        best.parameters.class_probability_by_renal[:, order],
        emission.class_means[order],
        emission.nuisance_effects[0],
        mask,
        emission.variance,
        best.log_likelihood,
        best.converged,
        best.iterations,
        best_start,
        np.mean(responsibility, axis=0),
        margin,
    )


def conditional_posterior(
    fit: ConditionalLatentFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
) -> np.ndarray:
    """Evaluate the fitted renal-conditional posterior deterministically."""

    parameters = ConditionalParameters(
        fit.class_probability_by_renal,
        EmissionParameters(
            fit.class_means_at_renal_normal,
            fit.renal_effect[None, :],
            fit.biomarker_variance,
        ),
    )
    responsibility, _ = conditional_e_step(
        biomarkers,
        renal,
        renal[:, None].astype(float),
        fit.renal_path_mask[None, :],
        parameters,
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
    "conditional_posterior",
    "fit_conditional_latent_model",
]
