"""Fit transport mixtures with missing biomarkers and explicit nuisance paths."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.random import Generator
from scipy.special import expit

from configs.endotype_discovery import FittingConfig
from configs.transportability import TransportExperimentConfig, TransportSimulationConfig
from traceesus.core.em import e_step, has_converged, m_step
from traceesus.core.markers import BIOMARKER_NAMES, Biomarker
from traceesus.core.model import FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.queries.posterior import anchor_order
from traceesus.simulators.multi_hospital import (
    MultiHospitalCohort,
    SourceHospitalPool,
    assay_calibrate_observed,
)


POOLED_ASSOCIATIVE = "Pooled associative latent class model"
TARGET_ADJUSTED_ASSOCIATIVE = "Target-calibrated associative latent model"
FROZEN_CAUSAL = "Frozen causal latent SCM"
MODULAR_CAUSAL = "Modular causal latent SCM"
ORACLE = "Target oracle (reference)"

TransportModelConfig: TypeAlias = FittingConfig | TransportExperimentConfig


@dataclass(frozen=True)
class MissingGaussianMixtureFit:
    """Oriented two-component diagonal Gaussian mixture under missingness."""

    class_probability: np.ndarray
    class_means: np.ndarray
    biomarker_variance: np.ndarray
    log_likelihood: float
    converged: bool
    iterations: int
    best_start: int
    effective_class_fraction: np.ndarray
    anchor_margin: float


def _fitting_config(config: TransportModelConfig) -> FittingConfig:
    if isinstance(config, FittingConfig):
        return config
    if isinstance(config, TransportExperimentConfig):
        return config.fitting
    raise TypeError("Transport models require FittingConfig or .fitting.")


def _source_blocks(data: Cohort) -> tuple[Cohort, ...]:
    return data.hospital_cohorts if isinstance(data, SourceHospitalPool) else (data,)


def fit_nuisance_paths(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    inflammation: np.ndarray,
    renal_path_mask: np.ndarray,
    inflammation_path_mask: np.ndarray,
) -> np.ndarray:
    """Estimate label-free permitted nuisance slopes marker by marker."""

    renal_mask = np.asarray(renal_path_mask, dtype=bool)
    inflammation_mask = np.asarray(inflammation_path_mask, dtype=bool)
    slopes = np.zeros((2, biomarkers.shape[1]), dtype=float)
    for marker in range(biomarkers.shape[1]):
        observed = np.isfinite(biomarkers[:, marker])
        columns = [np.ones(int(np.sum(observed)), dtype=float)]
        slope_rows: list[int] = []
        if renal_mask[marker]:
            columns.append(renal[observed].astype(float))
            slope_rows.append(0)
        if inflammation_mask[marker]:
            columns.append(inflammation[observed].astype(float))
            slope_rows.append(1)
        coefficients, _, _, _ = np.linalg.lstsq(
            np.column_stack(columns), biomarkers[observed, marker], rcond=None
        )
        for position, slope_row in enumerate(slope_rows, start=1):
            slopes[slope_row, marker] = coefficients[position]
    return slopes


def remove_nuisance_paths(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    inflammation: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    """Remove permitted nuisance contributions while preserving NaN cells."""

    return biomarkers - np.column_stack((renal, inflammation)) @ slopes


def _initial_responsibility(
    biomarkers: np.ndarray,
    rng: Generator,
    start_index: int,
    fitting: FittingConfig,
) -> np.ndarray:
    """Initialize classes; random directions are drawn only for start >=2."""

    column_mean = np.nanmean(biomarkers, axis=0)
    column_sd = np.nanstd(biomarkers, axis=0)
    column_sd = np.where(column_sd > 1e-8, column_sd, 1.0)
    standardized = np.where(
        np.isfinite(biomarkers), (biomarkers - column_mean) / column_sd, 0.0
    )
    if start_index == 0:
        projection = standardized[:, Biomarker.PTFV1] - standardized[:, Biomarker.COMPETING_VASCULAR]
    elif start_index == 1:
        projection = standardized[:, Biomarker.NT_PROBNP]
    else:
        direction = rng.normal(size=standardized.shape[1])
        direction /= np.linalg.norm(direction)
        projection = standardized @ direction
    projection_sd = max(float(np.std(projection)), 1e-8)
    atrial = expit(1.5 * (projection - np.median(projection)) / projection_sd)
    atrial = np.clip(atrial, fitting.probability_floor, 1.0 - fitting.probability_floor)
    return np.column_stack((atrial, 1.0 - atrial))


def _fit_start(
    biomarkers: np.ndarray,
    responsibility: np.ndarray,
    fitting: FittingConfig,
) -> dict[str, object]:
    """Run one missing-data mixture start through the shared EM primitives."""

    previous = -np.inf
    converged = False
    for iteration in range(1, fitting.maximum_em_iterations + 1):
        effective = np.sum(responsibility, axis=0)
        smoothing = fitting.beta_prior_pseudocount
        probability = (effective + smoothing) / (len(biomarkers) + 2.0 * smoothing)
        emission = m_step(
            biomarkers,
            responsibility,
            fitting.variance_floor,
            fitting.minimum_effective_class_fraction,
        )
        responsibility, log_likelihood = e_step(
            biomarkers,
            np.log(probability)[None, :],
            emission.class_means,
            emission.variance,
        )
        if has_converged(previous, log_likelihood, fitting.relative_log_likelihood_tolerance):
            converged = True
            break
        previous = log_likelihood
    return {
        "probability": probability,
        "emission": emission,
        "responsibility": responsibility,
        "log_likelihood": log_likelihood,
        "converged": converged,
        "iterations": iteration,
    }


def fit_missing_gaussian_mixture(
    biomarkers: np.ndarray,
    rng: Generator,
    fitting: FittingConfig,
) -> MissingGaussianMixtureFit:
    """Fit all starts; RNG draws follow start order during initialization only."""

    best: dict[str, object] | None = None
    for start_index in range(fitting.random_starts):
        responsibility = _initial_responsibility(biomarkers, rng, start_index, fitting)
        try:
            candidate = _fit_start(biomarkers, responsibility, fitting)
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        candidate["best_start"] = start_index
        if best is None or float(candidate["log_likelihood"]) > float(best["log_likelihood"]):
            best = candidate
    if best is None:
        raise RuntimeError("All missing-data mixture starts failed.")
    emission = best["emission"]
    order, margin = anchor_order(
        emission.class_means,
        emission.variance,
        atrial_electrical_index=Biomarker.PTFV1,
        competing_specific_index=Biomarker.COMPETING_VASCULAR,
    )
    responsibility = best["responsibility"][:, order]
    return MissingGaussianMixtureFit(
        best["probability"][order],
        emission.class_means[order],
        emission.variance,
        float(best["log_likelihood"]),
        bool(best["converged"]),
        int(best["iterations"]),
        int(best["best_start"]),
        np.mean(responsibility, axis=0),
        margin,
    )


def missing_gmm_posterior(
    fit: MissingGaussianMixtureFit, biomarkers: np.ndarray
) -> np.ndarray:
    """Evaluate the oriented missing-data mixture deterministically."""

    responsibility, _ = e_step(
        biomarkers,
        np.log(fit.class_probability)[None, :],
        fit.class_means,
        fit.biomarker_variance,
    )
    return responsibility


@dataclass(frozen=True)
class FittedPooledTransportMixture(FittedModel):
    fit_result: MissingGaussianMixtureFit

    def posterior(self, data: Cohort) -> np.ndarray:
        return missing_gmm_posterior(self.fit_result, assay_calibrate_observed(data))

    @property
    def n_parameters(self) -> int:
        return 10


@dataclass(frozen=True)
class PooledAssociativeTransportModel(Model):
    name: str = POOLED_ASSOCIATIVE

    def fit(
        self, data: Cohort, rng: Generator, config: TransportModelConfig
    ) -> FittedPooledTransportMixture:
        result = fit_missing_gaussian_mixture(
            assay_calibrate_observed(data), rng, _fitting_config(config)
        )
        return FittedPooledTransportMixture(result)


@dataclass(frozen=True)
class FittedResidualizedTransportMixture(FittedModel):
    fit_result: MissingGaussianMixtureFit
    renal_path_mask: np.ndarray
    inflammation_path_mask: np.ndarray
    source_slopes: tuple[np.ndarray, ...]
    active_slopes: np.ndarray

    def posterior(self, data: Cohort) -> np.ndarray:
        biomarkers = assay_calibrate_observed(data)
        residualized = remove_nuisance_paths(
            biomarkers,
            data.covariate("renal_dysfunction"),
            data.covariate("background_inflammation"),
            self.active_slopes,
        )
        return missing_gmm_posterior(self.fit_result, residualized)

    def recalibrate(self, target: Cohort) -> "FittedResidualizedTransportMixture":
        slopes = fit_nuisance_paths(
            assay_calibrate_observed(target),
            target.covariate("renal_dysfunction"),
            target.covariate("background_inflammation"),
            self.renal_path_mask,
            self.inflammation_path_mask,
        )
        return replace(self, active_slopes=slopes)

    @property
    def n_parameters(self) -> int:
        return int(10 + np.sum(self.renal_path_mask) + np.sum(self.inflammation_path_mask))


def _fit_residualized(
    data: Cohort,
    rng: Generator,
    config: TransportModelConfig,
    renal_mask: tuple[bool, bool, bool],
    inflammation_mask: tuple[bool, bool, bool],
) -> FittedResidualizedTransportMixture:
    """Fit site slopes in source order, then one pooled missing-data mixture."""

    renal_paths = np.asarray(renal_mask, dtype=bool)
    inflammation_paths = np.asarray(inflammation_mask, dtype=bool)
    residuals: list[np.ndarray] = []
    slopes_by_source: list[np.ndarray] = []
    for source in _source_blocks(data):
        biomarkers = assay_calibrate_observed(source)
        slopes = fit_nuisance_paths(
            biomarkers,
            source.covariate("renal_dysfunction"),
            source.covariate("background_inflammation"),
            renal_paths,
            inflammation_paths,
        )
        slopes_by_source.append(slopes)
        residuals.append(remove_nuisance_paths(
            biomarkers,
            source.covariate("renal_dysfunction"),
            source.covariate("background_inflammation"),
            slopes,
        ))
    result = fit_missing_gaussian_mixture(np.vstack(residuals), rng, _fitting_config(config))
    return FittedResidualizedTransportMixture(
        result,
        renal_paths,
        inflammation_paths,
        tuple(slopes_by_source),
        np.mean(np.stack(slopes_by_source), axis=0),
    )


@dataclass(frozen=True)
class TargetAdjustedAssociativeModel(Model):
    name: str = TARGET_ADJUSTED_ASSOCIATIVE

    def fit(self, data: Cohort, rng: Generator, config: TransportModelConfig) -> FittedResidualizedTransportMixture:
        return _fit_residualized(data, rng, config, (True, True, True), (True, True, True))


@dataclass(frozen=True)
class FrozenCausalSCM(Model):
    name: str = FROZEN_CAUSAL

    def fit(self, data: Cohort, rng: Generator, config: TransportModelConfig) -> FittedResidualizedTransportMixture:
        return _fit_residualized(data, rng, config, (True, False, False), (False, False, True))


@dataclass(frozen=True)
class ModularCausalSCM(Model):
    name: str = MODULAR_CAUSAL

    def fit(self, data: Cohort, rng: Generator, config: TransportModelConfig) -> FittedResidualizedTransportMixture:
        return _fit_residualized(data, rng, config, (True, False, False), (False, False, True))


@dataclass(frozen=True)
class FittedTargetTransportOracle(FittedModel):
    simulation: TransportSimulationConfig

    def posterior(self, data: Cohort) -> np.ndarray:
        if not isinstance(data, MultiHospitalCohort) or data.hospital is None:
            raise TypeError("The transport oracle requires hospital metadata.")
        slopes = np.zeros((2, len(BIOMARKER_NAMES)), dtype=float)
        slopes[0, Biomarker.NT_PROBNP] = data.hospital.renal_effect_nt_sd
        slopes[1, Biomarker.COMPETING_VASCULAR] = data.hospital.inflammation_effect_competing_sd
        residualized = remove_nuisance_paths(
            assay_calibrate_observed(data),
            data.covariate("renal_dysfunction"),
            data.covariate("background_inflammation"),
            slopes,
        )
        means = np.asarray((self.simulation.atrial_path_effects_sd, self.simulation.competing_path_effects_sd))
        variance = np.asarray(self.simulation.biomarker_noise_sd, dtype=float) ** 2
        probability = np.asarray((self.simulation.atrial_probability, 1.0 - self.simulation.atrial_probability))
        posterior, _ = e_step(residualized, np.log(probability)[None, :], means, variance)
        return posterior

    @property
    def n_parameters(self) -> int:
        return 0


@dataclass(frozen=True)
class TargetTransportOracle(Model):
    simulation: TransportSimulationConfig
    name: str = ORACLE

    def fit(self, data: Cohort, rng: Generator, config: TransportModelConfig) -> FittedTargetTransportOracle:
        return FittedTargetTransportOracle(self.simulation)


__all__ = [
    "FROZEN_CAUSAL",
    "MODULAR_CAUSAL",
    "ORACLE",
    "POOLED_ASSOCIATIVE",
    "TARGET_ADJUSTED_ASSOCIATIVE",
    "FrozenCausalSCM",
    "MissingGaussianMixtureFit",
    "ModularCausalSCM",
    "PooledAssociativeTransportModel",
    "TargetAdjustedAssociativeModel",
    "TargetTransportOracle",
    "fit_missing_gaussian_mixture",
    "fit_nuisance_paths",
    "missing_gmm_posterior",
    "remove_nuisance_paths",
]
