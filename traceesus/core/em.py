"""Provide the shared missingness-aware unconditional and conditional EM primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class EmissionParameters:
    """Gaussian emission parameters returned by the shared M-step."""

    class_means: np.ndarray
    nuisance_effects: np.ndarray
    variance: np.ndarray


@dataclass(frozen=True)
class ConditionalParameters:
    """Parameters for p(Z|R) p(B|Z,N) under a fixed nuisance path mask."""

    class_probability_by_renal: np.ndarray
    emission: EmissionParameters


@dataclass(frozen=True)
class ConditionalStartResult:
    """Result of one deterministic conditional-EM start."""

    parameters: ConditionalParameters
    responsibility: np.ndarray
    log_likelihood: float
    converged: bool
    iterations: int


def _observed_mask(
    biomarkers: np.ndarray,
    measurement_mask: np.ndarray | None,
) -> np.ndarray:
    """Validate or derive the Boolean matrix of observed measurements."""

    finite = np.isfinite(biomarkers)
    if measurement_mask is None:
        return finite
    observed = np.asarray(measurement_mask, dtype=bool)
    if observed.shape != biomarkers.shape:
        raise ValueError("measurement_mask must match biomarkers.")
    if np.any(observed & ~finite):
        raise ValueError("Observed measurements must contain finite values.")
    return observed


def diagonal_gaussian_log_density(
    biomarkers: np.ndarray,
    class_means: np.ndarray,
    variance: np.ndarray,
    measurement_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return missingness-aware diagonal-Gaussian log densities for every class."""

    observed = _observed_mask(biomarkers, measurement_mask)
    means = np.asarray(class_means, dtype=float)
    if means.ndim == 2:
        means = means[None, :, :]
    variances = np.asarray(variance, dtype=float)
    if variances.ndim == 1:
        variances = variances[None, None, :]
    elif variances.ndim == 2:
        variances = variances[None, :, :]
    filled = np.where(observed, biomarkers, 0.0)
    residual = filled[:, None, :] - means
    contribution = -0.5 * (
        residual**2 / variances + np.log(2.0 * np.pi * variances)
    )
    return np.sum(np.where(observed[:, None, :], contribution, 0.0), axis=2)


def normalize_responsibilities(
    log_joint: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Normalize row log-weights and return responsibilities plus log likelihood."""

    normalizer = logsumexp(log_joint, axis=1)
    responsibility = np.exp(log_joint - normalizer[:, None])
    return responsibility, float(np.sum(normalizer))


def e_step(
    biomarkers: np.ndarray,
    log_prior: np.ndarray,
    class_means: np.ndarray,
    variance: np.ndarray,
    measurement_mask: np.ndarray | None = None,
    nuisance_contribution: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Run the sole E-step for complete or partially observed biomarker data."""

    means = np.asarray(class_means, dtype=float)
    if nuisance_contribution is not None:
        means = means[None, :, :] + nuisance_contribution[:, None, :]
    density = diagonal_gaussian_log_density(
        biomarkers,
        means,
        variance,
        measurement_mask,
    )
    prior = np.asarray(log_prior, dtype=float)
    if prior.shape != density.shape:
        prior = np.broadcast_to(prior, density.shape)
    return normalize_responsibilities(prior + density)


def _masked_component_regression(
    response: np.ndarray,
    observed: np.ndarray,
    responsibility: np.ndarray,
    nuisance_design: np.ndarray,
    included: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate class intercepts and permitted shared nuisance slopes."""

    available = np.flatnonzero(observed)
    component_count = responsibility.shape[1]
    included_columns = np.flatnonzero(included)
    class_design = np.repeat(np.eye(component_count), available.size, axis=0)
    nuisance = np.tile(nuisance_design[available], (component_count, 1))
    design = np.column_stack((class_design, nuisance[:, included_columns]))
    weights = responsibility[available].T.reshape(-1)
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_response = np.tile(response[available], component_count) * np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_design,
        weighted_response,
        rcond=None,
    )
    effects = np.zeros(nuisance_design.shape[1], dtype=float)
    effects[included_columns] = coefficients[component_count:]
    return coefficients[:component_count], effects


def _emission_means(
    biomarkers: np.ndarray,
    observed: np.ndarray,
    responsibility: np.ndarray,
    nuisance_design: np.ndarray,
    path_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate masked emission intercepts and nuisance effects by marker."""

    component_count = responsibility.shape[1]
    marker_count = biomarkers.shape[1]
    if nuisance_design.shape[1] == 0:
        filled = np.where(observed, biomarkers, 0.0)
        denominator = responsibility.T @ observed.astype(float)
        if np.any(denominator <= 1e-8):
            raise FloatingPointError("No effective observations for a class marker.")
        means = (responsibility.T @ filled) / denominator
        return means, np.empty((0, marker_count), dtype=float)
    class_means = np.zeros((component_count, marker_count), dtype=float)
    nuisance_effects = np.zeros((nuisance_design.shape[1], marker_count), dtype=float)
    for marker in range(marker_count):
        means, effects = _masked_component_regression(
            biomarkers[:, marker],
            observed[:, marker],
            responsibility,
            nuisance_design,
            path_mask[:, marker],
        )
        class_means[:, marker] = means
        nuisance_effects[:, marker] = effects
    return class_means, nuisance_effects


def m_step(
    biomarkers: np.ndarray,
    responsibility: np.ndarray,
    variance_floor: float,
    minimum_class_fraction: float,
    measurement_mask: np.ndarray | None = None,
    nuisance_design: np.ndarray | None = None,
    path_mask: np.ndarray | None = None,
    shared_variance: bool = True,
) -> EmissionParameters:
    """Run the sole masked M-step for complete or partially observed data."""

    observed = _observed_mask(biomarkers, measurement_mask)
    patient_count, marker_count = biomarkers.shape
    effective_count = np.sum(responsibility, axis=0)
    if np.any(effective_count < minimum_class_fraction * patient_count):
        raise FloatingPointError("A latent class collapsed below the minimum size.")
    if nuisance_design is None:
        nuisance_design = np.empty((patient_count, 0), dtype=float)
    if path_mask is None:
        path_mask = np.zeros((nuisance_design.shape[1], marker_count), dtype=bool)
    class_means, nuisance_effects = _emission_means(
        biomarkers, observed, responsibility, nuisance_design, path_mask
    )
    nuisance = nuisance_design @ nuisance_effects
    residual = np.where(
        observed[:, None, :],
        biomarkers[:, None, :] - class_means[None, :, :] - nuisance[:, None, :],
        0.0,
    )
    weights = responsibility[:, :, None] * observed[:, None, :]
    numerator = np.sum(weights * residual**2, axis=0)
    denominator = np.sum(weights, axis=0)
    if np.any(denominator <= 1e-8):
        raise FloatingPointError("No effective observations for a class marker.")
    if shared_variance:
        variance = np.sum(numerator, axis=0) / np.sum(denominator, axis=0)
    else:
        variance = numerator / denominator
    variance = np.maximum(variance, variance_floor)
    return EmissionParameters(class_means, nuisance_effects, variance)


def has_converged(
    previous: float,
    current: float,
    relative_tolerance: float,
) -> bool:
    """Apply the retained monotone relative log-likelihood stopping rule."""

    improvement = current - previous
    tolerance = relative_tolerance * (1.0 + abs(previous))
    return bool(
        np.isfinite(previous)
        and improvement >= -1e-7
        and improvement <= tolerance
    )


def _conditional_inputs(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisance_design: np.ndarray,
    path_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the fixed shapes shared by conditional E- and M-steps."""

    renal_values = np.asarray(renal)
    design = np.asarray(nuisance_design, dtype=float)
    paths = np.asarray(path_mask, dtype=bool)
    patient_count, marker_count = biomarkers.shape
    if renal_values.shape != (patient_count,):
        raise ValueError("renal must contain one value per patient.")
    if design.ndim != 2 or design.shape[0] != patient_count:
        raise ValueError("nuisance_design must be (patient_count, nuisance_count).")
    if paths.shape != (design.shape[1], marker_count):
        raise ValueError("path_mask must be (nuisance_count, biomarker_count).")
    if not np.isin(renal_values, (0, 1)).all():
        raise ValueError("renal must contain only zero and one.")
    return renal_values.astype(int, copy=False), design, paths


def _conditional_class_probability(
    responsibility: np.ndarray,
    renal: np.ndarray,
    beta_prior_pseudocount: float,
    probability_floor: float,
) -> np.ndarray:
    """Update the retained smoothed two-class p(Z|R) term."""

    result = np.zeros((2, responsibility.shape[1]), dtype=float)
    for renal_value in (0, 1):
        stratum = renal == renal_value
        stratum_count = int(np.sum(stratum))
        if stratum_count == 0:
            raise FloatingPointError("A renal stratum was empty.")
        atrial = (
            float(np.sum(responsibility[stratum, 0])) + beta_prior_pseudocount
        ) / (stratum_count + 2.0 * beta_prior_pseudocount)
        atrial = float(np.clip(atrial, probability_floor, 1.0 - probability_floor))
        result[renal_value] = (atrial, 1.0 - atrial)
    return result


def conditional_m_step(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisance_design: np.ndarray,
    path_mask: np.ndarray,
    responsibility: np.ndarray,
    variance_floor: float,
    minimum_class_fraction: float,
    beta_prior_pseudocount: float,
    probability_floor: float,
    measurement_mask: np.ndarray | None = None,
) -> ConditionalParameters:
    """Run the sole M-step for p(Z|R) p(B|Z,N) with masked nuisance paths."""

    renal_values, design, paths = _conditional_inputs(
        biomarkers, renal, nuisance_design, path_mask
    )
    class_probability = _conditional_class_probability(
        responsibility,
        renal_values,
        beta_prior_pseudocount,
        probability_floor,
    )
    emission = m_step(
        biomarkers,
        responsibility,
        variance_floor,
        minimum_class_fraction,
        measurement_mask,
        design,
        paths,
    )
    return ConditionalParameters(class_probability, emission)


def conditional_e_step(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisance_design: np.ndarray,
    path_mask: np.ndarray,
    parameters: ConditionalParameters,
    measurement_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Run the sole E-step for p(Z|R) p(B|Z,N) with masked nuisance paths."""

    renal_values, design, paths = _conditional_inputs(
        biomarkers, renal, nuisance_design, path_mask
    )
    effects = parameters.emission.nuisance_effects
    if effects.shape != paths.shape:
        raise ValueError("Fitted nuisance effects must match path_mask.")
    if np.any(effects[~paths] != 0.0):
        raise ValueError("Fitted nuisance effects violate path_mask.")
    return e_step(
        biomarkers,
        np.log(parameters.class_probability_by_renal[renal_values]),
        parameters.emission.class_means,
        parameters.emission.variance,
        measurement_mask,
        design @ effects,
    )


def run_conditional_em_start(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    nuisance_design: np.ndarray,
    path_mask: np.ndarray,
    responsibility: np.ndarray,
    maximum_iterations: int,
    relative_tolerance: float,
    variance_floor: float,
    minimum_class_fraction: float,
    beta_prior_pseudocount: float,
    probability_floor: float,
    measurement_mask: np.ndarray | None = None,
) -> ConditionalStartResult:
    """Run one deterministic conditional-EM start through the sole E- and M-steps."""

    previous = -np.inf
    converged = False
    for iteration in range(1, maximum_iterations + 1):
        parameters = conditional_m_step(
            biomarkers, renal, nuisance_design, path_mask, responsibility,
            variance_floor, minimum_class_fraction, beta_prior_pseudocount,
            probability_floor, measurement_mask,
        )
        responsibility, log_likelihood = conditional_e_step(
            biomarkers, renal, nuisance_design, path_mask, parameters,
            measurement_mask,
        )
        if has_converged(previous, log_likelihood, relative_tolerance):
            converged = True
            break
        previous = log_likelihood
    return ConditionalStartResult(
        parameters, responsibility, log_likelihood, converged, iteration
    )


__all__ = [
    "ConditionalParameters",
    "ConditionalStartResult",
    "EmissionParameters",
    "conditional_e_step",
    "conditional_m_step",
    "diagonal_gaussian_log_density",
    "e_step",
    "has_converged",
    "m_step",
    "normalize_responsibilities",
    "run_conditional_em_start",
]
