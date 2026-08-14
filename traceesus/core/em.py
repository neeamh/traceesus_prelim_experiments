"""Provide the single missingness-aware Gaussian-mixture E-step and M-step."""

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


__all__ = [
    "EmissionParameters",
    "diagonal_gaussian_log_density",
    "e_step",
    "has_converged",
    "m_step",
    "normalize_responsibilities",
]
