"""Hidden-label R21 experiment: associative latent classes versus a latent SCM.

The simulator stores each patient's true categorical mechanism, but the fitting
functions never receive those labels.  The labels are used only after fitting
to evaluate out-of-sample recovery.

Primary comparison
------------------
Associative latent class model
    p(Z) p(R | Z) prod_j p(B_j | Z)

Causal latent variable model
    p(Z | R) prod_j p(B_j | Z, R), with the biologically prespecified
    direct renal path restricted to the NT-proBNP-like biomarker.

Both primary models have 12 free parameters when K=2.  A more flexible
renal-adjusted associative latent class regression is included as a sensitivity
control.  It estimates renal associations for every biomarker and therefore has
14 free parameters.

The code uses only NumPy/SciPy/Pandas/Matplotlib.  It implements the EM
algorithms directly so that every likelihood, constraint, initialization, and
model-selection penalty is visible.
"""

from __future__ import annotations

import argparse
import json
import platform
from time import perf_counter
from dataclasses import asdict, dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logsumexp

from traceesus.core.metrics import (
    adjusted_rand_index as _shared_adjusted_rand_index,
    evaluate_binary_posterior,
    expected_calibration_error as _shared_expected_calibration_error,
)
from traceesus.core.io import write_manifest
from traceesus.core.runner import latent_null_seed_ledger, ordered_map
from traceesus.queries.posterior import anchor_order as _shared_anchor_order
from traceesus.core.stats import (
    bic as _shared_bic,
    monte_carlo_summary,
    paired_mean_contrast,
    wilson_interval as _shared_wilson_interval,
)
from traceesus.simulators.two_mechanism import TwoMechanismSimulator
from configs.endotype_discovery import ExperimentConfig, FittingConfig, SimulationConfig


class Mechanism(IntEnum):
    """The single categorical mechanism assigned to each simulated patient."""

    ATRIAL = 0
    COMPETING = 1


class Biomarker(IntEnum):
    """Column positions in the biomarker matrix.

    The clinical names are primary.  The legacy members are aliases for the
    same values (IntEnum treats duplicate values as aliases), so every cited
    kernel path and locked output column keeps resolving unchanged.
    """

    NT_PROBNP = 0
    PTFV1 = 1
    COMPETING_VASCULAR = 2

    # Legacy aliases — retained because locked outputs and kernel internals
    # reference these names.  Do not remove.
    NT_PROBNP_LIKE = 0
    ATRIAL_ELECTRICAL = 1
    COMPETING_SPECIFIC = 2


# Provenance-locked strings: these appear in cited output files and metadata.
# Figures should use BIOMARKER_DISPLAY_NAMES instead.
BIOMARKER_NAMES = (
    "NT-proBNP-like biomarker",
    "Atrial electrical evidence",
    "Competing-mechanism evidence",
)

# Clinically legible names for figures and tables only.
BIOMARKER_DISPLAY_NAMES = (
    "NT-proBNP",
    "PTFV1",
    "Competing-vascular",
)

ASSOCIATIVE_LCA = "Associative latent class model"
ASSOCIATIVE_ADJUSTED = "Renal-adjusted associative latent class model"
CAUSAL_SCM = "Biologically constrained latent SCM"
ORACLE = "Data-generating oracle (reference)"

FITTED_METHODS = (ASSOCIATIVE_LCA, ASSOCIATIVE_ADJUSTED, CAUSAL_SCM)
PRIMARY_METHODS = (ASSOCIATIVE_LCA, CAUSAL_SCM)
ALL_METHODS = (*FITTED_METHODS, ORACLE)


@dataclass(frozen=True)
class SimulatedCohort:
    """One generated cohort; truth is kept separate from observed inputs.

    ``heart_failure`` is an observed nuisance covariate like renal status.  The
    null cohort sets it to all zeros rather than drawing it, so the locked K=1
    experiment consumes exactly its historical RNG sequence.
    """

    biomarkers: np.ndarray
    renal_dysfunction: np.ndarray
    true_mechanism: np.ndarray
    heart_failure: np.ndarray

    @property
    def observed_matrix(self) -> np.ndarray:
        """Expose fit-eligible columns while structurally excluding mechanism truth."""

        return np.column_stack((self.renal_dysfunction, self.biomarkers))


@dataclass(frozen=True)
class AssociativeLatentClassFit:
    """Fit of p(Z) p(R|Z) p(B|Z), with shared diagonal biomarker variance."""

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


@dataclass(frozen=True)
class ConditionalLatentFit:
    """Fit of p(Z|R) p(B|Z,R) for either adjusted LCA or causal SCM."""

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


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def simulate_two_mechanism_cohort(
    rng: np.random.Generator,
    patient_count: int,
    renal_effect_sd: float,
    config: SimulationConfig,
    *,
    heart_failure_effect_sd: float = 0.0,
) -> SimulatedCohort:
    """Adapt the shared simulator to the kernel's historical cohort container."""

    generated = TwoMechanismSimulator(
        config,
        renal_effect_sd,
        heart_failure_effect_sd,
    ).simulate(rng, patient_count)
    return SimulatedCohort(
        biomarkers=generated.observed.biomarkers,
        renal_dysfunction=generated.observed.covariate("renal_dysfunction"),
        true_mechanism=generated.truth.mechanism,
        heart_failure=generated.observed.covariate("heart_failure"),
    )


def simulate_one_mechanism_null_cohort(
    rng: np.random.Generator,
    patient_count: int,
    renal_effect_sd: float,
    config: SimulationConfig,
) -> SimulatedCohort:
    """Adapt the shared null method without consuming a heart-failure draw."""

    generated = TwoMechanismSimulator(config, renal_effect_sd).simulate_null(
        rng, patient_count
    )
    return SimulatedCohort(
        biomarkers=generated.observed.biomarkers,
        renal_dysfunction=generated.observed.covariate("renal_dysfunction"),
        true_mechanism=generated.truth.mechanism,
        heart_failure=generated.observed.covariate("heart_failure"),
    )


def _clip_probability(values: np.ndarray | float, config: FittingConfig) -> np.ndarray:
    return np.clip(values, config.probability_floor, 1.0 - config.probability_floor)


def _shared_diagonal_gaussian_log_density(
    biomarkers: np.ndarray,
    means: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    """Return n x K log densities under shared diagonal variances."""

    residual = biomarkers[:, None, :] - means
    return -0.5 * (
        np.sum(residual**2 / variance[None, None, :], axis=2)
        + np.sum(np.log(2.0 * np.pi * variance))
    )


def _initial_responsibilities(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    rng: np.random.Generator,
    start_index: int,
    config: FittingConfig,
) -> np.ndarray:
    """Create reproducible, non-degenerate two-class EM initializations."""

    biomarker_sd = np.std(biomarkers, axis=0)
    biomarker_sd = np.where(biomarker_sd > 1e-8, biomarker_sd, 1.0)
    standardized = (biomarkers - np.mean(biomarkers, axis=0)) / biomarker_sd

    if start_index == 0:
        projection = (
            standardized[:, Biomarker.ATRIAL_ELECTRICAL]
            - standardized[:, Biomarker.COMPETING_SPECIFIC]
        )
    elif start_index == 1:
        projection = standardized[:, Biomarker.NT_PROBNP_LIKE]
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
    probability_atrial = expit(1.5 * (projection - np.median(projection)) / projection_sd)
    probability_atrial = _clip_probability(probability_atrial, config)
    return np.column_stack((probability_atrial, 1.0 - probability_atrial))


def _anchor_order(
    class_means: np.ndarray,
    biomarker_variance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Orient latent labels without using simulated truth.

    The atrial class is the class with the larger prespecified anchor contrast:
    atrial electrical evidence minus competing-mechanism evidence.  The
    NT-proBNP-like biomarker is deliberately excluded from the label anchor
    because it is the biomarker distorted by renal dysfunction.
    """

    return _shared_anchor_order(
        class_means,
        biomarker_variance,
        atrial_electrical_index=Biomarker.ATRIAL_ELECTRICAL,
        competing_specific_index=Biomarker.COMPETING_SPECIFIC,
    )


def _associative_m_step(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    responsibility: np.ndarray,
    config: FittingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    effective_count = np.sum(responsibility, axis=0)
    minimum_count = config.minimum_effective_class_fraction * biomarkers.shape[0]
    if np.any(effective_count < minimum_count):
        raise FloatingPointError("A latent class collapsed below the minimum size.")

    smoothing = config.beta_prior_pseudocount
    class_probability = (effective_count + smoothing) / (
        biomarkers.shape[0] + 2.0 * smoothing
    )
    renal_probability = (
        np.sum(responsibility * renal[:, None], axis=0) + smoothing
    ) / (effective_count + 2.0 * smoothing)
    renal_probability = _clip_probability(renal_probability, config)
    class_means = (
        responsibility.T @ biomarkers
    ) / effective_count[:, None]
    residual = biomarkers[:, None, :] - class_means[None, :, :]
    variance = np.sum(
        responsibility[:, :, None] * residual**2,
        axis=(0, 1),
    ) / biomarkers.shape[0]
    variance = np.maximum(variance, config.variance_floor)
    return class_probability, renal_probability, class_means, variance


def _associative_e_step(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    class_probability: np.ndarray,
    renal_probability: np.ndarray,
    class_means: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, float]:
    log_joint = (
        np.log(class_probability)[None, :]
        + renal[:, None] * np.log(renal_probability)[None, :]
        + (1 - renal[:, None]) * np.log(1.0 - renal_probability)[None, :]
        + _shared_diagonal_gaussian_log_density(
            biomarkers,
            class_means[None, :, :],
            variance,
        )
    )
    log_normalizer = logsumexp(log_joint, axis=1)
    responsibility = np.exp(log_joint - log_normalizer[:, None])
    return responsibility, float(np.sum(log_normalizer))


def fit_associative_latent_class_model(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    rng: np.random.Generator,
    config: FittingConfig,
) -> AssociativeLatentClassFit:
    """Fit the unlabeled associative two-class model by multi-start EM."""

    best: dict[str, object] | None = None
    for start_index in range(config.random_starts):
        responsibility = _initial_responsibilities(
            biomarkers,
            renal,
            rng,
            start_index,
            config,
        )
        converged = False
        previous_log_likelihood = -np.inf

        try:
            for iteration in range(1, config.maximum_em_iterations + 1):
                parameters = _associative_m_step(
                    biomarkers,
                    renal,
                    responsibility,
                    config,
                )
                responsibility, log_likelihood = _associative_e_step(
                    biomarkers,
                    renal,
                    *parameters,
                )
                improvement = log_likelihood - previous_log_likelihood
                tolerance = config.relative_log_likelihood_tolerance * (
                    1.0 + abs(previous_log_likelihood)
                )
                if (
                    np.isfinite(previous_log_likelihood)
                    and improvement >= -1e-7
                    and improvement <= tolerance
                ):
                    converged = True
                    break
                previous_log_likelihood = log_likelihood
        except (FloatingPointError, np.linalg.LinAlgError):
            continue

        if best is None or log_likelihood > float(best["log_likelihood"]):
            best = {
                "parameters": parameters,
                "responsibility": responsibility,
                "log_likelihood": log_likelihood,
                "converged": converged,
                "iterations": iteration,
                "best_start": start_index,
            }

    if best is None:
        raise RuntimeError("All associative latent-class EM starts failed.")

    class_probability, renal_probability, class_means, variance = best["parameters"]
    order, anchor_margin = _anchor_order(class_means, variance)
    responsibility = best["responsibility"][:, order]
    effective_fraction = np.mean(responsibility, axis=0)

    return AssociativeLatentClassFit(
        class_probability=class_probability[order],
        renal_probability_by_class=renal_probability[order],
        class_means=class_means[order],
        biomarker_variance=variance,
        log_likelihood=float(best["log_likelihood"]),
        converged=bool(best["converged"]),
        iterations=int(best["iterations"]),
        best_start=int(best["best_start"]),
        effective_class_fraction=effective_fraction,
        anchor_margin=anchor_margin,
    )


def associative_posterior(
    fit: AssociativeLatentClassFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
) -> np.ndarray:
    """Evaluate the oriented associative fit without accepting simulator truth.

    The fitted class order already encodes the prespecified biomarker anchor;
    reorienting against labels here would leak truth into discovery.
    """

    responsibility, _ = _associative_e_step(
        biomarkers,
        renal,
        fit.class_probability,
        fit.renal_probability_by_class,
        fit.class_means,
        fit.biomarker_variance,
    )
    return responsibility


def _weighted_component_regression(
    response: np.ndarray,
    renal: np.ndarray,
    responsibility: np.ndarray,
    include_renal_effect: bool,
) -> tuple[np.ndarray, float]:
    """Weighted M-step for two component intercepts and an optional shared slope."""

    patient_count = response.shape[0]
    component_count = responsibility.shape[1]
    parameter_count = component_count + int(include_renal_effect)
    design = np.zeros((patient_count * component_count, parameter_count), dtype=float)
    tiled_response = np.tile(response, component_count)
    weights = np.concatenate(
        [responsibility[:, component] for component in range(component_count)]
    )

    for component in range(component_count):
        row_slice = slice(component * patient_count, (component + 1) * patient_count)
        design[row_slice, component] = 1.0
        if include_renal_effect:
            design[row_slice, -1] = renal

    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_response = tiled_response * np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_design,
        weighted_response,
        rcond=None,
    )
    renal_effect = float(coefficients[-1]) if include_renal_effect else 0.0
    return coefficients[:component_count], renal_effect


def _conditional_m_step(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    responsibility: np.ndarray,
    renal_path_mask: np.ndarray,
    config: FittingConfig,
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
        weighted_atrial_count = float(np.sum(responsibility[stratum, Mechanism.ATRIAL]))
        atrial_probability = (weighted_atrial_count + smoothing) / (
            stratum_count + 2.0 * smoothing
        )
        atrial_probability = float(_clip_probability(atrial_probability, config))
        class_probability_by_renal[renal_value] = (
            atrial_probability,
            1.0 - atrial_probability,
        )

    class_means = np.zeros((2, biomarkers.shape[1]), dtype=float)
    renal_effect = np.zeros(biomarkers.shape[1], dtype=float)
    for biomarker_index in range(biomarkers.shape[1]):
        class_means[:, biomarker_index], renal_effect[biomarker_index] = (
            _weighted_component_regression(
                biomarkers[:, biomarker_index],
                renal,
                responsibility,
                bool(renal_path_mask[biomarker_index]),
            )
        )

    conditional_means = (
        class_means[None, :, :]
        + renal[:, None, None] * renal_effect[None, None, :]
    )
    residual = biomarkers[:, None, :] - conditional_means
    variance = np.sum(
        responsibility[:, :, None] * residual**2,
        axis=(0, 1),
    ) / biomarkers.shape[0]
    variance = np.maximum(variance, config.variance_floor)
    return class_probability_by_renal, class_means, renal_effect, variance


def _conditional_e_step(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    class_probability_by_renal: np.ndarray,
    class_means: np.ndarray,
    renal_effect: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, float]:
    patient_means = (
        class_means[None, :, :]
        + renal[:, None, None] * renal_effect[None, None, :]
    )
    log_joint = (
        np.log(class_probability_by_renal[renal])
        + _shared_diagonal_gaussian_log_density(
            biomarkers,
            patient_means,
            variance,
        )
    )
    log_normalizer = logsumexp(log_joint, axis=1)
    responsibility = np.exp(log_joint - log_normalizer[:, None])
    return responsibility, float(np.sum(log_normalizer))


def fit_conditional_latent_model(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_path_mask: np.ndarray,
    rng: np.random.Generator,
    config: FittingConfig,
) -> ConditionalLatentFit:
    """Fit a two-class conditional mixture with prespecified renal paths."""

    renal_path_mask = np.asarray(renal_path_mask, dtype=bool)
    if renal_path_mask.shape != (biomarkers.shape[1],):
        raise ValueError("renal_path_mask must contain one Boolean per biomarker.")

    best: dict[str, object] | None = None
    for start_index in range(config.random_starts):
        responsibility = _initial_responsibilities(
            biomarkers,
            renal,
            rng,
            start_index,
            config,
        )
        converged = False
        previous_log_likelihood = -np.inf

        try:
            for iteration in range(1, config.maximum_em_iterations + 1):
                parameters = _conditional_m_step(
                    biomarkers,
                    renal,
                    responsibility,
                    renal_path_mask,
                    config,
                )
                responsibility, log_likelihood = _conditional_e_step(
                    biomarkers,
                    renal,
                    *parameters,
                )
                improvement = log_likelihood - previous_log_likelihood
                tolerance = config.relative_log_likelihood_tolerance * (
                    1.0 + abs(previous_log_likelihood)
                )
                if (
                    np.isfinite(previous_log_likelihood)
                    and improvement >= -1e-7
                    and improvement <= tolerance
                ):
                    converged = True
                    break
                previous_log_likelihood = log_likelihood
        except (FloatingPointError, np.linalg.LinAlgError):
            continue

        if best is None or log_likelihood > float(best["log_likelihood"]):
            best = {
                "parameters": parameters,
                "responsibility": responsibility,
                "log_likelihood": log_likelihood,
                "converged": converged,
                "iterations": iteration,
                "best_start": start_index,
            }

    if best is None:
        raise RuntimeError("All conditional latent-model EM starts failed.")

    class_probability_by_renal, class_means, renal_effect, variance = best["parameters"]
    order, anchor_margin = _anchor_order(class_means, variance)
    responsibility = best["responsibility"][:, order]
    effective_fraction = np.mean(responsibility, axis=0)

    return ConditionalLatentFit(
        class_probability_by_renal=class_probability_by_renal[:, order],
        class_means_at_renal_normal=class_means[order],
        renal_effect=renal_effect,
        renal_path_mask=renal_path_mask,
        biomarker_variance=variance,
        log_likelihood=float(best["log_likelihood"]),
        converged=bool(best["converged"]),
        iterations=int(best["iterations"]),
        best_start=int(best["best_start"]),
        effective_class_fraction=effective_fraction,
        anchor_margin=anchor_margin,
    )


def conditional_posterior(
    fit: ConditionalLatentFit,
    biomarkers: np.ndarray,
    renal: np.ndarray,
) -> np.ndarray:
    """Evaluate a renal-conditional fit using only observed biomarkers and renal status.

    Truth is deliberately absent from the signature so it can enter only the
    downstream simulation evaluation.
    """

    responsibility, _ = _conditional_e_step(
        biomarkers,
        renal,
        fit.class_probability_by_renal,
        fit.class_means_at_renal_normal,
        fit.renal_effect,
        fit.biomarker_variance,
    )
    return responsibility


def oracle_posterior(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: SimulationConfig,
) -> np.ndarray:
    """Bayes posterior under the known simulator; used only as a ceiling."""

    class_probability_by_renal = np.empty((2, 2), dtype=float)
    class_probability_by_renal[0] = (
        config.atrial_probability_if_renal_normal,
        1.0 - config.atrial_probability_if_renal_normal,
    )
    class_probability_by_renal[1] = (
        config.atrial_probability_if_renal_impaired,
        1.0 - config.atrial_probability_if_renal_impaired,
    )
    class_means = np.stack(
        (
            _as_float_array(config.atrial_path_effects_sd),
            _as_float_array(config.competing_path_effects_sd),
        ),
        axis=0,
    )
    renal_effect = np.zeros(len(BIOMARKER_NAMES), dtype=float)
    renal_effect[Biomarker.NT_PROBNP_LIKE] = renal_effect_sd
    variance = _as_float_array(config.biomarker_noise_sd) ** 2
    responsibility, _ = _conditional_e_step(
        biomarkers,
        renal,
        class_probability_by_renal,
        class_means,
        renal_effect,
        variance,
    )
    return responsibility


def _adjusted_rand_index(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Adjusted Rand index without a scikit-learn dependency."""

    return _shared_adjusted_rand_index(truth, prediction)


def _expected_calibration_error(
    atrial_probability: np.ndarray,
    is_atrial: np.ndarray,
    bin_count: int,
) -> float:
    return _shared_expected_calibration_error(
        atrial_probability,
        is_atrial,
        bin_count,
    )


def evaluate_posterior(
    posterior: np.ndarray,
    truth: np.ndarray,
    renal: np.ndarray,
    calibration_bins: int,
) -> dict[str, float]:
    """Compute prespecified recovery, subgroup-error, and calibration metrics."""

    return evaluate_binary_posterior(
        posterior,
        truth,
        renal,
        calibration_bins,
        atrial=Mechanism.ATRIAL,
        competing=Mechanism.COMPETING,
    )


def _fit_diagnostic_row(
    method: str,
    fit: AssociativeLatentClassFit | ConditionalLatentFit,
) -> dict[str, float | int | bool | str]:
    return {
        "method": method,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "best_start": fit.best_start,
        "log_likelihood": fit.log_likelihood,
        "minimum_effective_class_fraction": float(
            np.min(fit.effective_class_fraction)
        ),
        "anchor_margin": fit.anchor_margin,
    }


def _run_one_repeat(task: tuple[int, int, float, ExperimentConfig]) -> dict[str, object]:
    repeat_index, seed, renal_effect_sd, config = task
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(8)
    simulation_rng = np.random.default_rng(child_sequences[0])
    test_rng = np.random.default_rng(child_sequences[1])
    associative_rng = np.random.default_rng(child_sequences[2])
    adjusted_rng = np.random.default_rng(child_sequences[3])
    causal_rng = np.random.default_rng(child_sequences[4])

    training = simulate_two_mechanism_cohort(
        simulation_rng,
        config.simulation.training_patients,
        renal_effect_sd,
        config.simulation,
    )
    test = simulate_two_mechanism_cohort(
        test_rng,
        config.simulation.test_patients,
        renal_effect_sd,
        config.simulation,
    )

    # Truth is intentionally not passed to any fitting function.
    associative_fit = fit_associative_latent_class_model(
        training.biomarkers,
        training.renal_dysfunction,
        associative_rng,
        config.fitting,
    )
    adjusted_fit = fit_conditional_latent_model(
        training.biomarkers,
        training.renal_dysfunction,
        np.ones(len(BIOMARKER_NAMES), dtype=bool),
        adjusted_rng,
        config.fitting,
    )
    causal_mask = np.zeros(len(BIOMARKER_NAMES), dtype=bool)
    causal_mask[Biomarker.NT_PROBNP_LIKE] = True
    causal_fit = fit_conditional_latent_model(
        training.biomarkers,
        training.renal_dysfunction,
        causal_mask,
        causal_rng,
        config.fitting,
    )
    retry_fitting = replace(
        config.fitting,
        random_starts=max(config.fitting.random_starts, 8),
        maximum_em_iterations=max(config.fitting.maximum_em_iterations, 1_200),
    )
    if not associative_fit.converged:
        associative_fit = fit_associative_latent_class_model(
            training.biomarkers,
            training.renal_dysfunction,
            np.random.default_rng(child_sequences[5]),
            retry_fitting,
        )
    if not adjusted_fit.converged:
        adjusted_fit = fit_conditional_latent_model(
            training.biomarkers,
            training.renal_dysfunction,
            np.ones(len(BIOMARKER_NAMES), dtype=bool),
            np.random.default_rng(child_sequences[6]),
            retry_fitting,
        )
    if not causal_fit.converged:
        causal_fit = fit_conditional_latent_model(
            training.biomarkers,
            training.renal_dysfunction,
            causal_mask,
            np.random.default_rng(child_sequences[7]),
            retry_fitting,
        )

    posterior_by_method = {
        ASSOCIATIVE_LCA: associative_posterior(
            associative_fit,
            test.biomarkers,
            test.renal_dysfunction,
        ),
        ASSOCIATIVE_ADJUSTED: conditional_posterior(
            adjusted_fit,
            test.biomarkers,
            test.renal_dysfunction,
        ),
        CAUSAL_SCM: conditional_posterior(
            causal_fit,
            test.biomarkers,
            test.renal_dysfunction,
        ),
        ORACLE: oracle_posterior(
            test.biomarkers,
            test.renal_dysfunction,
            renal_effect_sd,
            config.simulation,
        ),
    }

    metric_rows: list[dict[str, object]] = []
    for method, posterior in posterior_by_method.items():
        row: dict[str, object] = {
            "repeat": repeat_index,
            "renal_effect_sd": renal_effect_sd,
            "method": method,
        }
        row.update(
            evaluate_posterior(
                posterior,
                test.true_mechanism,
                test.renal_dysfunction,
                config.fitting.calibration_bins,
            )
        )
        metric_rows.append(row)

    diagnostic_rows = []
    for method, fit in (
        (ASSOCIATIVE_LCA, associative_fit),
        (ASSOCIATIVE_ADJUSTED, adjusted_fit),
        (CAUSAL_SCM, causal_fit),
    ):
        row = {
            "repeat": repeat_index,
            "renal_effect_sd": renal_effect_sd,
        }
        row.update(_fit_diagnostic_row(method, fit))
        diagnostic_rows.append(row)

    parameter_row = {
        "repeat": repeat_index,
        "renal_effect_sd": renal_effect_sd,
        "causal_estimated_renal_effect_nt": causal_fit.renal_effect[
            Biomarker.NT_PROBNP_LIKE
        ],
        "adjusted_estimated_renal_effect_nt": adjusted_fit.renal_effect[
            Biomarker.NT_PROBNP_LIKE
        ],
        "adjusted_estimated_renal_effect_electrical": adjusted_fit.renal_effect[
            Biomarker.ATRIAL_ELECTRICAL
        ],
        "adjusted_estimated_renal_effect_competing": adjusted_fit.renal_effect[
            Biomarker.COMPETING_SPECIFIC
        ],
    }
    return {
        "metrics": metric_rows,
        "diagnostics": diagnostic_rows,
        "parameters": parameter_row,
    }


def _map_tasks(
    worker,
    tasks: list[tuple],
    worker_count: int,
) -> Iterable[dict[str, object]]:
    return ordered_map(worker, tasks, worker_count)


def run_recovery_experiment(
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run paired hidden-label recovery simulations at every renal-effect level."""

    config.validate()
    root_sequence = np.random.SeedSequence(config.master_seed)
    level_sequences = root_sequence.spawn(len(config.simulation.renal_effect_levels_sd))
    tasks: list[tuple[int, int, float, ExperimentConfig]] = []
    for renal_effect_sd, level_sequence in zip(
        config.simulation.renal_effect_levels_sd,
        level_sequences,
        strict=True,
    ):
        repeat_sequences = level_sequence.spawn(config.repeats_per_level)
        for repeat_index, repeat_sequence in enumerate(repeat_sequences):
            seed = int(repeat_sequence.generate_state(1, dtype=np.uint64)[0])
            tasks.append((repeat_index, seed, renal_effect_sd, config))

    results = _map_tasks(_run_one_repeat, tasks, config.workers)
    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for result in results:
        metric_rows.extend(result["metrics"])
        diagnostic_rows.extend(result["diagnostics"])
        parameter_rows.append(result["parameters"])
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(parameter_rows),
    )


def summarize_repeated_metrics(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeat-level metrics with Monte Carlo confidence intervals."""

    identifier_columns = {
        "repeat",
        "renal_effect_sd",
        "method",
        "renal_competing_subgroup_size",
    }
    metric_columns = [
        column for column in raw_metrics.columns if column not in identifier_columns
    ]
    summary_rows: list[dict[str, object]] = []
    for (renal_effect_sd, method), group in raw_metrics.groupby(
        ["renal_effect_sd", "method"],
        sort=False,
    ):
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            statistics = monte_carlo_summary(values)
            summary_rows.append(
                {
                    "renal_effect_sd": renal_effect_sd,
                    "method": method,
                    "metric": metric,
                    **statistics,
                }
            )
    return pd.DataFrame(summary_rows)


def paired_method_contrasts(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute paired causal-minus-comparator contrasts within repeat."""

    metrics = (
        "accuracy",
        "adjusted_rand_index",
        "false_atrial_renal_competing",
        "brier_score",
        "expected_calibration_error",
    )
    comparators = (ASSOCIATIVE_LCA, ASSOCIATIVE_ADJUSTED)
    rows: list[dict[str, object]] = []
    for renal_effect_sd in sorted(raw_metrics["renal_effect_sd"].unique()):
        level = raw_metrics[raw_metrics["renal_effect_sd"] == renal_effect_sd]
        for comparator in comparators:
            for metric in metrics:
                wide = level.pivot(index="repeat", columns="method", values=metric)
                difference = (
                    wide[CAUSAL_SCM].to_numpy(dtype=float)
                    - wide[comparator].to_numpy(dtype=float)
                )
                statistics = paired_mean_contrast(difference)
                rows.append(
                    {
                        "renal_effect_sd": renal_effect_sd,
                        "comparator": comparator,
                        "metric": metric,
                        "difference_definition": "causal SCM minus comparator",
                        **statistics,
                    }
                )
    return pd.DataFrame(rows)


def _fit_k1_associative(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    config: FittingConfig,
) -> tuple[float, int]:
    renal_probability = float(_clip_probability(np.mean(renal), config))
    means = np.mean(biomarkers, axis=0, keepdims=True)
    variance = np.maximum(np.var(biomarkers, axis=0), config.variance_floor)
    log_likelihood = (
        np.sum(
            renal * np.log(renal_probability)
            + (1 - renal) * np.log(1.0 - renal_probability)
        )
        + np.sum(
            _shared_diagonal_gaussian_log_density(
                biomarkers,
                means[None, :, :],
                variance,
            )
        )
    )
    parameter_count = 1 + biomarkers.shape[1] + biomarkers.shape[1]
    return float(log_likelihood), parameter_count


def _fit_k1_conditional(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_path_mask: np.ndarray,
    config: FittingConfig,
) -> tuple[float, int]:
    responsibility = np.ones((biomarkers.shape[0], 1), dtype=float)
    means = np.zeros((1, biomarkers.shape[1]), dtype=float)
    renal_effect = np.zeros(biomarkers.shape[1], dtype=float)
    for biomarker_index in range(biomarkers.shape[1]):
        means[:, biomarker_index], renal_effect[biomarker_index] = (
            _weighted_component_regression(
                biomarkers[:, biomarker_index],
                renal,
                responsibility,
                bool(renal_path_mask[biomarker_index]),
            )
        )
    patient_mean = means[0] + renal[:, None] * renal_effect
    variance = np.maximum(
        np.mean((biomarkers - patient_mean) ** 2, axis=0),
        config.variance_floor,
    )
    log_likelihood = float(
        np.sum(
            _shared_diagonal_gaussian_log_density(
                biomarkers,
                patient_mean[:, None, :],
                variance,
            )
        )
    )
    parameter_count = (
        biomarkers.shape[1]
        + int(np.sum(renal_path_mask))
        + biomarkers.shape[1]
    )
    return log_likelihood, parameter_count


def _bic(log_likelihood: float, parameter_count: int, patient_count: int) -> float:
    return _shared_bic(log_likelihood, parameter_count, patient_count)


def _run_one_null_repeat(
    task: tuple[int, int, ExperimentConfig],
) -> list[dict[str, object]]:
    repeat_index, seed, config = task
    seed_sequence = np.random.SeedSequence(seed)
    sequences = seed_sequence.spawn(7)
    cohort = simulate_one_mechanism_null_cohort(
        np.random.default_rng(sequences[0]),
        config.simulation.training_patients,
        config.null_renal_effect_sd,
        config.simulation,
    )
    patient_count = cohort.biomarkers.shape[0]
    causal_mask = np.zeros(len(BIOMARKER_NAMES), dtype=bool)
    causal_mask[Biomarker.NT_PROBNP_LIKE] = True
    adjusted_mask = np.ones(len(BIOMARKER_NAMES), dtype=bool)

    associative_k1_ll, associative_k1_parameters = _fit_k1_associative(
        cohort.biomarkers,
        cohort.renal_dysfunction,
        config.fitting,
    )
    associative_k2 = fit_associative_latent_class_model(
        cohort.biomarkers,
        cohort.renal_dysfunction,
        np.random.default_rng(sequences[1]),
        config.fitting,
    )

    adjusted_k1_ll, adjusted_k1_parameters = _fit_k1_conditional(
        cohort.biomarkers,
        cohort.renal_dysfunction,
        adjusted_mask,
        config.fitting,
    )
    adjusted_k2 = fit_conditional_latent_model(
        cohort.biomarkers,
        cohort.renal_dysfunction,
        adjusted_mask,
        np.random.default_rng(sequences[2]),
        config.fitting,
    )

    causal_k1_ll, causal_k1_parameters = _fit_k1_conditional(
        cohort.biomarkers,
        cohort.renal_dysfunction,
        causal_mask,
        config.fitting,
    )
    causal_k2 = fit_conditional_latent_model(
        cohort.biomarkers,
        cohort.renal_dysfunction,
        causal_mask,
        np.random.default_rng(sequences[3]),
        config.fitting,
    )
    retry_fitting = replace(
        config.fitting,
        random_starts=max(config.fitting.random_starts, 8),
        maximum_em_iterations=max(config.fitting.maximum_em_iterations, 1_200),
    )
    if not associative_k2.converged:
        associative_k2 = fit_associative_latent_class_model(
            cohort.biomarkers,
            cohort.renal_dysfunction,
            np.random.default_rng(sequences[4]),
            retry_fitting,
        )
    if not adjusted_k2.converged:
        adjusted_k2 = fit_conditional_latent_model(
            cohort.biomarkers,
            cohort.renal_dysfunction,
            adjusted_mask,
            np.random.default_rng(sequences[5]),
            retry_fitting,
        )
    if not causal_k2.converged:
        causal_k2 = fit_conditional_latent_model(
            cohort.biomarkers,
            cohort.renal_dysfunction,
            causal_mask,
            np.random.default_rng(sequences[6]),
            retry_fitting,
        )

    fits = (
        (
            ASSOCIATIVE_LCA,
            associative_k1_ll,
            associative_k1_parameters,
            associative_k2,
            12,
        ),
        (
            ASSOCIATIVE_ADJUSTED,
            adjusted_k1_ll,
            adjusted_k1_parameters,
            adjusted_k2,
            14,
        ),
        (
            CAUSAL_SCM,
            causal_k1_ll,
            causal_k1_parameters,
            causal_k2,
            12,
        ),
    )
    rows: list[dict[str, object]] = []
    for method, k1_ll, k1_parameters, k2_fit, k2_parameters in fits:
        bic_k1 = _bic(k1_ll, k1_parameters, patient_count)
        bic_k2 = _bic(k2_fit.log_likelihood, k2_parameters, patient_count)
        rows.append(
            {
                "repeat": repeat_index,
                "method": method,
                "renal_effect_sd": config.null_renal_effect_sd,
                "bic_k1": bic_k1,
                "bic_k2": bic_k2,
                "delta_bic_k2_minus_k1": bic_k2 - bic_k1,
                "selected_k2": bic_k2 < bic_k1,
                "k2_converged": k2_fit.converged,
                "k2_minimum_effective_class_fraction": float(
                    np.min(k2_fit.effective_class_fraction)
                ),
            }
        )
    return rows


def run_k1_null_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Test whether BIC invents K=2 endotypes when the true mechanism count is one."""

    tasks = [
        (
            repeat_index,
            seed,
            config,
        )
        for repeat_index, seed in enumerate(
            latent_null_seed_ledger(config.master_seed, config.null_repeats)
        )
    ]
    results = _map_tasks(_run_one_null_repeat, tasks, config.workers)
    rows = [row for result in results for row in result]
    return pd.DataFrame(rows)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    return _shared_wilson_interval(successes, trials)


def summarize_k1_null(raw_null: pd.DataFrame) -> pd.DataFrame:
    """Report false K=2 discovery under the homogeneous K=1 negative control."""

    rows: list[dict[str, object]] = []
    for method, group in raw_null.groupby("method", sort=False):
        successes = int(np.sum(group["selected_k2"]))
        trials = group.shape[0]
        ci_low, ci_high = _wilson_interval(successes, trials)
        rows.append(
            {
                "method": method,
                "repeat_count": trials,
                "false_k2_selections": successes,
                "false_k2_rate": successes / trials,
                "wilson_ci95_low": ci_low,
                "wilson_ci95_high": ci_high,
                "median_delta_bic_k2_minus_k1": float(
                    np.median(group["delta_bic_k2_minus_k1"])
                ),
                "k2_convergence_rate": float(np.mean(group["k2_converged"])),
            }
        )
    return pd.DataFrame(rows)


def build_example_patient(
    config: ExperimentConfig,
) -> dict[str, object]:
    """Create one reproducible strong-confounding example for interpretation."""

    renal_effect_sd = max(config.simulation.renal_effect_levels_sd)
    seed_sequence = np.random.SeedSequence(config.master_seed + 808_080)
    sequences = seed_sequence.spawn(4)
    training = simulate_two_mechanism_cohort(
        np.random.default_rng(sequences[0]),
        config.simulation.training_patients,
        renal_effect_sd,
        config.simulation,
    )
    test = simulate_two_mechanism_cohort(
        np.random.default_rng(sequences[1]),
        5_000,
        renal_effect_sd,
        config.simulation,
    )
    associative_fit = fit_associative_latent_class_model(
        training.biomarkers,
        training.renal_dysfunction,
        np.random.default_rng(sequences[2]),
        config.fitting,
    )
    causal_mask = np.zeros(len(BIOMARKER_NAMES), dtype=bool)
    causal_mask[Biomarker.NT_PROBNP_LIKE] = True
    causal_fit = fit_conditional_latent_model(
        training.biomarkers,
        training.renal_dysfunction,
        causal_mask,
        np.random.default_rng(sequences[3]),
        config.fitting,
    )
    associative = associative_posterior(
        associative_fit,
        test.biomarkers,
        test.renal_dysfunction,
    )
    causal = conditional_posterior(
        causal_fit,
        test.biomarkers,
        test.renal_dysfunction,
    )
    eligible = np.flatnonzero(
        (test.true_mechanism == Mechanism.COMPETING)
        & (test.renal_dysfunction == 1)
        & (np.argmax(associative, axis=1) == Mechanism.ATRIAL)
        & (np.argmax(causal, axis=1) == Mechanism.COMPETING)
    )
    selection_reason = "associative atrial / causal competing disagreement"
    if eligible.size == 0:
        eligible = np.flatnonzero(
            (test.true_mechanism == Mechanism.COMPETING)
            & (test.renal_dysfunction == 1)
        )
        selection_reason = "largest associative-minus-causal atrial probability gap"
        confidence_gap = (
            associative[eligible, Mechanism.ATRIAL]
            - causal[eligible, Mechanism.ATRIAL]
        )
    else:
        confidence_gap = (
            associative[eligible, Mechanism.ATRIAL]
            + causal[eligible, Mechanism.COMPETING]
            + 0.25 * test.biomarkers[eligible, Biomarker.NT_PROBNP_LIKE]
            - 0.10
            * np.abs(test.biomarkers[eligible, Biomarker.ATRIAL_ELECTRICAL])
            + 0.10 * test.biomarkers[eligible, Biomarker.COMPETING_SPECIFIC]
        )
    patient_index = int(eligible[np.argmax(confidence_gap)])
    renal_contribution = causal_fit.renal_effect * test.renal_dysfunction[patient_index]
    renal_neutralized = test.biomarkers[patient_index] - renal_contribution
    return {
        "patient_index_in_example_cohort": patient_index,
        "selection_reason": selection_reason,
        "true_mechanism": "competing",
        "renal_dysfunction": int(test.renal_dysfunction[patient_index]),
        "renal_effect_sd": renal_effect_sd,
        "observed_biomarkers": {
            name: float(value)
            for name, value in zip(
                BIOMARKER_NAMES,
                test.biomarkers[patient_index],
                strict=True,
            )
        },
        "causal_estimated_renal_contribution": {
            name: float(value)
            for name, value in zip(
                BIOMARKER_NAMES,
                renal_contribution,
                strict=True,
            )
        },
        "causal_renal_neutralized_biomarkers": {
            name: float(value)
            for name, value in zip(
                BIOMARKER_NAMES,
                renal_neutralized,
                strict=True,
            )
        },
        "associative_atrial_probability": float(
            associative[patient_index, Mechanism.ATRIAL]
        ),
        "causal_atrial_probability": float(
            causal[patient_index, Mechanism.ATRIAL]
        ),
    }


def validation_checks(
    raw_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    parameters: pd.DataFrame,
    raw_null: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Programmatic analysis checks saved with the experiment."""

    expected_metric_rows = (
        len(config.simulation.renal_effect_levels_sd)
        * config.repeats_per_level
        * len(ALL_METHODS)
    )
    expected_diagnostic_rows = (
        len(config.simulation.renal_effect_levels_sd)
        * config.repeats_per_level
        * len(FITTED_METHODS)
    )
    expected_null_rows = config.null_repeats * len(FITTED_METHODS)
    strongest = max(config.simulation.renal_effect_levels_sd)
    causal_parameters = parameters[parameters["renal_effect_sd"] == strongest]
    renal_effect_bias = float(
        np.mean(causal_parameters["causal_estimated_renal_effect_nt"]) - strongest
    )

    checks = {
        "metric_row_count_matches": raw_metrics.shape[0] == expected_metric_rows,
        "diagnostic_row_count_matches": diagnostics.shape[0] == expected_diagnostic_rows,
        "null_row_count_matches": raw_null.shape[0] == expected_null_rows,
        "all_probabilistic_metrics_finite": bool(
            np.isfinite(
                raw_metrics[
                    [
                        "accuracy",
                        "adjusted_rand_index",
                        "false_atrial_renal_competing",
                        "brier_score",
                        "expected_calibration_error",
                    ]
                ].to_numpy()
            ).all()
        ),
        "fitted_model_convergence_rate": float(np.mean(diagnostics["converged"])),
        "minimum_effective_class_fraction": float(
            diagnostics["minimum_effective_class_fraction"].min()
        ),
        "strong_level_causal_renal_effect_bias_sd": renal_effect_bias,
        "strong_level_causal_renal_effect_bias_within_0_10_sd": abs(renal_effect_bias)
        < 0.10,
        "null_k2_convergence_rate": float(np.mean(raw_null["k2_converged"])),
        "truth_not_accepted_by_fit_function_interfaces": True,
    }
    checks["all_required_checks_pass"] = bool(
        checks["metric_row_count_matches"]
        and checks["diagnostic_row_count_matches"]
        and checks["null_row_count_matches"]
        and checks["all_probabilistic_metrics_finite"]
        and checks["fitted_model_convergence_rate"] >= 0.99
        and checks["strong_level_causal_renal_effect_bias_within_0_10_sd"]
        and checks["null_k2_convergence_rate"] >= 0.95
    )
    return checks


def run_full_experiment(
    config: ExperimentConfig,
    output_directory: Path | str,
) -> dict[str, object]:
    """Run, summarize, plot, validate, and save the complete experiment."""

    started = perf_counter()
    config.validate()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    raw_metrics, diagnostics, parameters = run_recovery_experiment(config)
    summary = summarize_repeated_metrics(raw_metrics)
    contrasts = paired_method_contrasts(raw_metrics)
    raw_null = run_k1_null_experiment(config)
    null_summary = summarize_k1_null(raw_null)
    example = build_example_patient(config)
    checks = validation_checks(
        raw_metrics,
        diagnostics,
        parameters,
        raw_null,
        config,
    )

    raw_metrics.to_csv(output_directory / "raw_recovery_metrics.csv", index=False)
    summary.to_csv(output_directory / "recovery_summary.csv", index=False)
    contrasts.to_csv(output_directory / "paired_contrasts.csv", index=False)
    diagnostics.to_csv(output_directory / "fit_diagnostics.csv", index=False)
    parameters.to_csv(output_directory / "parameter_recovery.csv", index=False)
    raw_null.to_csv(output_directory / "k1_null_raw.csv", index=False)
    null_summary.to_csv(output_directory / "k1_null_summary.csv", index=False)
    (output_directory / "example_patient.json").write_text(
        json.dumps(example, indent=2),
        encoding="utf-8",
    )
    (output_directory / "validation_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "experiment_config": asdict(config),
        "model_parameter_counts_k2": {
            ASSOCIATIVE_LCA: 12,
            ASSOCIATIVE_ADJUSTED: 14,
            CAUSAL_SCM: 12,
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "truth_usage": (
            "True mechanisms are generated and stored by the simulator but are "
            "never passed to a fit function; they are used only by evaluate_posterior."
        ),
        "label_orientation": (
            "Latent labels are oriented using the prespecified electrical-minus-"
            "competing biomarker anchor; the misleading NT-proBNP-like marker and "
            "the simulated truth labels are excluded from orientation."
        ),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    manifest = write_manifest(
        output_directory,
        experiment="endotype_discovery",
        config=config,
        master_seed=config.master_seed,
        wall_clock_runtime_seconds=perf_counter() - started,
    )
    return {
        "raw_metrics": raw_metrics,
        "summary": summary,
        "contrasts": contrasts,
        "diagnostics": diagnostics,
        "parameters": parameters,
        "raw_null": raw_null,
        "null_summary": null_summary,
        "example": example,
        "validation_checks": checks,
        "metadata": metadata,
        "manifest": manifest,
    }


def _format_percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def print_key_results(results: dict[str, object], config: ExperimentConfig) -> None:
    """Print prespecified summaries from saved reductions rather than new calculations."""

    summary: pd.DataFrame = results["summary"]
    contrasts: pd.DataFrame = results["contrasts"]
    null_summary: pd.DataFrame = results["null_summary"]
    strongest = max(config.simulation.renal_effect_levels_sd)

    print("\nStrong-confounding results")
    for method in ALL_METHODS:
        accuracy = _metric_summary_lookup(summary, strongest, method, "accuracy")
        false_atrial = _metric_summary_lookup(
            summary,
            strongest,
            method,
            "false_atrial_renal_competing",
        )
        print(
            f"- {method}: accuracy {_format_percentage(accuracy['mean'])}; "
            f"false atrial {_format_percentage(false_atrial['mean'])}"
        )

    for comparator in (ASSOCIATIVE_LCA, ASSOCIATIVE_ADJUSTED):
        contrast = contrasts[
            (contrasts["renal_effect_sd"] == strongest)
            & (contrasts["comparator"] == comparator)
            & (contrasts["metric"] == "accuracy")
        ].iloc[0]
        print(
            f"- Causal minus {comparator} accuracy: "
            f"{100.0 * contrast['mean_difference']:.2f} percentage points "
            f"(95% MC CI {100.0 * contrast['ci95_low']:.2f} to "
            f"{100.0 * contrast['ci95_high']:.2f})"
        )

    print("\nK=1 null results")
    for row in null_summary.itertuples(index=False):
        print(
            f"- {row.method}: selected a spurious K=2 model in "
            f"{row.false_k2_selections}/{row.repeat_count} repeats "
            f"({_format_percentage(row.false_k2_rate)})"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse the legacy CLI so existing notebooks retain their invocation contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_latent_endotyping"),
        help="Directory for tables and metadata.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=500,
        help="Recovery repeats per renal-effect level.",
    )
    parser.add_argument(
        "--null-repeats",
        type=int,
        default=500,
        help="K=1 null repeats.",
    )
    parser.add_argument(
        "--training-patients",
        type=int,
        default=800,
        help="Unlabeled training patients per repeat.",
    )
    parser.add_argument(
        "--test-patients",
        type=int,
        default=1_000,
        help="Independent labeled-only-for-evaluation patients per repeat.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes; use 1 for the most portable run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a 10-repeat smoke test without changing scientific defaults.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the compatibility workflow and fail if required scientific checks do not pass."""

    arguments = parse_arguments()
    repeats = 10 if arguments.quick else arguments.repeats
    null_repeats = 10 if arguments.quick else arguments.null_repeats
    simulation = replace(
        SimulationConfig(),
        training_patients=arguments.training_patients,
        test_patients=arguments.test_patients,
    )
    config = ExperimentConfig(
        repeats_per_level=repeats,
        null_repeats=null_repeats,
        workers=arguments.workers,
        simulation=simulation,
    )
    results = run_full_experiment(config, arguments.output_dir)
    print_key_results(results, config)
    if not results["validation_checks"]["all_required_checks_pass"]:
        raise RuntimeError("One or more required validation checks failed.")


if __name__ == "__main__":
    main()
