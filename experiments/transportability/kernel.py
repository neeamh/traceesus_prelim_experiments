"""Cross-hospital transportability experiment for latent causal endotyping.

The simulator creates three unlabeled source hospitals and a held-out target
hospital.  Biological mechanism signatures are invariant.  Renal prevalence
and its NT-proBNP-like effect, background inflammation, assay calibration, and
biomarker missingness vary by hospital.

No model receives the simulated mechanism labels.  Target recalibration, when
used, receives only renal status, inflammation status, assay metadata, and
unlabeled biomarkers from a small calibration cohort.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
from time import perf_counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit, logsumexp

from traceesus.core.io import write_manifest
from traceesus.core.runner import ordered_map, transport_seed_ledger
from traceesus.queries.posterior import anchor_order as _shared_anchor_order
from traceesus.core.stats import (
    monte_carlo_summary,
    paired_mean_contrast,
    paired_nanmean_contrast,
)

from traceesus.experiments.endotype_discovery.kernel import (
    BIOMARKER_NAMES,
    Biomarker,
    FittingConfig,
    Mechanism,
    evaluate_posterior,
)
from configs.transportability import (
    ABLATION_TARGETS,
    HospitalSpec,
    SOURCE_HOSPITALS,
    TARGET_HOSPITALS,
    TransportExperimentConfig,
    TransportSimulationConfig,
)


POOLED_ASSOCIATIVE = "Pooled associative latent class model"
TARGET_ADJUSTED_ASSOCIATIVE = "Target-calibrated associative latent model"
FROZEN_CAUSAL = "Frozen causal latent SCM"
MODULAR_CAUSAL = "Modular causal latent SCM"
ORACLE = "Target oracle (reference)"

FITTED_METHODS = (
    POOLED_ASSOCIATIVE,
    TARGET_ADJUSTED_ASSOCIATIVE,
    FROZEN_CAUSAL,
    MODULAR_CAUSAL,
)
MAIN_METHODS = (
    POOLED_ASSOCIATIVE,
    TARGET_ADJUSTED_ASSOCIATIVE,
    MODULAR_CAUSAL,
)
ALL_METHODS = (*FITTED_METHODS, ORACLE)

@dataclass(frozen=True)
class HospitalCohort:
    """Raw observed measurements plus simulator-only truth fields."""

    raw_biomarkers: np.ndarray
    biomarker_observed: np.ndarray
    renal_dysfunction: np.ndarray
    background_inflammation: np.ndarray
    true_mechanism: np.ndarray
    complete_calibrated_biomarkers: np.ndarray
    hospital: HospitalSpec


@dataclass(frozen=True)
class MissingGaussianMixtureFit:
    """Retain an oriented unlabeled mixture fit under biomarker missingness.

    The anchor margin records whether the prespecified biological contrast can
    orient latent labels without consulting simulator truth.
    """

    class_probability: np.ndarray
    class_means: np.ndarray
    biomarker_variance: np.ndarray
    log_likelihood: float
    converged: bool
    iterations: int
    best_start: int
    effective_class_fraction: np.ndarray
    anchor_margin: float


def simulate_hospital(
    rng: np.random.Generator,
    patient_count: int,
    hospital: HospitalSpec,
    config: TransportSimulationConfig,
) -> HospitalCohort:
    """Generate one hospital while holding mechanism biology invariant."""

    renal = rng.binomial(1, hospital.renal_prevalence, patient_count).astype(np.int8)
    inflammation = rng.binomial(
        1,
        hospital.inflammation_prevalence,
        patient_count,
    ).astype(np.int8)
    is_atrial = rng.random(patient_count) < config.atrial_probability
    mechanism = np.where(
        is_atrial,
        Mechanism.ATRIAL,
        Mechanism.COMPETING,
    ).astype(np.int8)

    class_effects = np.asarray(
        (config.atrial_path_effects_sd, config.competing_path_effects_sd),
        dtype=float,
    )
    calibrated = class_effects[mechanism].copy()
    calibrated[:, Biomarker.NT_PROBNP_LIKE] += (
        hospital.renal_effect_nt_sd * renal
    )
    calibrated[:, Biomarker.COMPETING_SPECIFIC] += (
        hospital.inflammation_effect_competing_sd * inflammation
    )
    calibrated += rng.normal(
        0.0,
        np.asarray(config.biomarker_noise_sd),
        size=calibrated.shape,
    )

    assay_offset = np.asarray(hospital.assay_offset, dtype=float)
    assay_scale = np.asarray(hospital.assay_scale, dtype=float)
    raw_biomarkers = assay_offset + assay_scale * calibrated

    base_probability = np.asarray(hospital.missingness_base, dtype=float)
    missing_log_odds = np.broadcast_to(
        logit(base_probability),
        raw_biomarkers.shape,
    ).copy()
    missing_log_odds[:, Biomarker.NT_PROBNP_LIKE] += (
        config.renal_missingness_log_odds_nt * renal
    )
    missing_log_odds[:, Biomarker.COMPETING_SPECIFIC] += (
        config.inflammation_missingness_log_odds_competing * inflammation
    )
    missing = rng.random(raw_biomarkers.shape) < expit(missing_log_odds)
    observed = ~missing
    raw_with_missing = raw_biomarkers.copy()
    raw_with_missing[missing] = np.nan

    return HospitalCohort(
        raw_biomarkers=raw_with_missing,
        biomarker_observed=observed,
        renal_dysfunction=renal,
        background_inflammation=inflammation,
        true_mechanism=mechanism,
        complete_calibrated_biomarkers=calibrated,
        hospital=hospital,
    )


def assay_calibrate(cohort: HospitalCohort) -> np.ndarray:
    """Invert known laboratory calibration while preserving missing values."""

    offset = np.asarray(cohort.hospital.assay_offset, dtype=float)
    scale = np.asarray(cohort.hospital.assay_scale, dtype=float)
    return (cohort.raw_biomarkers - offset) / scale


def fit_nuisance_paths(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    inflammation: np.ndarray,
    renal_path_mask: np.ndarray,
    inflammation_path_mask: np.ndarray,
) -> np.ndarray:
    """Estimate label-free nuisance slopes biomarker by biomarker.

    The estimator is valid here because renal dysfunction and background
    inflammation are generated independently of the true mechanism.
    """

    renal_path_mask = np.asarray(renal_path_mask, dtype=bool)
    inflammation_path_mask = np.asarray(inflammation_path_mask, dtype=bool)
    slopes = np.zeros((2, biomarkers.shape[1]), dtype=float)
    for biomarker_index in range(biomarkers.shape[1]):
        observed = np.isfinite(biomarkers[:, biomarker_index])
        columns = [np.ones(int(np.sum(observed)), dtype=float)]
        column_map: list[int] = []
        if renal_path_mask[biomarker_index]:
            columns.append(renal[observed].astype(float))
            column_map.append(0)
        if inflammation_path_mask[biomarker_index]:
            columns.append(inflammation[observed].astype(float))
            column_map.append(1)
        design = np.column_stack(columns)
        response = biomarkers[observed, biomarker_index]
        coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
        for coefficient_index, slope_row in enumerate(column_map, start=1):
            slopes[slope_row, biomarker_index] = coefficients[coefficient_index]
    return slopes


def remove_nuisance_paths(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    inflammation: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    """Residualize prespecified site-varying paths while preserving missing values.

    This modular step isolates the stable mechanism signal; changing its matrix
    operation would change both the causal estimand and floating-point order.
    """

    covariates = np.column_stack((renal, inflammation))
    return biomarkers - covariates @ slopes


def _initial_responsibility(
    biomarkers: np.ndarray,
    rng: np.random.Generator,
    start_index: int,
    fitting: FittingConfig,
) -> np.ndarray:
    column_mean = np.nanmean(biomarkers, axis=0)
    column_sd = np.nanstd(biomarkers, axis=0)
    column_sd = np.where(column_sd > 1e-8, column_sd, 1.0)
    standardized = (biomarkers - column_mean) / column_sd
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    if start_index == 0:
        projection = (
            standardized[:, Biomarker.ATRIAL_ELECTRICAL]
            - standardized[:, Biomarker.COMPETING_SPECIFIC]
        )
    elif start_index == 1:
        projection = standardized[:, Biomarker.NT_PROBNP_LIKE]
    else:
        direction = rng.normal(size=standardized.shape[1])
        direction /= np.linalg.norm(direction)
        projection = standardized @ direction
    projection_sd = max(float(np.std(projection)), 1e-8)
    probability = expit(1.5 * (projection - np.median(projection)) / projection_sd)
    probability = np.clip(
        probability,
        fitting.probability_floor,
        1.0 - fitting.probability_floor,
    )
    return np.column_stack((probability, 1.0 - probability))


def _missing_gmm_m_step(
    biomarkers: np.ndarray,
    responsibility: np.ndarray,
    fitting: FittingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_count, biomarker_count = biomarkers.shape
    effective_count = np.sum(responsibility, axis=0)
    if np.any(
        effective_count < fitting.minimum_effective_class_fraction * patient_count
    ):
        raise FloatingPointError("A latent class collapsed.")
    smoothing = fitting.beta_prior_pseudocount
    class_probability = (effective_count + smoothing) / (
        patient_count + 2.0 * smoothing
    )
    observed = np.isfinite(biomarkers)
    class_means = np.zeros((2, biomarker_count), dtype=float)
    for component in range(2):
        for biomarker_index in range(biomarker_count):
            available = observed[:, biomarker_index]
            weights = responsibility[available, component]
            denominator = float(np.sum(weights))
            if denominator <= 1e-8:
                raise FloatingPointError("No effective observations for a class marker.")
            class_means[component, biomarker_index] = float(
                np.sum(weights * biomarkers[available, biomarker_index]) / denominator
            )

    variance = np.zeros(biomarker_count, dtype=float)
    for biomarker_index in range(biomarker_count):
        available = observed[:, biomarker_index]
        residual = (
            biomarkers[available, biomarker_index, None]
            - class_means[None, :, biomarker_index]
        )
        numerator = np.sum(responsibility[available] * residual**2)
        denominator = np.sum(responsibility[available])
        variance[biomarker_index] = numerator / denominator
    variance = np.maximum(variance, fitting.variance_floor)
    return class_probability, class_means, variance


def _missing_gmm_e_step(
    biomarkers: np.ndarray,
    class_probability: np.ndarray,
    class_means: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, float]:
    observed = np.isfinite(biomarkers)
    filled = np.where(observed, biomarkers, 0.0)
    log_joint = np.broadcast_to(
        np.log(class_probability)[None, :],
        (biomarkers.shape[0], 2),
    ).copy()
    for component in range(2):
        residual = filled - class_means[component]
        contribution = -0.5 * (
            residual**2 / variance + np.log(2.0 * np.pi * variance)
        )
        log_joint[:, component] += np.sum(
            np.where(observed, contribution, 0.0),
            axis=1,
        )
    normalizer = logsumexp(log_joint, axis=1)
    responsibility = np.exp(log_joint - normalizer[:, None])
    return responsibility, float(np.sum(normalizer))


def _anchor_fit(
    class_probability: np.ndarray,
    class_means: np.ndarray,
    variance: np.ndarray,
    responsibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    order, margin = _shared_anchor_order(
        class_means,
        variance,
        atrial_electrical_index=Biomarker.ATRIAL_ELECTRICAL,
        competing_specific_index=Biomarker.COMPETING_SPECIFIC,
    )
    return (
        class_probability[order],
        class_means[order],
        responsibility[:, order],
        margin,
    )


def fit_missing_gaussian_mixture(
    biomarkers: np.ndarray,
    rng: np.random.Generator,
    fitting: FittingConfig,
) -> MissingGaussianMixtureFit:
    """Fit a two-component diagonal Gaussian mixture with missing-data likelihood."""

    best: dict[str, object] | None = None
    for start_index in range(fitting.random_starts):
        responsibility = _initial_responsibility(
            biomarkers,
            rng,
            start_index,
            fitting,
        )
        previous_log_likelihood = -np.inf
        converged = False
        try:
            for iteration in range(1, fitting.maximum_em_iterations + 1):
                parameters = _missing_gmm_m_step(
                    biomarkers,
                    responsibility,
                    fitting,
                )
                responsibility, log_likelihood = _missing_gmm_e_step(
                    biomarkers,
                    *parameters,
                )
                improvement = log_likelihood - previous_log_likelihood
                tolerance = fitting.relative_log_likelihood_tolerance * (
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
        raise RuntimeError("All missing-data mixture starts failed.")

    class_probability, class_means, variance = best["parameters"]
    class_probability, class_means, responsibility, margin = _anchor_fit(
        class_probability,
        class_means,
        variance,
        best["responsibility"],
    )
    return MissingGaussianMixtureFit(
        class_probability=class_probability,
        class_means=class_means,
        biomarker_variance=variance,
        log_likelihood=float(best["log_likelihood"]),
        converged=bool(best["converged"]),
        iterations=int(best["iterations"]),
        best_start=int(best["best_start"]),
        effective_class_fraction=np.mean(responsibility, axis=0),
        anchor_margin=margin,
    )


def missing_gmm_posterior(
    fit: MissingGaussianMixtureFit,
    biomarkers: np.ndarray,
) -> np.ndarray:
    """Evaluate the oriented mixture with missingness integrated exactly as in EM."""

    responsibility, _ = _missing_gmm_e_step(
        biomarkers,
        fit.class_probability,
        fit.class_means,
        fit.biomarker_variance,
    )
    return responsibility


def target_oracle_posterior(
    cohort: HospitalCohort,
    config: TransportSimulationConfig,
) -> np.ndarray:
    """Provide a data-generating ceiling, not a deployable transport method.

    The oracle uses true target nuisance paths but never target mechanism labels;
    it exists only to separate transport error from irreducible Bayes error.
    """

    biomarkers = assay_calibrate(cohort)
    true_slopes = np.zeros((2, len(BIOMARKER_NAMES)), dtype=float)
    true_slopes[0, Biomarker.NT_PROBNP_LIKE] = cohort.hospital.renal_effect_nt_sd
    true_slopes[
        1,
        Biomarker.COMPETING_SPECIFIC,
    ] = cohort.hospital.inflammation_effect_competing_sd
    residualized = remove_nuisance_paths(
        biomarkers,
        cohort.renal_dysfunction,
        cohort.background_inflammation,
        true_slopes,
    )
    class_means = np.asarray(
        (config.atrial_path_effects_sd, config.competing_path_effects_sd),
        dtype=float,
    )
    variance = np.asarray(config.biomarker_noise_sd, dtype=float) ** 2
    class_probability = np.asarray(
        (config.atrial_probability, 1.0 - config.atrial_probability)
    )
    posterior, _ = _missing_gmm_e_step(
        residualized,
        class_probability,
        class_means,
        variance,
    )
    return posterior


def _extended_metrics(
    posterior: np.ndarray,
    cohort: HospitalCohort,
    fitting: FittingConfig,
) -> dict[str, float]:
    metrics = evaluate_posterior(
        posterior,
        cohort.true_mechanism,
        cohort.renal_dysfunction,
        fitting.calibration_bins,
    )
    prediction = np.argmax(posterior, axis=1)
    calibrated = assay_calibrate(cohort)
    any_missing = np.any(~np.isfinite(calibrated), axis=1)
    complete = ~any_missing
    electrical_missing = ~np.isfinite(
        calibrated[:, Biomarker.ATRIAL_ELECTRICAL]
    )
    metrics.update(
        {
            "accuracy_complete_case": float(
                np.mean(prediction[complete] == cohort.true_mechanism[complete])
            )
            if np.any(complete)
            else np.nan,
            "accuracy_any_missing": float(
                np.mean(
                    prediction[any_missing] == cohort.true_mechanism[any_missing]
                )
            )
            if np.any(any_missing)
            else np.nan,
            "accuracy_electrical_missing": float(
                np.mean(
                    prediction[electrical_missing]
                    == cohort.true_mechanism[electrical_missing]
                )
            )
            if np.any(electrical_missing)
            else np.nan,
            "biomarker_missing_fraction": float(
                np.mean(~cohort.biomarker_observed)
            ),
            "all_biomarkers_missing_rate": float(
                np.mean(np.all(~cohort.biomarker_observed, axis=1))
            ),
        }
    )
    return metrics


def _fit_with_retry(
    biomarkers: np.ndarray,
    rng_primary: np.random.Generator,
    rng_retry: np.random.Generator,
    fitting: FittingConfig,
) -> MissingGaussianMixtureFit:
    fit = fit_missing_gaussian_mixture(biomarkers, rng_primary, fitting)
    if fit.converged:
        return fit
    retry = replace(
        fitting,
        random_starts=max(8, fitting.random_starts),
        maximum_em_iterations=max(1_200, fitting.maximum_em_iterations),
    )
    return fit_missing_gaussian_mixture(biomarkers, rng_retry, retry)


def _run_one_repeat(
    task: tuple[int, int, TransportExperimentConfig],
) -> dict[str, object]:
    repeat_index, seed, config = task
    seed_sequence = np.random.SeedSequence(seed)
    sequences = seed_sequence.spawn(12)

    source_cohorts = [
        simulate_hospital(
            np.random.default_rng(sequences[hospital_index]),
            config.simulation.source_patients_per_hospital,
            hospital,
            config.simulation,
        )
        for hospital_index, hospital in enumerate(config.source_hospitals)
    ]
    calibrated_sources = [assay_calibrate(cohort) for cohort in source_cohorts]
    all_source_biomarkers = np.vstack(calibrated_sources)

    causal_renal_mask = np.zeros(len(BIOMARKER_NAMES), dtype=bool)
    causal_renal_mask[Biomarker.NT_PROBNP_LIKE] = True
    causal_inflammation_mask = np.zeros(len(BIOMARKER_NAMES), dtype=bool)
    causal_inflammation_mask[Biomarker.COMPETING_SPECIFIC] = True
    adjusted_mask = np.ones(len(BIOMARKER_NAMES), dtype=bool)

    source_causal_slopes = []
    source_adjusted_slopes = []
    causal_residuals = []
    adjusted_residuals = []
    maximum_assay_reconstruction_error = 0.0
    for cohort, biomarkers in zip(source_cohorts, calibrated_sources, strict=True):
        observed = cohort.biomarker_observed
        maximum_assay_reconstruction_error = max(
            maximum_assay_reconstruction_error,
            float(
                np.max(
                    np.abs(
                        biomarkers[observed]
                        - cohort.complete_calibrated_biomarkers[observed]
                    )
                )
            ),
        )
        causal_slopes = fit_nuisance_paths(
            biomarkers,
            cohort.renal_dysfunction,
            cohort.background_inflammation,
            causal_renal_mask,
            causal_inflammation_mask,
        )
        adjusted_slopes = fit_nuisance_paths(
            biomarkers,
            cohort.renal_dysfunction,
            cohort.background_inflammation,
            adjusted_mask,
            adjusted_mask,
        )
        source_causal_slopes.append(causal_slopes)
        source_adjusted_slopes.append(adjusted_slopes)
        causal_residuals.append(
            remove_nuisance_paths(
                biomarkers,
                cohort.renal_dysfunction,
                cohort.background_inflammation,
                causal_slopes,
            )
        )
        adjusted_residuals.append(
            remove_nuisance_paths(
                biomarkers,
                cohort.renal_dysfunction,
                cohort.background_inflammation,
                adjusted_slopes,
            )
        )

    pooled_fit = _fit_with_retry(
        all_source_biomarkers,
        np.random.default_rng(sequences[3]),
        np.random.default_rng(sequences[4]),
        config.fitting,
    )
    adjusted_fit = _fit_with_retry(
        np.vstack(adjusted_residuals),
        np.random.default_rng(sequences[5]),
        np.random.default_rng(sequences[6]),
        config.fitting,
    )
    causal_fit = _fit_with_retry(
        np.vstack(causal_residuals),
        np.random.default_rng(sequences[7]),
        np.random.default_rng(sequences[8]),
        config.fitting,
    )
    frozen_causal_slopes = np.mean(np.stack(source_causal_slopes), axis=0)

    metric_rows: list[dict[str, object]] = []
    target_diagnostic_rows: list[dict[str, object]] = []
    target_seed_calibration = int(sequences[9].generate_state(1, dtype=np.uint64)[0])
    target_seed_test = int(sequences[10].generate_state(1, dtype=np.uint64)[0])

    for shift_index, target_spec in enumerate(config.target_hospitals):
        calibration = simulate_hospital(
            np.random.default_rng(target_seed_calibration),
            config.simulation.target_calibration_patients,
            target_spec,
            config.simulation,
        )
        test = simulate_hospital(
            np.random.default_rng(target_seed_test),
            config.simulation.target_test_patients,
            target_spec,
            config.simulation,
        )
        calibration_biomarkers = assay_calibrate(calibration)
        test_biomarkers = assay_calibrate(test)
        target_causal_slopes = fit_nuisance_paths(
            calibration_biomarkers,
            calibration.renal_dysfunction,
            calibration.background_inflammation,
            causal_renal_mask,
            causal_inflammation_mask,
        )
        target_adjusted_slopes = fit_nuisance_paths(
            calibration_biomarkers,
            calibration.renal_dysfunction,
            calibration.background_inflammation,
            adjusted_mask,
            adjusted_mask,
        )

        adjusted_test = remove_nuisance_paths(
            test_biomarkers,
            test.renal_dysfunction,
            test.background_inflammation,
            target_adjusted_slopes,
        )
        frozen_causal_test = remove_nuisance_paths(
            test_biomarkers,
            test.renal_dysfunction,
            test.background_inflammation,
            frozen_causal_slopes,
        )
        modular_causal_test = remove_nuisance_paths(
            test_biomarkers,
            test.renal_dysfunction,
            test.background_inflammation,
            target_causal_slopes,
        )

        posterior_by_method = {
            POOLED_ASSOCIATIVE: missing_gmm_posterior(
                pooled_fit,
                test_biomarkers,
            ),
            TARGET_ADJUSTED_ASSOCIATIVE: missing_gmm_posterior(
                adjusted_fit,
                adjusted_test,
            ),
            FROZEN_CAUSAL: missing_gmm_posterior(
                causal_fit,
                frozen_causal_test,
            ),
            MODULAR_CAUSAL: missing_gmm_posterior(
                causal_fit,
                modular_causal_test,
            ),
            ORACLE: target_oracle_posterior(test, config.simulation),
        }
        for method, posterior in posterior_by_method.items():
            row: dict[str, object] = {
                "repeat": repeat_index,
                "shift_index": shift_index,
                "shift": target_spec.name,
                "method": method,
            }
            row.update(_extended_metrics(posterior, test, config.fitting))
            metric_rows.append(row)

        true_slopes = np.zeros((2, len(BIOMARKER_NAMES)), dtype=float)
        true_slopes[0, Biomarker.NT_PROBNP_LIKE] = target_spec.renal_effect_nt_sd
        true_slopes[
            1,
            Biomarker.COMPETING_SPECIFIC,
        ] = target_spec.inflammation_effect_competing_sd
        target_diagnostic_rows.append(
            {
                "repeat": repeat_index,
                "shift_index": shift_index,
                "shift": target_spec.name,
                "causal_renal_path_estimate": target_causal_slopes[
                    0,
                    Biomarker.NT_PROBNP_LIKE,
                ],
                "true_renal_path": target_spec.renal_effect_nt_sd,
                "causal_inflammation_path_estimate": target_causal_slopes[
                    1,
                    Biomarker.COMPETING_SPECIFIC,
                ],
                "true_inflammation_path": (
                    target_spec.inflammation_effect_competing_sd
                ),
                "adjusted_irrelevant_slope_rms": float(
                    np.sqrt(
                        np.mean(
                            target_adjusted_slopes[
                                ~np.asarray(
                                    (
                                        causal_renal_mask,
                                        causal_inflammation_mask,
                                    )
                                )
                            ]
                            ** 2
                        )
                    )
                ),
                "target_missing_fraction": float(
                    np.mean(~test.biomarker_observed)
                ),
                "target_all_missing_rate": float(
                    np.mean(np.all(~test.biomarker_observed, axis=1))
                ),
            }
        )

    fit_rows = []
    for method, fit in (
        (POOLED_ASSOCIATIVE, pooled_fit),
        (TARGET_ADJUSTED_ASSOCIATIVE, adjusted_fit),
        (MODULAR_CAUSAL, causal_fit),
    ):
        fit_rows.append(
            {
                "repeat": repeat_index,
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
        )
    return {
        "metrics": metric_rows,
        "fit_diagnostics": fit_rows,
        "target_diagnostics": target_diagnostic_rows,
        "maximum_assay_reconstruction_error": maximum_assay_reconstruction_error,
    }


def _map_tasks(
    worker,
    tasks: list[tuple],
    workers: int,
) -> Iterable[dict[str, object]]:
    return ordered_map(worker, tasks, workers)


def run_transport_experiment(
    config: TransportExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Execute repeats through the fixed ordered seed ledger and result collection.

    Ordered collection is mandatory because reordering rows or reductions could
    change exact artifacts even when mathematical estimates are equivalent.
    """

    config.validate()
    seeds = transport_seed_ledger(config.master_seed, config.repeats)
    tasks = [
        (
            repeat_index,
            seed,
            config,
        )
        for repeat_index, seed in enumerate(seeds)
    ]
    results = _map_tasks(_run_one_repeat, tasks, config.workers)
    metric_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    maximum_assay_error = 0.0
    for result in results:
        metric_rows.extend(result["metrics"])
        fit_rows.extend(result["fit_diagnostics"])
        target_rows.extend(result["target_diagnostics"])
        maximum_assay_error = max(
            maximum_assay_error,
            float(result["maximum_assay_reconstruction_error"]),
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(fit_rows),
        pd.DataFrame(target_rows),
        maximum_assay_error,
    )


def summarize_metrics(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Reduce repeat metrics in first-seen group order for byte-stable outputs."""

    identifiers = {
        "repeat",
        "shift_index",
        "shift",
        "method",
        "renal_competing_subgroup_size",
    }
    metric_columns = [
        column for column in raw_metrics.columns if column not in identifiers
    ]
    rows: list[dict[str, object]] = []
    for (shift_index, shift, method), group in raw_metrics.groupby(
        ["shift_index", "shift", "method"],
        sort=False,
    ):
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            statistics = monte_carlo_summary(values)
            rows.append(
                {
                    "shift_index": shift_index,
                    "shift": shift,
                    "method": method,
                    "metric": metric,
                    **statistics,
                }
            )
    return pd.DataFrame(rows)


def paired_contrasts(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare methods within the same repeat so shared cohort noise cancels."""

    metrics = (
        "accuracy",
        "adjusted_rand_index",
        "false_atrial_renal_competing",
        "brier_score",
        "expected_calibration_error",
        "accuracy_any_missing",
    )
    comparators = (
        POOLED_ASSOCIATIVE,
        TARGET_ADJUSTED_ASSOCIATIVE,
        FROZEN_CAUSAL,
    )
    rows: list[dict[str, object]] = []
    for shift_index, shift in (
        raw_metrics[["shift_index", "shift"]]
        .drop_duplicates()
        .sort_values("shift_index")
        .itertuples(index=False, name=None)
    ):
        level = raw_metrics[raw_metrics["shift_index"] == shift_index]
        for comparator in comparators:
            wide = level.pivot(index="repeat", columns="method")
            for metric in metrics:
                difference = (
                    wide[metric][MODULAR_CAUSAL]
                    - wide[metric][comparator]
                ).to_numpy(dtype=float)
                statistics = paired_nanmean_contrast(difference)
                rows.append(
                    {
                        "shift_index": shift_index,
                        "shift": shift,
                        "comparator": comparator,
                        "metric": metric,
                        "difference_definition": "modular causal minus comparator",
                        **statistics,
                    }
                )
    return pd.DataFrame(rows)


def transport_degradation(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired strong-minus-no-shift accuracy within repeat and method."""

    accuracy = raw_metrics.pivot_table(
        index=["repeat", "method"],
        columns="shift_index",
        values="accuracy",
    )
    strongest_index = int(raw_metrics["shift_index"].max())
    degradation = accuracy[strongest_index] - accuracy[0]
    rows = []
    for method, values in degradation.groupby(level="method"):
        array = values.to_numpy(dtype=float)
        statistics = paired_mean_contrast(array)
        rows.append(
            {
                "method": method,
                "difference_definition": "strong-shift minus no-shift accuracy",
                **statistics,
            }
        )
    return pd.DataFrame(rows)


def ablation_accuracy_changes(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize each target ablation minus its paired no-shift accuracy."""

    accuracy = raw_metrics.pivot_table(
        index=["repeat", "method"],
        columns=["shift_index", "shift"],
        values="accuracy",
    )
    baseline = accuracy.iloc[:, 0]
    rows: list[dict[str, object]] = []
    for shift_index, shift in accuracy.columns[1:]:
        difference = accuracy[(shift_index, shift)] - baseline
        for method, values in difference.groupby(level="method"):
            array = values.to_numpy(dtype=float)
            statistics = paired_mean_contrast(array)
            rows.append(
                {
                    "shift_index": shift_index,
                    "shift": shift,
                    "method": method,
                    "difference_definition": "ablation minus no-shift accuracy",
                    **statistics,
                }
            )
    return pd.DataFrame(rows)


def negative_control_check(
    contrasts: pd.DataFrame,
    config: TransportExperimentConfig,
) -> dict[str, object]:
    """Test no-shift equivalence before interpreting gains under hospital shift."""

    row = contrasts[
        (contrasts["shift_index"] == 0)
        & (contrasts["comparator"] == TARGET_ADJUSTED_ASSOCIATIVE)
        & (contrasts["metric"] == "accuracy")
    ].iloc[0]
    margin = config.equivalence_margin_accuracy
    return {
        "comparison": "modular causal minus target-calibrated associative",
        "margin": margin,
        "mean_difference": float(row["mean_difference"]),
        "ci95_low": float(row["ci95_low"]),
        "ci95_high": float(row["ci95_high"]),
        "ci_entirely_within_equivalence_margin": bool(
            row["ci95_low"] > -margin and row["ci95_high"] < margin
        ),
    }


def _summary_series(
    summary: pd.DataFrame,
    method: str,
    metric: str,
) -> pd.DataFrame:
    return summary[
        (summary["method"] == method)
        & (summary["metric"] == metric)
    ].sort_values("shift_index")


def plot_transport_figure(
    summary: pd.DataFrame,
    config: TransportExperimentConfig,
    output_directory: Path,
) -> None:
    """Render primary recovery and renal-subgroup error panels from saved summaries."""

    shifts = np.arange(len(config.target_hospitals))
    labels = tuple(hospital.name.replace(" shift", "") for hospital in config.target_hospitals)
    styles = {
        POOLED_ASSOCIATIVE: ("#D97706", (0, (5, 3)), "s", "white"),
        TARGET_ADJUSTED_ASSOCIATIVE: ("#6B7280", (0, (2, 2)), "D", "white"),
        MODULAR_CAUSAL: ("#1D4ED8", "solid", "o", "#1D4ED8"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharex=True)
    panels = (
        (
            axes[0],
            "accuracy",
            "True-mechanism recovery",
            "Patients correctly assigned (%)",
        ),
        (
            axes[1],
            "false_atrial_renal_competing",
            "False atrial classification",
            "Renal/competing patients called atrial (%)",
        ),
    )
    for axis, metric, title, ylabel in panels:
        for method in MAIN_METHODS:
            series = _summary_series(summary, method, metric)
            color, linestyle, marker, facecolor = styles[method]
            mean = 100.0 * series["mean"].to_numpy()
            low = np.clip(100.0 * series["ci95_low"].to_numpy(), 0.0, 100.0)
            high = np.clip(100.0 * series["ci95_high"].to_numpy(), 0.0, 100.0)
            axis.plot(
                shifts,
                mean,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor=facecolor,
                markeredgecolor=color,
                linewidth=2.2,
                markersize=6.5,
                label=method,
            )
            axis.fill_between(
                shifts,
                low,
                high,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(title, loc="left", fontsize=11.5, fontweight="semibold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(shifts, labels)
        axis.set_xlabel("Held-out hospital shift severity")
        axis.set_ylim(0.0, 100.0)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle(
        "Figure T1. Cross-hospital transport of hidden-mechanism recovery",
        x=0.065,
        y=1.03,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.065,
        0.965,
        (
            f"Three unlabeled source hospitals; target calibration n="
            f"{config.simulation.target_calibration_patients:,}; independent target "
            f"test n={config.simulation.target_test_patients:,}; "
            f"{config.repeats} paired repeats"
        ),
        ha="left",
        fontsize=9.5,
        color="#4B5563",
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=3,
        frameon=False,
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0.035, 0.09, 0.995, 0.92))
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_directory / f"figure_T1_transportability.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_transport_controls(
    summary: pd.DataFrame,
    degradation: pd.DataFrame,
    config: TransportExperimentConfig,
    output_directory: Path,
) -> None:
    """Render oracle/frozen controls needed to localize transport degradation."""

    shifts = np.arange(len(config.target_hospitals))
    labels = tuple(hospital.name.replace(" shift", "") for hospital in config.target_hospitals)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    control_styles = {
        TARGET_ADJUSTED_ASSOCIATIVE: ("#6B7280", (0, (5, 3)), "D", "white"),
        FROZEN_CAUSAL: ("#93C5FD", (0, (2, 2)), "^", "white"),
        MODULAR_CAUSAL: ("#1D4ED8", "solid", "o", "#1D4ED8"),
        ORACLE: ("#111827", (0, (1.5, 2.5)), "s", "white"),
    }
    for method, (color, linestyle, marker, facecolor) in control_styles.items():
        series = _summary_series(summary, method, "accuracy")
        axes[0].plot(
            shifts,
            100.0 * series["mean"].to_numpy(),
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            linewidth=2.0,
            label=method,
        )
    axes[0].set_title(
        "Target calibration and oracle reference",
        loc="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    axes[0].set_ylabel("Patients correctly assigned (%)")
    axes[0].set_xlabel("Held-out hospital shift severity")
    axes[0].set_xticks(shifts, labels)
    axes[0].set_ylim(0.0, 100.0)
    axes[0].grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=8.0, loc="lower left")

    order = (
        POOLED_ASSOCIATIVE,
        TARGET_ADJUSTED_ASSOCIATIVE,
        FROZEN_CAUSAL,
        MODULAR_CAUSAL,
    )
    indexed = degradation.set_index("method")
    values = [100.0 * indexed.loc[method, "mean_difference"] for method in order]
    lower = [
        100.0
        * (
            indexed.loc[method, "mean_difference"]
            - indexed.loc[method, "ci95_low"]
        )
        for method in order
    ]
    upper = [
        100.0
        * (
            indexed.loc[method, "ci95_high"]
            - indexed.loc[method, "mean_difference"]
        )
        for method in order
    ]
    colors = ("#D97706", "#9CA3AF", "#93C5FD", "#1D4ED8")
    short_labels = ("Pooled\nassoc.", "Adjusted\nassoc.", "Frozen\ncausal", "Modular\ncausal")
    bars = axes[1].bar(
        np.arange(len(order)),
        values,
        color=colors,
        edgecolor="#374151",
        linewidth=0.8,
        yerr=np.asarray((lower, upper)),
        capsize=4,
    )
    axes[1].bar_label(
        bars,
        labels=[f"{value:+.1f}" for value in values],
        padding=3,
        fontsize=9,
    )
    axes[1].axhline(0.0, color="#374151", linewidth=1.0)
    axes[1].set_title(
        "Accuracy change: strong shift minus no shift",
        loc="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    axes[1].set_ylabel("Percentage-point change")
    axes[1].set_xticks(np.arange(len(order)), short_labels)
    axes[1].grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.7)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        "Figure T2. Transport controls and degradation",
        x=0.065,
        y=1.02,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.065,
        0.95,
        "Target calibration uses no mechanism labels; negative values indicate lost accuracy",
        ha="left",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.035, 0.04, 0.995, 0.91))
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_directory / f"figure_T2_transport_controls.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_ablation_figure(
    changes: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Show one-shift-at-a-time accuracy changes against the no-shift baseline."""

    methods = MAIN_METHODS
    method_styles = {
        POOLED_ASSOCIATIVE: ("#D97706", "s", "white"),
        TARGET_ADJUSTED_ASSOCIATIVE: ("#6B7280", "D", "white"),
        MODULAR_CAUSAL: ("#1D4ED8", "o", "#1D4ED8"),
    }
    shifts = (
        changes[["shift_index", "shift"]]
        .drop_duplicates()
        .sort_values("shift_index")
    )
    positions = np.arange(shifts.shape[0], dtype=float)
    offsets = (-0.20, 0.0, 0.20)
    fig, axis = plt.subplots(figsize=(10.4, 5.4))
    for method, offset in zip(methods, offsets, strict=True):
        method_rows = (
            changes[changes["method"] == method]
            .merge(shifts, on=["shift_index", "shift"], how="right")
            .sort_values("shift_index")
        )
        color, marker, facecolor = method_styles[method]
        mean = 100.0 * method_rows["mean_difference"].to_numpy()
        lower = 100.0 * (
            method_rows["mean_difference"] - method_rows["ci95_low"]
        ).to_numpy()
        upper = 100.0 * (
            method_rows["ci95_high"] - method_rows["mean_difference"]
        ).to_numpy()
        axis.errorbar(
            positions + offset,
            mean,
            yerr=np.asarray((lower, upper)),
            fmt=marker,
            color=color,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markersize=7,
            linewidth=1.6,
            capsize=3,
            label=method,
        )
    axis.axhline(0.0, color="#374151", linewidth=1.0)
    axis.set_xticks(positions, shifts["shift"].tolist())
    axis.set_ylabel("Accuracy change from no shift (percentage points)")
    fig.suptitle(
        "Figure T3. One-factor target-hospital shift ablations",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.92,
        "Each nuisance component is shifted to its strong value; 500 paired repeats",
        fontsize=9.5,
        color="#4B5563",
    )
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        frameon=False,
        fontsize=8.7,
        loc="lower left",
        ncol=1,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.87))
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_directory / f"figure_T3_shift_ablations.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def validation_checks(
    raw_metrics: pd.DataFrame,
    fit_diagnostics: pd.DataFrame,
    target_diagnostics: pd.DataFrame,
    maximum_assay_error: float,
    negative_control: dict[str, object],
    config: TransportExperimentConfig,
) -> dict[str, object]:
    """Make scientific and structural run invariants explicit in machine-readable form.

    In particular, this asserts that fit interfaces cannot accept simulator
    truth and that known assay calibration reconstructs the intended scale.
    """

    expected_metric_rows = (
        config.repeats * len(config.target_hospitals) * len(ALL_METHODS)
    )
    expected_fit_rows = config.repeats * 3
    expected_target_rows = config.repeats * len(config.target_hospitals)
    strong_target = target_diagnostics[
        target_diagnostics["shift_index"] == target_diagnostics["shift_index"].max()
    ]
    renal_bias = float(
        np.mean(
            strong_target["causal_renal_path_estimate"]
            - strong_target["true_renal_path"]
        )
    )
    inflammation_bias = float(
        np.mean(
            strong_target["causal_inflammation_path_estimate"]
            - strong_target["true_inflammation_path"]
        )
    )
    path_bias_tolerance = 0.30 if config.repeats < 50 else 0.15
    fit_interfaces_exclude_truth = all(
        "true_mechanism" not in inspect.signature(function).parameters
        for function in (
            fit_missing_gaussian_mixture,
            fit_nuisance_paths,
        )
    )
    checks = {
        "metric_row_count_matches": raw_metrics.shape[0] == expected_metric_rows,
        "fit_row_count_matches": fit_diagnostics.shape[0] == expected_fit_rows,
        "target_diagnostic_row_count_matches": (
            target_diagnostics.shape[0] == expected_target_rows
        ),
        "all_source_latent_fits_converged": bool(
            fit_diagnostics["converged"].all()
        ),
        "minimum_effective_class_fraction": float(
            fit_diagnostics["minimum_effective_class_fraction"].min()
        ),
        "all_metrics_finite": bool(
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
        "maximum_assay_reconstruction_error": maximum_assay_error,
        "assay_calibration_exact_to_numerical_tolerance": maximum_assay_error
        < 1e-12,
        "strong_target_renal_path_bias": renal_bias,
        "strong_target_inflammation_path_bias": inflammation_bias,
        "target_path_bias_tolerance_sd": path_bias_tolerance,
        "strong_target_path_biases_within_tolerance": (
            abs(renal_bias) < path_bias_tolerance
            and abs(inflammation_bias) < path_bias_tolerance
        ),
        "no_shift_negative_control_passes": bool(
            negative_control["ci_entirely_within_equivalence_margin"]
        ),
        "truth_not_accepted_by_fit_interfaces": fit_interfaces_exclude_truth,
    }
    checks["all_required_checks_pass"] = bool(
        checks["metric_row_count_matches"]
        and checks["fit_row_count_matches"]
        and checks["target_diagnostic_row_count_matches"]
        and checks["all_source_latent_fits_converged"]
        and checks["all_metrics_finite"]
        and checks["assay_calibration_exact_to_numerical_tolerance"]
        and checks["strong_target_path_biases_within_tolerance"]
        and checks["no_shift_negative_control_passes"]
        and checks["truth_not_accepted_by_fit_interfaces"]
    )
    return checks


def run_full_experiment(
    config: TransportExperimentConfig,
    output_directory: Path | str,
) -> dict[str, object]:
    """Preserve the standalone main-transport workflow for legacy notebooks.

    New callers should use the OOP facade; this function retains historical
    imports, filenames, and arithmetic for reproducibility.
    """

    started = perf_counter()
    config.validate()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_metrics, fit_diagnostics, target_diagnostics, maximum_assay_error = (
        run_transport_experiment(config)
    )
    summary = summarize_metrics(raw_metrics)
    contrasts = paired_contrasts(raw_metrics)
    degradation = transport_degradation(raw_metrics)
    negative_control = negative_control_check(contrasts, config)
    checks = validation_checks(
        raw_metrics,
        fit_diagnostics,
        target_diagnostics,
        maximum_assay_error,
        negative_control,
        config,
    )

    raw_metrics.to_csv(output_directory / "raw_transport_metrics.csv", index=False)
    summary.to_csv(output_directory / "transport_summary.csv", index=False)
    contrasts.to_csv(output_directory / "paired_transport_contrasts.csv", index=False)
    degradation.to_csv(output_directory / "transport_degradation.csv", index=False)
    fit_diagnostics.to_csv(output_directory / "fit_diagnostics.csv", index=False)
    target_diagnostics.to_csv(
        output_directory / "target_calibration_diagnostics.csv",
        index=False,
    )
    (output_directory / "negative_control.json").write_text(
        json.dumps(negative_control, indent=2),
        encoding="utf-8",
    )
    (output_directory / "validation_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "experiment_config": asdict(config),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "truth_usage": (
            "True mechanism labels are stored by the simulator and used only "
            "inside evaluate_posterior after all fitting and prediction."
        ),
        "target_calibration": (
            "Assay metadata plus renal status, background inflammation, and "
            "unlabeled biomarkers from the target calibration cohort; no "
            "mechanism labels are used."
        ),
        "assay_boundary": (
            "Assay offset and scale metadata are assumed known and applied "
            "equally to every method. Unknown calibration is not identified by "
            "this experiment."
        ),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    plot_transport_figure(summary, config, output_directory)
    plot_transport_controls(summary, degradation, config, output_directory)
    manifest = write_manifest(
        output_directory,
        experiment="transportability",
        config=config,
        master_seed=config.master_seed,
        wall_clock_runtime_seconds=perf_counter() - started,
    )
    return {
        "raw_metrics": raw_metrics,
        "summary": summary,
        "contrasts": contrasts,
        "degradation": degradation,
        "fit_diagnostics": fit_diagnostics,
        "target_diagnostics": target_diagnostics,
        "negative_control": negative_control,
        "validation_checks": checks,
        "metadata": metadata,
        "manifest": manifest,
    }


def run_ablation_experiment(
    config: TransportExperimentConfig,
    output_directory: Path | str,
) -> dict[str, object]:
    """Run strong one-factor target shifts using the same hidden-label design."""

    ablation_config = replace(config, target_hospitals=ABLATION_TARGETS)
    ablation_config.validate()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_metrics, fit_diagnostics, target_diagnostics, maximum_assay_error = (
        run_transport_experiment(ablation_config)
    )
    summary = summarize_metrics(raw_metrics)
    contrasts = paired_contrasts(raw_metrics)
    changes = ablation_accuracy_changes(raw_metrics)
    negative_control = negative_control_check(contrasts, ablation_config)
    checks = validation_checks(
        raw_metrics,
        fit_diagnostics,
        target_diagnostics,
        maximum_assay_error,
        negative_control,
        ablation_config,
    )
    raw_metrics.to_csv(output_directory / "raw_ablation_metrics.csv", index=False)
    summary.to_csv(output_directory / "ablation_summary.csv", index=False)
    contrasts.to_csv(output_directory / "paired_ablation_contrasts.csv", index=False)
    changes.to_csv(output_directory / "ablation_accuracy_changes.csv", index=False)
    fit_diagnostics.to_csv(output_directory / "fit_diagnostics.csv", index=False)
    target_diagnostics.to_csv(
        output_directory / "target_calibration_diagnostics.csv",
        index=False,
    )
    (output_directory / "negative_control.json").write_text(
        json.dumps(negative_control, indent=2),
        encoding="utf-8",
    )
    (output_directory / "validation_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )
    plot_ablation_figure(changes, output_directory)
    return {
        "raw_metrics": raw_metrics,
        "summary": summary,
        "contrasts": contrasts,
        "changes": changes,
        "fit_diagnostics": fit_diagnostics,
        "target_diagnostics": target_diagnostics,
        "negative_control": negative_control,
        "validation_checks": checks,
    }


def run_exact_no_shift_negative_control(
    config: TransportExperimentConfig,
    output_directory: Path | str,
) -> dict[str, object]:
    """Run an exact source-target identity control.

    Unlike the "No shift" point in the main multi-source curve, which matches
    the reference source hospital, every source and target hospital in this
    dedicated control has identical generating parameters. This is the literal
    negative control for the proposal's identical-distribution claim.
    """

    baseline = replace(TARGET_HOSPITALS[0], name="Exact no shift")
    identical_sources = tuple(
        replace(baseline, name=f"Identical source {label}")
        for label in ("A", "B", "C")
    )
    exact_config = replace(
        config,
        source_hospitals=identical_sources,
        target_hospitals=(baseline,),
    )
    exact_config.validate()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_metrics, fit_diagnostics, target_diagnostics, maximum_assay_error = (
        run_transport_experiment(exact_config)
    )
    summary = summarize_metrics(raw_metrics)
    contrasts = paired_contrasts(raw_metrics)
    negative_control = negative_control_check(contrasts, exact_config)
    checks = validation_checks(
        raw_metrics,
        fit_diagnostics,
        target_diagnostics,
        maximum_assay_error,
        negative_control,
        exact_config,
    )
    raw_metrics.to_csv(output_directory / "raw_metrics.csv", index=False)
    summary.to_csv(output_directory / "summary.csv", index=False)
    contrasts.to_csv(output_directory / "paired_contrasts.csv", index=False)
    fit_diagnostics.to_csv(output_directory / "fit_diagnostics.csv", index=False)
    target_diagnostics.to_csv(
        output_directory / "target_calibration_diagnostics.csv",
        index=False,
    )
    (output_directory / "negative_control.json").write_text(
        json.dumps(negative_control, indent=2),
        encoding="utf-8",
    )
    (output_directory / "validation_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "experiment_config": asdict(exact_config),
        "purpose": (
            "Literal no-shift negative control: all source and target hospitals "
            "share identical generating parameters."
        ),
        "truth_usage": (
            "True mechanism labels are used only after prediction for "
            "simulation evaluation."
        ),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "raw_metrics": raw_metrics,
        "summary": summary,
        "contrasts": contrasts,
        "fit_diagnostics": fit_diagnostics,
        "target_diagnostics": target_diagnostics,
        "negative_control": negative_control,
        "validation_checks": checks,
        "metadata": metadata,
    }


def _lookup(
    summary: pd.DataFrame,
    shift_index: int,
    method: str,
    metric: str,
) -> pd.Series:
    match = summary[
        (summary["shift_index"] == shift_index)
        & (summary["method"] == method)
        & (summary["metric"] == metric)
    ]
    if match.shape[0] != 1:
        raise RuntimeError("Expected exactly one summary row.")
    return match.iloc[0]


def print_key_results(
    results: dict[str, object],
    config: TransportExperimentConfig,
) -> None:
    """Print the prespecified no-shift, strong-shift, and control comparisons."""

    summary: pd.DataFrame = results["summary"]
    contrasts: pd.DataFrame = results["contrasts"]
    strongest = len(config.target_hospitals) - 1
    print("\nNo-shift accuracy")
    for method in MAIN_METHODS:
        row = _lookup(summary, 0, method, "accuracy")
        print(f"- {method}: {100.0 * row['mean']:.2f}%")
    print("\nStrong-shift accuracy and false atrial classification")
    for method in (*MAIN_METHODS, FROZEN_CAUSAL, ORACLE):
        accuracy = _lookup(summary, strongest, method, "accuracy")
        false_atrial = _lookup(
            summary,
            strongest,
            method,
            "false_atrial_renal_competing",
        )
        print(
            f"- {method}: accuracy {100.0 * accuracy['mean']:.2f}%; "
            f"false atrial {100.0 * false_atrial['mean']:.2f}%"
        )
    print("\nStrong-shift modular causal contrasts")
    for comparator in (POOLED_ASSOCIATIVE, TARGET_ADJUSTED_ASSOCIATIVE, FROZEN_CAUSAL):
        row = contrasts[
            (contrasts["shift_index"] == strongest)
            & (contrasts["comparator"] == comparator)
            & (contrasts["metric"] == "accuracy")
        ].iloc[0]
        print(
            f"- versus {comparator}: {100.0 * row['mean_difference']:.2f} pp "
            f"(95% MC CI {100.0 * row['ci95_low']:.2f} to "
            f"{100.0 * row['ci95_high']:.2f})"
        )
    print("\nNegative control")
    print(json.dumps(results["negative_control"], indent=2))


def parse_arguments() -> argparse.Namespace:
    """Parse the historical transport CLI without changing its defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_transportability"),
    )
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 10 repeats without changing the scientific defaults.",
    )
    parser.add_argument(
        "--with-ablation",
        action="store_true",
        help="Also run the one-factor strong-shift ablations in a subdirectory.",
    )
    parser.add_argument(
        "--with-exact-negative-control",
        action="store_true",
        help=(
            "Also run the literal identical source-target distribution "
            "negative control in a subdirectory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run requested compatibility analyses and enforce every required check."""

    started = perf_counter()
    arguments = parse_arguments()
    config = TransportExperimentConfig(
        repeats=10 if arguments.quick else arguments.repeats,
        workers=arguments.workers,
    )
    results = run_full_experiment(config, arguments.output_dir)
    print_key_results(results, config)
    if not results["validation_checks"]["all_required_checks_pass"]:
        raise RuntimeError("One or more required validation checks failed.")
    if arguments.with_ablation:
        ablation_results = run_ablation_experiment(
            config,
            arguments.output_dir / "ablations",
        )
        if not ablation_results["validation_checks"]["all_required_checks_pass"]:
            raise RuntimeError("One or more ablation validation checks failed.")
    if arguments.with_exact_negative_control:
        exact_results = run_exact_no_shift_negative_control(
            config,
            arguments.output_dir / "exact_no_shift_control",
        )
        if not exact_results["validation_checks"]["all_required_checks_pass"]:
            raise RuntimeError("The exact no-shift negative control failed validation.")
    write_manifest(
        arguments.output_dir,
        experiment="transportability",
        config=config,
        master_seed=config.master_seed,
        wall_clock_runtime_seconds=perf_counter() - started,
    )


if __name__ == "__main__":
    main()
