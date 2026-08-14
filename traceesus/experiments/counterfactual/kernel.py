"""Reproducible preliminary R21 simulation.

The experiment deliberately separates three questions:

1. Kidney-blind posterior matching:
   Which latent mechanism does the biomarker vector resemble if the renal
   biomarker path is omitted?
2. Kidney-aware counterfactual scoring:
   After posterior-integrated abduction, which mechanism is both sufficient
   for and necessary to the observed biomarker pattern?
3. Same-SCM posterior diagnostic:
   What is the Bayes-optimal top-1 ranking under the same kidney-aware SCM?

The third method is not a competing headline result. It is a required fairness
diagnostic: counterfactual querying cannot be claimed to outperform a correctly
specified posterior for expected top-1 latent-class accuracy when both use the
same evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
from time import perf_counter
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logsumexp

from traceesus.core.stats import (
    bic,
    bounded_rate_monte_carlo_summary,
    paired_rate_contrast,
    wilson_interval as shared_wilson_interval,
)
from traceesus.core.runner import nested_seed_sequence_ledger, seed_sequence_ledger
from traceesus.core.io import write_manifest
from traceesus.simulators.two_mechanism import TwoMechanismSimulator
from configs.counterfactual import ExperimentConfig


ATRIAL = 0
COMPETING = 1
METHOD_POSTERIOR_BLIND = "Posterior matching (kidney-blind)"
METHOD_COUNTERFACTUAL = "Counterfactual scoring (kidney-aware)"
METHOD_POSTERIOR_FULL = "Posterior (same kidney-aware SCM)"
PRIMARY_METHODS = (METHOD_POSTERIOR_BLIND, METHOD_COUNTERFACTUAL)


def atrial_prior_given_renal(renal: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """Return P(Z=atrial | renal status) from the prespecified structural prior."""

    log_odds = (
        config.atrial_log_odds_when_renal_normal
        + config.renal_to_atrial_log_odds * renal
    )
    return expit(log_odds)


def simulate_two_mechanism_study(
    config: ExperimentConfig,
    renal_effect_sd: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Adapt the shared simulator to the historical preliminary-study mapping."""

    generated = TwoMechanismSimulator(config, renal_effect_sd).simulate(
        rng, config.patients_per_repeat
    )
    renal = generated.observed.covariate("renal_dysfunction")

    return {
        "mechanism": generated.truth.mechanism,
        "renal": renal,
        "biomarkers": generated.observed.biomarkers,
        "atrial_prior": atrial_prior_given_renal(renal, config),
    }


def _posterior_from_model(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: ExperimentConfig,
    *,
    include_renal_path: bool,
) -> np.ndarray:
    """Compute exact two-class posterior probabilities under a specified model."""

    effects = np.asarray(config.mechanism_effects, dtype=float)
    noise_sd = np.asarray(config.biomarker_noise_sd, dtype=float)
    n = biomarkers.shape[0]

    if include_renal_path:
        atrial_prior = atrial_prior_given_renal(renal, config)
        renal_contribution = np.zeros((n, 3), dtype=float)
        renal_contribution[:, 0] = renal_effect_sd * renal
    else:
        renal_zero = np.zeros(1, dtype=int)
        prior_if_normal = atrial_prior_given_renal(renal_zero, config)[0]
        renal_one = np.ones(1, dtype=int)
        prior_if_impaired = atrial_prior_given_renal(renal_one, config)[0]
        marginal_atrial_prior = (
            (1.0 - config.renal_prevalence) * prior_if_normal
            + config.renal_prevalence * prior_if_impaired
        )
        atrial_prior = np.full(n, marginal_atrial_prior, dtype=float)
        renal_contribution = np.zeros((n, 3), dtype=float)

    class_priors = np.column_stack((atrial_prior, 1.0 - atrial_prior))
    log_joint = np.empty((n, 2), dtype=float)
    for mechanism in (ATRIAL, COMPETING):
        candidate_mean = effects[mechanism] + renal_contribution
        standardized_residual = (biomarkers - candidate_mean) / noise_sd
        log_likelihood = -0.5 * np.sum(standardized_residual**2, axis=1)
        log_joint[:, mechanism] = (
            log_likelihood + np.log(class_priors[:, mechanism])
        )

    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


def kidney_blind_posterior(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: ExperimentConfig,
) -> np.ndarray:
    """Biomarker-resemblance posterior that deliberately omits the renal path."""

    return _posterior_from_model(
        biomarkers,
        renal,
        renal_effect_sd,
        config,
        include_renal_path=False,
    )


def kidney_aware_posterior(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: ExperimentConfig,
) -> np.ndarray:
    """Correct posterior under the same renal-aware SCM used by the causal query."""

    return _posterior_from_model(
        biomarkers,
        renal,
        renal_effect_sd,
        config,
        include_renal_path=True,
    )


def posterior_integrated_counterfactual_scores(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: ExperimentConfig,
) -> dict[str, np.ndarray]:
    """Compute sufficiency and disablement by abduction-action-prediction.

    For every patient, we enumerate both latent-mechanism branches. Within each
    branch we abduct that branch's exogenous biomarker residual, reuse the same
    residual under intervention, and average the resulting counterfactual
    quantity over the kidney-aware posterior. This avoids the invalid shortcut
    of abducting once "as if" each candidate were already true.

    In this deliberately symmetric K=2 toy model, normalized sufficiency and
    disablement are monotone transformations of the correctly specified
    posterior. That is a feature, not a bug: it establishes that any gain over
    the kidney-blind resemblance baseline comes from modeling the renal path,
    not from claiming a causal query beats the Bayes classifier by magic.
    """

    posterior = kidney_aware_posterior(
        biomarkers, renal, renal_effect_sd, config
    )
    effects = np.asarray(config.mechanism_effects, dtype=float)
    noise_sd = np.asarray(config.biomarker_noise_sd, dtype=float)
    n = biomarkers.shape[0]

    renal_contribution = np.zeros((n, 3), dtype=float)
    renal_contribution[:, 0] = renal_effect_sd * renal

    # U[i, z, :] is the branch-specific exogenous residual abducted under Z=z.
    exogenous_residual = np.empty((n, 2, 3), dtype=float)
    for branch in (ATRIAL, COMPETING):
        branch_mean = effects[branch] + renal_contribution
        exogenous_residual[:, branch, :] = biomarkers - branch_mean

    normalized_disablement = np.empty((n, 2), dtype=float)
    normalized_sufficiency = np.empty((n, 2), dtype=float)

    for candidate in (ATRIAL, COMPETING):
        disablement_by_branch = np.empty((n, 2), dtype=float)
        sufficiency_fit_by_branch = np.empty((n, 2), dtype=float)

        for branch in (ATRIAL, COMPETING):
            factual_gate = effects[branch]
            disabled_gate = (
                np.zeros(3, dtype=float)
                if branch == candidate
                else effects[branch]
            )
            biomarkers_if_disabled = (
                renal_contribution
                + disabled_gate
                + exogenous_residual[:, branch, :]
            )
            disabled_distance = np.sum(
                ((biomarkers - biomarkers_if_disabled) / noise_sd) ** 2,
                axis=1,
            )
            disablement_by_branch[:, branch] = disabled_distance

            biomarkers_if_candidate_only = (
                renal_contribution
                + effects[candidate]
                + exogenous_residual[:, branch, :]
            )
            sufficient_distance = np.sum(
                ((biomarkers - biomarkers_if_candidate_only) / noise_sd) ** 2,
                axis=1,
            )
            sufficiency_fit_by_branch[:, branch] = np.exp(
                -0.5 * sufficient_distance
            )

        expected_disablement = np.sum(
            posterior * disablement_by_branch, axis=1
        )
        maximum_disablement = np.sum((effects[candidate] / noise_sd) ** 2)
        normalized_disablement[:, candidate] = (
            expected_disablement / maximum_disablement
        )

        expected_sufficiency = np.sum(
            posterior * sufficiency_fit_by_branch, axis=1
        )
        other = COMPETING if candidate == ATRIAL else ATRIAL
        mismatch_distance = np.sum(
            ((effects[candidate] - effects[other]) / noise_sd) ** 2
        )
        mismatch_fit = np.exp(-0.5 * mismatch_distance)
        normalized_sufficiency[:, candidate] = (
            (expected_sufficiency - mismatch_fit) / (1.0 - mismatch_fit)
        )

    combined = (
        config.counterfactual_disablement_weight * normalized_disablement
        + config.counterfactual_sufficiency_weight * normalized_sufficiency
    )
    return {
        "combined": combined,
        "disablement": normalized_disablement,
        "sufficiency": normalized_sufficiency,
        "posterior": posterior,
    }


def predict_with_random_ties(
    scores: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose the highest-scoring mechanism without favoring class 0 on ties."""

    maxima = np.max(scores, axis=1, keepdims=True)
    tied = np.isclose(scores, maxima, rtol=0.0, atol=1e-12)
    prediction = np.argmax(scores, axis=1)
    tie_rows = np.flatnonzero(np.sum(tied, axis=1) > 1)
    for row in tie_rows:
        prediction[row] = rng.choice(np.flatnonzero(tied[row]))
    return prediction


def _metric_rows(
    *,
    repeat: int,
    strength_index: int,
    renal_effect_sd: float,
    method: str,
    truth: np.ndarray,
    renal: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, float | int | str]]:
    accuracy = float(np.mean(prediction == truth))
    confounded_competing = (renal == 1) & (truth == COMPETING)
    subgroup_n = int(np.sum(confounded_competing))
    if subgroup_n == 0:
        raise RuntimeError("Confounded competing-mechanism subgroup is empty.")
    false_atrial = float(
        np.mean(prediction[confounded_competing] == ATRIAL)
    )
    return [
        {
            "repeat": repeat,
            "strength_index": strength_index,
            "renal_effect_sd": renal_effect_sd,
            "method": method,
            "metric": "true_mechanism_accuracy",
            "value": accuracy,
            "denominator": int(truth.size),
        },
        {
            "repeat": repeat,
            "strength_index": strength_index,
            "renal_effect_sd": renal_effect_sd,
            "method": method,
            "metric": "false_atrial_confounded_competing",
            "value": false_atrial,
            "denominator": subgroup_n,
        },
    ]


def run_main_simulation(config: ExperimentConfig) -> pd.DataFrame:
    """Run all paired main-experiment repeats."""

    config.validate()
    rows: list[dict[str, float | int | str]] = []
    seed_ledger = nested_seed_sequence_ledger(
        config.seed,
        len(config.confounding_strengths_sd),
        config.repeats_per_level,
    )

    for strength_index, (renal_effect_sd, repeat_seeds) in enumerate(
        zip(config.confounding_strengths_sd, seed_ledger, strict=True)
    ):
        for repeat, repeat_seed in enumerate(repeat_seeds):
            data_seed, tie_seed = repeat_seed.spawn(2)
            data_rng = np.random.default_rng(data_seed)
            tie_rngs = [
                np.random.default_rng(seed)
                for seed in tie_seed.spawn(3)
            ]
            data = simulate_two_mechanism_study(
                config, renal_effect_sd, data_rng
            )
            truth = data["mechanism"]
            renal = data["renal"]
            biomarkers = data["biomarkers"]

            blind_scores = kidney_blind_posterior(
                biomarkers, renal, renal_effect_sd, config
            )
            counterfactual = posterior_integrated_counterfactual_scores(
                biomarkers, renal, renal_effect_sd, config
            )
            full_scores = counterfactual["posterior"]

            score_sets = (
                (METHOD_POSTERIOR_BLIND, blind_scores),
                (METHOD_COUNTERFACTUAL, counterfactual["combined"]),
                (METHOD_POSTERIOR_FULL, full_scores),
            )
            for (method, scores), tie_rng in zip(
                score_sets, tie_rngs, strict=True
            ):
                prediction = predict_with_random_ties(scores, tie_rng)
                rows.extend(
                    _metric_rows(
                        repeat=repeat,
                        strength_index=strength_index,
                        renal_effect_sd=renal_effect_sd,
                        method=method,
                        truth=truth,
                        renal=renal,
                        prediction=prediction,
                    )
                )

    result = pd.DataFrame(rows)
    expected_rows = (
        len(config.confounding_strengths_sd)
        * config.repeats_per_level
        * 3
        * 2
    )
    if len(result) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} metric rows, got {len(result)}.")
    return result


def summarize_repeated_simulation(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize expected performance with 95% Monte Carlo confidence intervals."""

    group_columns = [
        "strength_index",
        "renal_effect_sd",
        "method",
        "metric",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in raw_metrics.groupby(group_columns, sort=True):
        values = group["value"].to_numpy(dtype=float)
        statistics = bounded_rate_monte_carlo_summary(values)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                **statistics,
                "mean_denominator": float(group["denominator"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def summarize_paired_differences(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute paired method contrasts using the same simulated study in each pair."""

    pivot = raw_metrics.pivot(
        index=[
            "repeat",
            "strength_index",
            "renal_effect_sd",
            "metric",
        ],
        columns="method",
        values="value",
    ).reset_index()
    rows: list[dict[str, Any]] = []
    for keys, group in pivot.groupby(
        ["strength_index", "renal_effect_sd", "metric"], sort=True
    ):
        metric = keys[2]
        if metric == "true_mechanism_accuracy":
            difference = (
                group[METHOD_COUNTERFACTUAL]
                - group[METHOD_POSTERIOR_BLIND]
            )
            contrast = "counterfactual_minus_kidney_blind"
        else:
            difference = (
                group[METHOD_POSTERIOR_BLIND]
                - group[METHOD_COUNTERFACTUAL]
            )
            contrast = "kidney_blind_minus_counterfactual"
        statistics = paired_rate_contrast(difference)
        rows.append(
            {
                "strength_index": keys[0],
                "renal_effect_sd": keys[1],
                "metric": metric,
                "contrast": contrast,
                **statistics,
            }
        )
    return pd.DataFrame(rows)


def _log_diag_gaussian(
    x: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Return log densities for every row and diagonal-Gaussian component."""

    centered = x[:, None, :] - means[None, :, :]
    return -0.5 * (
        np.sum(np.log(2.0 * np.pi * variances), axis=1)[None, :]
        + np.sum(centered**2 / variances[None, :, :], axis=2)
    )


def _fit_one_diagonal_gaussian(
    x: np.ndarray,
    variance_floor: float,
) -> dict[str, Any]:
    means = np.mean(x, axis=0, keepdims=True)
    variances = np.maximum(np.var(x, axis=0, keepdims=True), variance_floor)
    log_likelihood = float(np.sum(_log_diag_gaussian(x, means, variances)))
    return {
        "log_likelihood": log_likelihood,
        "weights": np.array([1.0]),
        "means": means,
        "variances": variances,
        "converged": True,
        "iterations": 1,
    }


def _fit_two_diagonal_gaussians(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    starts: int,
    max_iter: int,
    variance_floor: float,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Fit a two-component diagonal Gaussian mixture with multiple EM starts."""

    n, dimension = x.shape
    global_variance = np.maximum(np.var(x, axis=0), variance_floor)
    best: dict[str, Any] | None = None

    for start in range(starts):
        if start == 0:
            direction = np.zeros(dimension)
            direction[np.argmax(global_variance)] = 1.0
        else:
            direction = rng.normal(size=dimension)
            direction /= np.linalg.norm(direction)
        projection = x @ direction
        lower, upper = np.quantile(projection, [0.30, 0.70])
        lower_group = x[projection <= lower]
        upper_group = x[projection >= upper]
        means = np.vstack((lower_group.mean(axis=0), upper_group.mean(axis=0)))
        variances = np.tile(global_variance, (2, 1))
        weights = np.array([0.5, 0.5])
        previous_log_likelihood = -np.inf
        converged = False

        for iteration in range(1, max_iter + 1):
            log_joint = (
                np.log(np.clip(weights, 1e-12, None))[None, :]
                + _log_diag_gaussian(x, means, variances)
            )
            row_log_likelihood = logsumexp(log_joint, axis=1)
            log_likelihood = float(np.sum(row_log_likelihood))
            responsibilities = np.exp(
                log_joint - row_log_likelihood[:, None]
            )

            effective_n = np.maximum(responsibilities.sum(axis=0), 1e-8)
            weights = effective_n / n
            means = (responsibilities.T @ x) / effective_n[:, None]
            centered = x[:, None, :] - means[None, :, :]
            variances = np.einsum(
                "nk,nkd->kd", responsibilities, centered**2
            ) / effective_n[:, None]
            variances = np.maximum(variances, variance_floor)

            if np.isfinite(previous_log_likelihood):
                improvement = log_likelihood - previous_log_likelihood
                if abs(improvement) <= tolerance * (
                    1.0 + abs(previous_log_likelihood)
                ):
                    converged = True
                    break
            previous_log_likelihood = log_likelihood

        final_log_joint = (
            np.log(np.clip(weights, 1e-12, None))[None, :]
            + _log_diag_gaussian(x, means, variances)
        )
        final_log_likelihood = float(
            np.sum(logsumexp(final_log_joint, axis=1))
        )
        candidate = {
            "log_likelihood": final_log_likelihood,
            "weights": weights.copy(),
            "means": means.copy(),
            "variances": variances.copy(),
            "converged": converged,
            "iterations": iteration,
        }
        if best is None or candidate["log_likelihood"] > best["log_likelihood"]:
            best = candidate

    if best is None:
        raise RuntimeError("No two-component GMM fit was produced.")
    return best


def run_k1_null_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Test whether BIC selects a spurious K=2 model when truth is homogeneous.

    The null data retain the strongest renal biomarker distortion. The primary
    pipeline subtracts that known renal path before comparing K=1 versus K=2
    diagonal Gaussian models. This tests model selection; fixing K=1 in advance
    would not test whether the procedure invents endotypes.
    """

    config.validate()
    repeat_seeds = seed_sequence_ledger(
        config.seed + 1_000_003,
        config.null_repeats,
    )
    rows: list[dict[str, Any]] = []
    n = config.null_patients_per_repeat
    dimension = 3
    parameter_count_k1 = 2 * dimension
    parameter_count_k2 = (2 - 1) + 2 * dimension * 2
    simulator = TwoMechanismSimulator(config, config.null_renal_effect_sd)

    for repeat, repeat_seed in enumerate(repeat_seeds):
        data_seed, fit_seed = repeat_seed.spawn(2)
        data_rng = np.random.default_rng(data_seed)
        fit_rng = np.random.default_rng(fit_seed)
        generated = simulator.simulate_null(data_rng, n)
        renal = generated.observed.covariate("renal_dysfunction")
        renal_contribution = np.zeros((n, dimension), dtype=float)
        renal_contribution[:, 0] = config.null_renal_effect_sd * renal
        biomarkers = generated.observed.biomarkers

        # Correct the observed direct renal path before latent-class selection.
        residualized = biomarkers - renal_contribution
        fit_k1 = _fit_one_diagonal_gaussian(
            residualized, config.gmm_variance_floor
        )
        fit_k2 = _fit_two_diagonal_gaussians(
            residualized,
            fit_rng,
            starts=config.null_gmm_starts,
            max_iter=config.null_gmm_max_iter,
            variance_floor=config.gmm_variance_floor,
        )
        bic_k1 = bic(fit_k1["log_likelihood"], parameter_count_k1, n)
        bic_k2 = bic(fit_k2["log_likelihood"], parameter_count_k2, n)
        minimum_weight = float(np.min(fit_k2["weights"]))
        select_k2 = bool(
            bic_k2 < bic_k1
            and minimum_weight >= config.null_min_component_weight
            and fit_k2["converged"]
        )
        rows.append(
            {
                "repeat": repeat,
                "truth_k": 1,
                "selected_k": 2 if select_k2 else 1,
                "select_k2": select_k2,
                "bic_k1": bic_k1,
                "bic_k2": bic_k2,
                "bic_k2_minus_k1": bic_k2 - bic_k1,
                "minimum_k2_weight": minimum_weight,
                "k2_converged": bool(fit_k2["converged"]),
                "k2_iterations": int(fit_k2["iterations"]),
            }
        )
    return pd.DataFrame(rows)


def wilson_interval(
    successes: int,
    trials: int,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if trials <= 0:
        raise ValueError("trials must be positive.")
    if z_value == 1.959963984540054:
        return shared_wilson_interval(successes, trials)
    proportion = successes / trials
    denominator = 1.0 + z_value**2 / trials
    center = (
        proportion + z_value**2 / (2.0 * trials)
    ) / denominator
    half_width = (
        z_value
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_value**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def summarize_k1_null(null_results: pd.DataFrame) -> pd.DataFrame:
    """Quantify spurious K=2 selection as the prespecified identifiability control.

    The Wilson interval and median BIC contrast are retained verbatim because
    they bound model-selection behavior when the simulated truth has only K=1.
    """

    false_selections = int(null_results["select_k2"].sum())
    repeats = int(len(null_results))
    ci_low, ci_high = wilson_interval(false_selections, repeats)
    return pd.DataFrame(
        [
            {
                "truth_k": 1,
                "comparison": "BIC-selected K=1 versus K=2 after renal adjustment",
                "repeats": repeats,
                "false_k2_selections": false_selections,
                "false_k2_rate": false_selections / repeats,
                "wilson_ci_low": ci_low,
                "wilson_ci_high": ci_high,
                "k2_convergence_rate": float(
                    null_results["k2_converged"].mean()
                ),
                "median_bic_k2_minus_k1": float(
                    null_results["bic_k2_minus_k1"].median()
                ),
            }
        ]
    )


def software_versions() -> dict[str, str]:
    """Record numerical-library versions needed to interpret exact reproducibility."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": version("matplotlib"),
    }


def run_full_experiment(
    config: ExperimentConfig,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """Run, summarize, and persist the full preliminary experiment."""

    started = perf_counter()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_metrics = run_main_simulation(config)
    summary = summarize_repeated_simulation(raw_metrics)
    paired = summarize_paired_differences(raw_metrics)
    null_results = run_k1_null_experiment(config)
    null_summary = summarize_k1_null(null_results)

    raw_metrics.to_csv(output_dir / "main_simulation_raw_metrics.csv", index=False)
    summary.to_csv(output_dir / "main_simulation_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_method_differences.csv", index=False)
    null_results.to_csv(output_dir / "k1_null_raw_results.csv", index=False)
    null_summary.to_csv(output_dir / "k1_null_summary.csv", index=False)

    metadata = {
        "config": asdict(config),
        "software_versions": software_versions(),
        "primary_estimand": (
            "Expected top-1 true-mechanism ranking accuracy across repeated "
            "simulated studies."
        ),
        "subgroup_estimand": (
            "Expected false atrial classification among renal-impaired "
            "patients whose true mechanism is competing."
        ),
        "ci_definition": (
            "Two-sided 95% t confidence interval for the mean across paired "
            "simulation repeats; empirical 2.5th and 97.5th repeat quantiles "
            "are saved separately."
        ),
        "null_definition": (
            "False K=2 selection rate when truth is one homogeneous Gaussian "
            "regime after subtracting the prespecified renal biomarker path."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    manifest = write_manifest(
        output_dir,
        experiment="counterfactual",
        config=config,
        master_seed=config.seed,
        wall_clock_runtime_seconds=perf_counter() - started,
    )

    return {
        "raw_metrics": raw_metrics,
        "summary": summary,
        "paired_differences": paired,
        "null_results": null_results,
        "null_summary": null_summary,
        "figure_png": output_dir / "figure_P1.png",
        "figure_pdf": output_dir / "figure_P1.pdf",
        "metadata": metadata,
        "manifest": manifest,
    }


def _format_console_summary(artifacts: dict[str, Any]) -> str:
    summary = artifacts["summary"]
    selected = summary[
        (summary["method"].isin(PRIMARY_METHODS))
        & (
            summary["metric"].isin(
                [
                    "true_mechanism_accuracy",
                    "false_atrial_confounded_competing",
                ]
            )
        )
    ][
        [
            "renal_effect_sd",
            "method",
            "metric",
            "mean",
            "ci_low",
            "ci_high",
        ]
    ].copy()
    for column in ("mean", "ci_low", "ci_high"):
        selected[column] = (100.0 * selected[column]).round(2)
    null_summary = artifacts["null_summary"].copy()
    null_summary["false_k2_rate"] = (
        100.0 * null_summary["false_k2_rate"]
    ).round(2)
    null_summary["wilson_ci_low"] = (
        100.0 * null_summary["wilson_ci_low"]
    ).round(2)
    null_summary["wilson_ci_high"] = (
        100.0 * null_summary["wilson_ci_high"]
    ).round(2)
    return (
        "\nMain experiment (%):\n"
        + selected.to_string(index=False)
        + "\n\nK=1 null (% false K=2 selections):\n"
        + null_summary.to_string(index=False)
    )


def main() -> None:
    """Keep the historical standalone CLI available for notebook compatibility."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV, JSON, PNG, and PDF artifacts.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override main repeats per confounding level.",
    )
    parser.add_argument(
        "--null-repeats",
        type=int,
        default=None,
        help="Override K=1 null repeats.",
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=None,
        help="Override patients per main repeat.",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.repeats is not None:
        config = replace(config, repeats_per_level=args.repeats)
    if args.null_repeats is not None:
        config = replace(config, null_repeats=args.null_repeats)
    if args.patients is not None:
        config = replace(config, patients_per_repeat=args.patients)

    artifacts = run_full_experiment(config, args.output_dir)
    print(_format_console_summary(artifacts))
    print(f"\nArtifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
