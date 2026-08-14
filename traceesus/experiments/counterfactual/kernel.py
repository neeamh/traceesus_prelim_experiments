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
from scipy.special import expit

from traceesus.core.stats import (
    bic,
    bounded_rate_monte_carlo_summary,
    paired_rate_contrast,
    wilson_interval as shared_wilson_interval,
)
from traceesus.core.runner import nested_seed_sequence_ledger, seed_sequence_ledger
from traceesus.core.io import write_manifest
from traceesus.simulators.two_mechanism import TwoMechanismSimulator
from traceesus.models.known_scm import (
    fit_one_diagonal_gaussian as _fit_one_diagonal_gaussian,
    fit_two_diagonal_gaussians as _fit_two_diagonal_gaussians,
    kidney_aware_posterior,
    kidney_blind_posterior,
    posterior_integrated_counterfactual_scores,
)
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
