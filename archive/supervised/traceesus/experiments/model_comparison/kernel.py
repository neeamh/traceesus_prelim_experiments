"""Fitted associative classifiers versus a fitted structural causal model.

This is the direct version of the user's question:

* Associative model: logistic regression learns a mapping from observed
  variables to the atrial-versus-competing label. It does not encode a graph or
  perform interventions.
* Structural causal model: a prespecified graph represents renal dysfunction as
  a direct cause of the NT-proBNP-like marker, estimates its structural
  equations from training data, and ranks mechanisms using posterior-integrated
  counterfactual disablement and sufficiency.

The adjusted logistic-regression control is scientifically necessary. It shows
whether the SCM wins because of its causal structure or merely because it was
given renal status while the associative baseline was not.
"""

from __future__ import annotations

import argparse
import json
import platform
from time import perf_counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logsumexp, logit

from traceesus.core.stats import (
    bounded_rate_monte_carlo_summary,
    paired_rate_contrast,
)
from traceesus.core.runner import nested_seed_sequence_ledger
from traceesus.core.io import write_manifest
from configs.model_comparison import ComparisonConfig

from archive.counterfactual.kernel import (
    ATRIAL,
    COMPETING,
    ExperimentConfig,
    simulate_two_mechanism_study,
)


ASSOCIATIVE_BIOMARKERS = "Associative logistic regression (biomarkers only)"
ASSOCIATIVE_ADJUSTED = "Associative logistic regression (+ kidney status)"
SCM_COUNTERFACTUAL = "Structural causal model (counterfactual)"
METHODS = (ASSOCIATIVE_BIOMARKERS, ASSOCIATIVE_ADJUSTED, SCM_COUNTERFACTUAL)


@dataclass(frozen=True)
class LogisticModel:
    """Retain the fitted standardization with coefficients for exact prediction.

    Prediction must reuse the training mean and scale; recomputing either on a
    test cohort would change both the estimand and the proposal-locked numbers.
    """

    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_sd: np.ndarray
    converged: bool
    iterations: int


@dataclass(frozen=True)
class FittedSCM:
    """Store the supervised structural-equation fit used by both causal queries.

    Mechanism labels enter this fit by design.  The object must therefore not be
    interpreted as evidence of unsupervised endotype recovery.
    """

    mechanism_effects: np.ndarray
    renal_effects: np.ndarray
    biomarker_noise_sd: np.ndarray
    prior_log_odds_intercept: float
    prior_renal_log_odds: float


def fit_logistic_regression(
    features: np.ndarray,
    atrial_label: np.ndarray,
    config: ComparisonConfig,
) -> LogisticModel:
    """Fit standard L2-penalized logistic regression by Newton iterations."""

    feature_mean = np.mean(features, axis=0)
    feature_sd = np.std(features, axis=0)
    feature_sd = np.where(feature_sd > 1e-12, feature_sd, 1.0)
    standardized = (features - feature_mean) / feature_sd
    design = np.column_stack((np.ones(features.shape[0]), standardized))

    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty_matrix = np.eye(design.shape[1])
    penalty_matrix[0, 0] = 0.0
    converged = False

    for iteration in range(1, config.logistic_max_iter + 1):
        probability = expit(design @ coefficients)
        working_weight = np.maximum(probability * (1.0 - probability), 1e-8)
        gradient = (
            design.T @ (probability - atrial_label)
            + config.logistic_l2_penalty * penalty_matrix @ coefficients
        )
        hessian = (
            design.T @ (working_weight[:, None] * design)
            + config.logistic_l2_penalty * penalty_matrix
        )
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.max(np.abs(step)) < config.logistic_tolerance:
            converged = True
            break

    return LogisticModel(
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_sd=feature_sd,
        converged=converged,
        iterations=iteration,
    )


def logistic_atrial_probability(
    model: LogisticModel,
    features: np.ndarray,
) -> np.ndarray:
    """Apply training-set scaling so test probabilities match the fitted classifier."""

    standardized = (features - model.feature_mean) / model.feature_sd
    design = np.column_stack((np.ones(features.shape[0]), standardized))
    return expit(design @ model.coefficients)


def fit_structural_causal_model(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    mechanism: np.ndarray,
    config: ComparisonConfig,
) -> FittedSCM:
    """Fit the prespecified mechanism/renal biomarker structural equations."""

    n = biomarkers.shape[0]
    design = np.column_stack(
        (
            (mechanism == ATRIAL).astype(float),
            (mechanism == COMPETING).astype(float),
            renal.astype(float),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, biomarkers, rcond=None)
    residual = biomarkers - design @ coefficients
    degrees_of_freedom = n - design.shape[1]
    noise_sd = np.sqrt(np.sum(residual**2, axis=0) / degrees_of_freedom)
    noise_sd = np.maximum(noise_sd, config.variance_floor)

    renal_normal = renal == 0
    renal_impaired = renal == 1
    count_normal = int(np.sum(renal_normal))
    count_impaired = int(np.sum(renal_impaired))
    atrial_normal = int(np.sum(renal_normal & (mechanism == ATRIAL)))
    atrial_impaired = int(np.sum(renal_impaired & (mechanism == ATRIAL)))
    smoothing = config.scm_prior_smoothing
    p_atrial_normal = (atrial_normal + smoothing) / (
        count_normal + 2.0 * smoothing
    )
    p_atrial_impaired = (atrial_impaired + smoothing) / (
        count_impaired + 2.0 * smoothing
    )
    prior_intercept = float(logit(p_atrial_normal))
    prior_renal_effect = float(
        logit(p_atrial_impaired) - prior_intercept
    )

    return FittedSCM(
        mechanism_effects=coefficients[:2, :],
        renal_effects=coefficients[2, :],
        biomarker_noise_sd=noise_sd,
        prior_log_odds_intercept=prior_intercept,
        prior_renal_log_odds=prior_renal_effect,
    )


def scm_posterior(
    model: FittedSCM,
    biomarkers: np.ndarray,
    renal: np.ndarray,
) -> np.ndarray:
    """Condition the fitted SCM on renal status for the same-model reference posterior.

    This posterior is retained to show whether the counterfactual score changes
    rankings beyond conditioning on the fitted kidney-aware SCM.
    """

    atrial_prior = expit(
        model.prior_log_odds_intercept
        + model.prior_renal_log_odds * renal
    )
    class_prior = np.column_stack((atrial_prior, 1.0 - atrial_prior))
    renal_contribution = renal[:, None] * model.renal_effects
    log_joint = np.empty((biomarkers.shape[0], 2), dtype=float)

    for candidate in (ATRIAL, COMPETING):
        candidate_mean = (
            model.mechanism_effects[candidate] + renal_contribution
        )
        standardized_residual = (
            biomarkers - candidate_mean
        ) / model.biomarker_noise_sd
        log_likelihood = (
            -0.5 * np.sum(standardized_residual**2, axis=1)
            - np.sum(np.log(model.biomarker_noise_sd))
        )
        log_joint[:, candidate] = (
            log_likelihood + np.log(class_prior[:, candidate])
        )

    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


def scm_counterfactual_scores(
    model: FittedSCM,
    biomarkers: np.ndarray,
    renal: np.ndarray,
    config: ComparisonConfig,
) -> dict[str, np.ndarray]:
    """Posterior-average disablement and sufficiency under the fitted SCM."""

    posterior = scm_posterior(model, biomarkers, renal)
    effects = model.mechanism_effects
    noise_sd = model.biomarker_noise_sd
    n = biomarkers.shape[0]
    renal_contribution = renal[:, None] * model.renal_effects

    exogenous_residual = np.empty((n, 2, biomarkers.shape[1]))
    for branch in (ATRIAL, COMPETING):
        exogenous_residual[:, branch, :] = (
            biomarkers - effects[branch] - renal_contribution
        )

    normalized_disablement = np.empty((n, 2))
    normalized_sufficiency = np.empty((n, 2))

    for candidate in (ATRIAL, COMPETING):
        disablement_by_branch = np.empty((n, 2))
        sufficiency_by_branch = np.empty((n, 2))

        for branch in (ATRIAL, COMPETING):
            disabled_effect = (
                np.zeros(biomarkers.shape[1])
                if branch == candidate
                else effects[branch]
            )
            if_disabled = (
                renal_contribution
                + disabled_effect
                + exogenous_residual[:, branch, :]
            )
            disablement_by_branch[:, branch] = np.sum(
                ((biomarkers - if_disabled) / noise_sd) ** 2,
                axis=1,
            )

            if_candidate_only = (
                renal_contribution
                + effects[candidate]
                + exogenous_residual[:, branch, :]
            )
            sufficient_distance = np.sum(
                ((biomarkers - if_candidate_only) / noise_sd) ** 2,
                axis=1,
            )
            sufficiency_by_branch[:, branch] = np.exp(
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
            posterior * sufficiency_by_branch, axis=1
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


def predict_from_atrial_probability(
    probability: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resolve exact 0.5 ties with the historical RNG stream to preserve pairing."""

    prediction = np.where(probability > 0.5, ATRIAL, COMPETING)
    tied = np.flatnonzero(np.isclose(probability, 0.5, rtol=0.0, atol=1e-12))
    if tied.size:
        prediction[tied] = rng.choice(
            np.array([ATRIAL, COMPETING]), size=tied.size
        )
    return prediction


def predict_from_scores(
    scores: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resolve score ties randomly because deterministic tie policy would change results."""

    maximum = np.max(scores, axis=1, keepdims=True)
    tied = np.isclose(scores, maximum, rtol=0.0, atol=1e-12)
    prediction = np.argmax(scores, axis=1)
    for row in np.flatnonzero(np.sum(tied, axis=1) > 1):
        prediction[row] = rng.choice(np.flatnonzero(tied[row]))
    return prediction


def metric_rows(
    *,
    repeat: int,
    strength_index: int,
    renal_effect_sd: float,
    method: str,
    truth: np.ndarray,
    renal: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    """Emit the two prespecified estimands without changing repeat or method order."""

    subgroup = (renal == 1) & (truth == COMPETING)
    subgroup_n = int(np.sum(subgroup))
    if subgroup_n == 0:
        raise RuntimeError("The renal-impaired competing-mechanism subgroup is empty.")
    return [
        {
            "repeat": repeat,
            "strength_index": strength_index,
            "renal_effect_sd": renal_effect_sd,
            "method": method,
            "metric": "true_mechanism_accuracy",
            "value": float(np.mean(prediction == truth)),
            "denominator": int(truth.size),
        },
        {
            "repeat": repeat,
            "strength_index": strength_index,
            "renal_effect_sd": renal_effect_sd,
            "method": method,
            "metric": "false_atrial_confounded_competing",
            "value": float(np.mean(prediction[subgroup] == ATRIAL)),
            "denominator": subgroup_n,
        },
    ]


def run_comparison(config: ComparisonConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run 500 paired train/test experiments at every renal-effect level."""

    config.validate()
    dgp_template = ExperimentConfig(seed=config.seed)
    training_dgp = replace(
        dgp_template, patients_per_repeat=config.training_patients
    )
    test_dgp = replace(
        dgp_template, patients_per_repeat=config.test_patients
    )
    rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    seed_ledger = nested_seed_sequence_ledger(
        config.seed,
        len(config.confounding_strengths_sd),
        config.repeats_per_level,
    )

    for strength_index, (renal_effect_sd, repeat_seeds) in enumerate(
        zip(config.confounding_strengths_sd, seed_ledger, strict=True)
    ):
        for repeat, repeat_seed in enumerate(repeat_seeds):
            train_seed, test_seed, tie_seed = repeat_seed.spawn(3)
            training = simulate_two_mechanism_study(
                training_dgp,
                renal_effect_sd,
                np.random.default_rng(train_seed),
            )
            test = simulate_two_mechanism_study(
                test_dgp,
                renal_effect_sd,
                np.random.default_rng(test_seed),
            )
            atrial_training_label = (
                training["mechanism"] == ATRIAL
            ).astype(float)

            biomarkers_only_model = fit_logistic_regression(
                training["biomarkers"],
                atrial_training_label,
                config,
            )
            adjusted_training_features = np.column_stack(
                (training["biomarkers"], training["renal"])
            )
            adjusted_model = fit_logistic_regression(
                adjusted_training_features,
                atrial_training_label,
                config,
            )
            structural_model = fit_structural_causal_model(
                training["biomarkers"],
                training["renal"],
                training["mechanism"],
                config,
            )

            tie_rngs = [
                np.random.default_rng(seed) for seed in tie_seed.spawn(3)
            ]
            biomarkers_only_prediction = predict_from_atrial_probability(
                logistic_atrial_probability(
                    biomarkers_only_model, test["biomarkers"]
                ),
                tie_rngs[0],
            )
            adjusted_test_features = np.column_stack(
                (test["biomarkers"], test["renal"])
            )
            adjusted_prediction = predict_from_atrial_probability(
                logistic_atrial_probability(
                    adjusted_model, adjusted_test_features
                ),
                tie_rngs[1],
            )
            counterfactual = scm_counterfactual_scores(
                structural_model,
                test["biomarkers"],
                test["renal"],
                config,
            )
            structural_prediction = predict_from_scores(
                counterfactual["combined"], tie_rngs[2]
            )

            for method, prediction in (
                (ASSOCIATIVE_BIOMARKERS, biomarkers_only_prediction),
                (ASSOCIATIVE_ADJUSTED, adjusted_prediction),
                (SCM_COUNTERFACTUAL, structural_prediction),
            ):
                rows.extend(
                    metric_rows(
                        repeat=repeat,
                        strength_index=strength_index,
                        renal_effect_sd=renal_effect_sd,
                        method=method,
                        truth=test["mechanism"],
                        renal=test["renal"],
                        prediction=prediction,
                    )
                )

            fit_rows.append(
                {
                    "repeat": repeat,
                    "strength_index": strength_index,
                    "renal_effect_sd": renal_effect_sd,
                    "biomarkers_only_converged": biomarkers_only_model.converged,
                    "biomarkers_only_iterations": biomarkers_only_model.iterations,
                    "adjusted_converged": adjusted_model.converged,
                    "adjusted_iterations": adjusted_model.iterations,
                    "scm_renal_effect_nt_like": structural_model.renal_effects[0],
                    "scm_renal_effect_atrial_electrical": structural_model.renal_effects[1],
                    "scm_renal_effect_competing_marker": structural_model.renal_effects[2],
                    "maximum_counterfactual_posterior_difference": float(
                        np.max(
                            np.abs(
                                counterfactual["combined"]
                                - counterfactual["posterior"]
                            )
                        )
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    fits = pd.DataFrame(fit_rows)
    expected_metric_rows = (
        len(config.confounding_strengths_sd)
        * config.repeats_per_level
        * len(METHODS)
        * 2
    )
    if len(metrics) != expected_metric_rows:
        raise AssertionError(
            f"Expected {expected_metric_rows} metric rows, got {len(metrics)}."
        )
    return metrics, fits


def summarize_metrics(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Reduce repeats in legacy group order so floating-point accumulation is stable."""

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


def summarize_paired_contrasts(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Use within-repeat differences to preserve the experiment's paired design."""

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
    contrasts = (
        ("scm_minus_biomarkers_only", SCM_COUNTERFACTUAL, ASSOCIATIVE_BIOMARKERS),
        ("scm_minus_kidney_adjusted", SCM_COUNTERFACTUAL, ASSOCIATIVE_ADJUSTED),
    )
    rows: list[dict[str, Any]] = []
    for keys, group in pivot.groupby(
        ["strength_index", "renal_effect_sd", "metric"], sort=True
    ):
        for contrast_name, first, second in contrasts:
            difference = group[first] - group[second]
            # For false-atrium errors, reverse the sign so positive means fewer
            # errors for the SCM.
            if keys[2] == "false_atrial_confounded_competing":
                difference = -difference
                contrast_name_output = contrast_name.replace(
                    "scm_minus", "associative_minus_scm"
                )
            else:
                contrast_name_output = contrast_name
            statistics = paired_rate_contrast(difference)
            rows.append(
                {
                    "strength_index": keys[0],
                    "renal_effect_sd": keys[1],
                    "metric": keys[2],
                    "contrast": contrast_name_output,
                    **statistics,
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(
    summary: pd.DataFrame,
    config: ComparisonConfig,
    output_path: Path,
) -> None:
    """Plot overall accuracy and subgroup false atrial classification."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_style = {
        ASSOCIATIVE_BIOMARKERS: {
            "color": "#C65A23",
            "marker": "o",
            "linestyle": "--",
            "markerfacecolor": "white",
            "linewidth": 2.2,
        },
        ASSOCIATIVE_ADJUSTED: {
            "color": "#68737D",
            "marker": "^",
            "linestyle": ":",
            "markerfacecolor": "white",
            "linewidth": 1.8,
        },
        SCM_COUNTERFACTUAL: {
            "color": "#235789",
            "marker": "s",
            "linestyle": "-",
            "markerfacecolor": "#235789",
            "linewidth": 2.3,
        },
    }
    panels = (
        (
            "true_mechanism_accuracy",
            "A",
            "True-mechanism classification accuracy",
            "Patients correctly classified (%)",
        ),
        (
            "false_atrial_confounded_competing",
            "B",
            "False atrial classification in the renal subgroup",
            "Competing-mechanism patients classified atrial (%)",
        ),
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), constrained_layout=True)
    x = np.arange(len(config.confounding_strengths_sd))

    for axis, (metric, letter, title, ylabel) in zip(
        axes, panels, strict=True
    ):
        for method in METHODS:
            selected = summary[
                (summary["metric"] == metric)
                & (summary["method"] == method)
            ].sort_values("strength_index")
            mean = 100.0 * selected["mean"].to_numpy()
            low = 100.0 * selected["ci_low"].to_numpy()
            high = 100.0 * selected["ci_high"].to_numpy()
            style = method_style[method]
            axis.fill_between(
                x,
                low,
                high,
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )
            axis.errorbar(
                x,
                mean,
                yerr=np.vstack((mean - low, high - mean)),
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=6.0,
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=1.3,
                elinewidth=1.0,
                capsize=2.8,
                capthick=1.0,
            )
        axis.set_xticks(x, config.confounding_labels)
        axis.set_xlabel("Direct renal effect on the NT-proBNP-like marker")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{letter}. {title}", loc="left", fontweight="semibold")
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#4A5560")

    accuracy_rows = summary[summary["metric"] == "true_mechanism_accuracy"]
    accuracy_min = 100.0 * accuracy_rows["ci_low"].min()
    axes[0].set_ylim(
        max(0.0, np.floor((accuracy_min - 3.0) / 5.0) * 5.0),
        85.0,
    )
    axes[1].set_ylim(0.0, 45.0)
    axes[0].legend(
        frameon=False,
        loc="lower left",
        fontsize=8.1,
    )
    fig.suptitle(
        "Figure P1. Associative classifiers versus a structural causal model",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#20262E",
    )
    fig.text(
        0.01,
        -0.025,
        (
            f"Means across {config.repeats_per_level} paired train/test simulations "
            f"({config.training_patients:,} training; {config.test_patients:,} test "
            "patients per repeat); shaded bands and error bars are 95% Monte "
            "Carlo CIs. All methods use the same training and test patients."
        ),
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4A5560",
    )
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_full_comparison(
    config: ComparisonConfig,
    output_dir: str | Path = "outputs_associative_vs_scm",
) -> dict[str, Any]:
    """Preserve the standalone-script workflow for notebooks and legacy callers.

    The OOP facade is preferred for new runs, but this compatibility entry point
    keeps historical notebook imports and output filenames operational.
    """

    started = perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics, fit_diagnostics = run_comparison(config)
    summary = summarize_metrics(raw_metrics)
    paired = summarize_paired_contrasts(raw_metrics)
    raw_metrics.to_csv(output_dir / "raw_metrics.csv", index=False)
    fit_diagnostics.to_csv(output_dir / "fit_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired.to_csv(output_dir / "paired_contrasts.csv", index=False)
    plot_comparison(summary, config, output_dir / "figure_P1_associative_vs_scm.png")
    metadata = {
        "config": asdict(config),
        "models": {
            ASSOCIATIVE_BIOMARKERS: (
                "L2-penalized logistic regression fit to the three continuous "
                "biomarkers; no graph and no renal-status input."
            ),
            ASSOCIATIVE_ADJUSTED: (
                "The same logistic regression fit to the three biomarkers plus "
                "observed renal status; no graph and no interventions."
            ),
            SCM_COUNTERFACTUAL: (
                "Fitted prespecified SCM with mechanism and renal biomarker "
                "equations; posterior-integrated disablement and sufficiency."
            ),
        },
        "training_label_boundary": (
            "All three models use known synthetic mechanism labels during "
            "training. This is supervised classification, not unsupervised "
            "endotype discovery."
        ),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "ci_definition": (
            "Two-sided 95% t confidence interval for the mean across 500 paired "
            "train/test simulation repeats."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    manifest = write_manifest(
        output_dir,
        experiment="model_comparison",
        config=config,
        master_seed=config.seed,
        wall_clock_runtime_seconds=perf_counter() - started,
    )
    return {
        "raw_metrics": raw_metrics,
        "fit_diagnostics": fit_diagnostics,
        "summary": summary,
        "paired_contrasts": paired,
        "figure_png": output_dir / "figure_P1_associative_vs_scm.png",
        "figure_pdf": output_dir / "figure_P1_associative_vs_scm.pdf",
        "metadata": metadata,
        "manifest": manifest,
    }


def console_summary(artifacts: dict[str, Any]) -> str:
    """Render the historical diagnostic text without recomputing experiment results."""

    summary = artifacts["summary"][
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
        summary[column] = (100.0 * summary[column]).round(2)
    fit = artifacts["fit_diagnostics"]
    return (
        summary.to_string(index=False)
        + "\n\nFit diagnostics:\n"
        + f"biomarkers-only convergence={fit['biomarkers_only_converged'].mean():.3f}, "
        + f"adjusted convergence={fit['adjusted_converged'].mean():.3f}, "
        + "max |counterfactual - same-SCM posterior|="
        + f"{fit['maximum_counterfactual_posterior_difference'].max():.3e}"
    )


def main() -> None:
    """Keep the original script CLI as a compatibility path into the frozen kernel."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_associative_vs_scm"),
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--training-patients", type=int, default=None)
    parser.add_argument("--test-patients", type=int, default=None)
    args = parser.parse_args()

    config = ComparisonConfig()
    if args.repeats is not None:
        config = replace(config, repeats_per_level=args.repeats)
    if args.training_patients is not None:
        config = replace(config, training_patients=args.training_patients)
    if args.test_patients is not None:
        config = replace(config, test_patients=args.test_patients)

    artifacts = run_full_comparison(config, args.output_dir)
    print(console_summary(artifacts))
    print(f"\nArtifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
