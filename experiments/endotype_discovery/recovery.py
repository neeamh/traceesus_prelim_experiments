"""Registry-driven paired recovery while preserving the legacy RNG ledger."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from traceesus.core.metrics import evaluate_binary_posterior
from traceesus.core.model import (
    FitDiagnostics,
    FittedModel,
    Model,
    assert_truth_free_fit_interfaces,
)
from traceesus.core.runner import (
    latent_recovery_seed_ledger,
    ordered_map,
    redundancy_sweep_seed_ledger,
)
from traceesus.core.stats import paired_mean_contrast
from traceesus.models.oracle import DataGeneratingOracle
from traceesus.simulators.two_mechanism import TwoMechanismSimulator
from traceesus.core.simulator import Cohort

from . import kernel


_LOCKED_MODEL_COUNT = 3


def _fit_stream_indices(model_index: int) -> tuple[int, int]:
    """Map registry rows to append-stable initial and retry streams.

    The three cited models keep legacy initial streams 2--4 and retry streams
    5--7.  New appended variants receive stream pairs beginning at 8, so adding
    a row cannot shift a historical model's retry generator.
    """

    if model_index < _LOCKED_MODEL_COUNT:
        return 2 + model_index, 2 + _LOCKED_MODEL_COUNT + model_index
    extra_index = model_index - _LOCKED_MODEL_COUNT
    return 2 + 2 * _LOCKED_MODEL_COUNT + 2 * extra_index, 3 + 2 * _LOCKED_MODEL_COUNT + 2 * extra_index


def _legacy_fit_result(fitted: FittedModel) -> Any:
    """Return a frozen-kernel fit record only for legacy parameter recovery."""

    try:
        return fitted.fit_result
    except AttributeError as error:
        raise TypeError(
            "Legacy parameter-recovery models must retain their frozen fit_result."
        ) from error


def _diagnostic_row(
    method: str,
    diagnostics: FitDiagnostics,
) -> dict[str, object]:
    """Serialize convergence diagnostics in the historical column order."""

    return {
        "method": method,
        "converged": diagnostics.converged,
        "iterations": diagnostics.iterations,
        "best_start": diagnostics.best_start,
        "log_likelihood": diagnostics.log_likelihood,
        "minimum_effective_class_fraction": float(
            np.min(diagnostics.effective_class_fraction)
        ),
        "anchor_margin": diagnostics.anchor_margin,
    }


def _nuisance_subgroups(observed: Cohort) -> dict[str, np.ndarray]:
    """Partition patients into the four prespecified nuisance profiles.

    Masks are defined on nuisance status only.  ``evaluate_binary_posterior``
    applies the competing-mechanism filter for false-atrial attribution, so a
    caller cannot accidentally use a different denominator than the metric.

    The caller gates this on sweep mode: the locked experiment must keep its
    exact historical output schema, so subgroup columns exist only in
    redundancy-sweep artifacts.
    """

    renal = observed.covariate("renal_dysfunction") == 1
    heart_failure = observed.covariate("heart_failure") == 1
    return {
        "uncomplicated": ~renal & ~heart_failure,
        "renal_only": renal & ~heart_failure,
        "heart_failure_only": ~renal & heart_failure,
        "redundant": renal & heart_failure,
    }


def _run_registry_repeat(
    task: tuple[
        int,
        int,
        float,
        kernel.ExperimentConfig,
        tuple[Model, ...],
    ],
) -> dict[str, object]:
    """Execute one paired repeat with the exact legacy eight-stream layout.

    A sixth task element, when present, is the heart-failure effect for the
    redundancy sweep.  The locked five-element path is untouched: subgroup
    columns and the HF level column appear only in sweep tasks, so the cited
    output files keep their exact historical schema.
    """

    repeat_index, seed, renal_effect_sd, config, models = task[:5]
    heart_failure_effect_sd: float | None = task[5] if len(task) > 5 else None
    extra_models = max(0, len(models) - _LOCKED_MODEL_COUNT)
    child_sequences = np.random.SeedSequence(seed).spawn(
        2 + 2 * _LOCKED_MODEL_COUNT + 2 * extra_models
    )
    simulator = TwoMechanismSimulator(
        config.simulation,
        renal_effect_sd,
        heart_failure_effect_sd if heart_failure_effect_sd is not None else 0.0,
    )
    training = simulator.simulate(
        np.random.default_rng(child_sequences[0]),
        config.simulation.training_patients,
    )
    test = simulator.simulate(
        np.random.default_rng(child_sequences[1]),
        config.simulation.test_patients,
    )

    fitted_models: list[tuple[Model, FittedModel, FitDiagnostics | None]] = []
    retry_config = replace(
        config.fitting,
        random_starts=max(config.fitting.random_starts, 8),
        maximum_em_iterations=max(config.fitting.maximum_em_iterations, 1_200),
    )
    for model_index, model in enumerate(models):
        initial_stream, retry_stream = _fit_stream_indices(model_index)
        fitted = model.fit(
            training.observed,
            np.random.default_rng(child_sequences[initial_stream]),
            config.fitting,
        )
        diagnostics = fitted.fit_diagnostics()
        if diagnostics is not None and not bool(diagnostics.converged):
            fitted = model.fit(
                training.observed,
                np.random.default_rng(child_sequences[retry_stream]),
                retry_config,
            )
            diagnostics = fitted.fit_diagnostics()
        fitted_models.append((model, fitted, diagnostics))

    posterior_by_method: dict[str, np.ndarray] = {
        model.name: fitted.posterior(test.observed)
        for model, fitted, _ in fitted_models
    }
    if heart_failure_effect_sd is None:
        # Locked path only: the renal-only oracle is the true-DGP ceiling here.
        # Under a nonzero HF path it would be misspecified, so sweep runs
        # deliberately omit it rather than report a false ceiling.
        oracle = DataGeneratingOracle(renal_effect_sd, config.simulation)
        oracle_fitted = oracle.fit(
            test.observed, np.random.default_rng(0), config.fitting
        )
        posterior_by_method[oracle.name] = oracle_fitted.posterior(test.observed)

    metric_rows: list[dict[str, object]] = []
    renal = test.observed.covariate("renal_dysfunction")
    subgroups = (
        _nuisance_subgroups(test.observed)
        if heart_failure_effect_sd is not None
        else None
    )
    for method, posterior in posterior_by_method.items():
        row: dict[str, object] = {
            "repeat": repeat_index,
            "renal_effect_sd": renal_effect_sd,
            "method": method,
        }
        if heart_failure_effect_sd is not None:
            row["heart_failure_effect_sd"] = heart_failure_effect_sd
        row.update(
            evaluate_binary_posterior(
                posterior,
                test.truth.mechanism,
                renal,
                config.fitting.calibration_bins,
                atrial=kernel.Mechanism.ATRIAL,
                competing=kernel.Mechanism.COMPETING,
                subgroups=subgroups,
            )
        )
        metric_rows.append(row)

    diagnostic_rows: list[dict[str, object]] = []
    for model, _, diagnostics in fitted_models:
        if diagnostics is None:
            continue
        row = {"repeat": repeat_index, "renal_effect_sd": renal_effect_sd}
        row.update(_diagnostic_row(model.name, diagnostics))
        diagnostic_rows.append(row)

    fitted_by_name = {model.name: fitted for model, fitted, _ in fitted_models}
    parameter_row = {
        "repeat": repeat_index,
        "renal_effect_sd": renal_effect_sd,
    }
    if (
        kernel.CAUSAL_SCM in fitted_by_name
        and kernel.ASSOCIATIVE_ADJUSTED in fitted_by_name
    ):
        causal_fit = _legacy_fit_result(fitted_by_name[kernel.CAUSAL_SCM])
        adjusted_fit = _legacy_fit_result(fitted_by_name[kernel.ASSOCIATIVE_ADJUSTED])
        parameter_row.update(
            {
                "causal_estimated_renal_effect_nt": causal_fit.renal_effect[
                    kernel.Biomarker.NT_PROBNP_LIKE
                ],
                "adjusted_estimated_renal_effect_nt": adjusted_fit.renal_effect[
                    kernel.Biomarker.NT_PROBNP_LIKE
                ],
                "adjusted_estimated_renal_effect_electrical": adjusted_fit.renal_effect[
                    kernel.Biomarker.ATRIAL_ELECTRICAL
                ],
                "adjusted_estimated_renal_effect_competing": adjusted_fit.renal_effect[
                    kernel.Biomarker.COMPETING_SPECIFIC
                ],
            }
        )
    return {
        "metrics": metric_rows,
        "diagnostics": diagnostic_rows,
        "parameters": parameter_row,
    }


def run_model_registry(
    config: kernel.ExperimentConfig,
    models: Sequence[Model],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run every registered model on identical cohorts and paired fit streams.

    Adding a fitted recovery variant requires a new ``Model`` subclass and one
    registry entry.  The simulator, evaluation, retry, and row-assembly logic
    remain unchanged.  Existing model order is scientifically locked because it
    maps directly to the three historical initial-fit and retry streams.
    """

    config.validate()
    registered = tuple(models)
    if not registered:
        raise ValueError("At least one fitted recovery model must be registered.")
    model_names = tuple(model.name for model in registered)
    if len(set(model_names)) != len(model_names):
        raise ValueError("Registered recovery model names must be unique.")
    if DataGeneratingOracle.name in model_names:
        raise ValueError("The data-generating oracle name is reserved for evaluation.")
    assert_truth_free_fit_interfaces(registered)
    ledgers = latent_recovery_seed_ledger(
        config.master_seed,
        len(config.simulation.renal_effect_levels_sd),
        config.repeats_per_level,
    )
    tasks: list[tuple[int, int, float, kernel.ExperimentConfig, tuple[Model, ...]]] = []
    for renal_effect_sd, seeds in zip(
        config.simulation.renal_effect_levels_sd,
        ledgers,
        strict=True,
    ):
        for repeat_index, seed in enumerate(seeds):
            tasks.append((repeat_index, seed, renal_effect_sd, config, registered))

    results = ordered_map(_run_registry_repeat, tasks, config.workers)
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


_LOCKED_CONTRAST_METRICS = (
    "accuracy",
    "adjusted_rand_index",
    "false_atrial_renal_competing",
    "brier_score",
    "expected_calibration_error",
)


def _contrast_metrics(raw_metrics: pd.DataFrame) -> tuple[str, ...]:
    """Return locked metrics first, then any subgroup metrics, deterministically.

    Locked names keep their historical position so existing contrast rows do not
    move; subgroup columns are appended in sorted order.  This mirrors the
    append-stable seed ledger — a new column extends the output rather than
    reshuffling it.
    """

    appended = sorted(
        column
        for column in raw_metrics.columns
        if column.startswith(("accuracy__", "false_atrial__", "mean_posterior_entropy__"))
    )
    return (*_LOCKED_CONTRAST_METRICS, *appended)


def paired_registry_contrasts(
    raw_metrics: pd.DataFrame,
    models: Sequence[Model],
    *,
    level_column: str = "renal_effect_sd",
) -> pd.DataFrame:
    """Compare the causal reference with every other registered fitted model.

    Deriving comparators from the registry makes an appended ablation row flow
    into paired outputs without editing the repeat or summary machinery.  The
    current three-model registry produces the historical row order exactly.

    ``level_column`` defaults to the locked renal level so cited contrast files
    are byte-identical; the redundancy sweep passes its own level column.
    """

    metrics = _contrast_metrics(raw_metrics)

    comparators = tuple(
        model.name for model in models if model.name != kernel.CAUSAL_SCM
    )
    rows: list[dict[str, object]] = []
    for level_value in sorted(raw_metrics[level_column].unique()):
        level = raw_metrics[raw_metrics[level_column] == level_value]
        for comparator in comparators:
            for metric in metrics:
                wide = level.pivot(index="repeat", columns="method", values=metric)
                causal_values = wide[kernel.CAUSAL_SCM].to_numpy(dtype=float)
                comparator_values = wide[comparator].to_numpy(dtype=float)
                paired = np.isfinite(causal_values) & np.isfinite(comparator_values)
                difference = causal_values[paired] - comparator_values[paired]
                if difference.size < 2:
                    rows.append(
                        {
                            level_column: level_value,
                            "comparator": comparator,
                            "metric": metric,
                            "difference_definition": "causal SCM minus comparator",
                            "repeat_count": int(difference.size),
                            "mean_difference": float("nan"),
                            "ci95_low": float("nan"),
                            "ci95_high": float("nan"),
                        }
                    )
                    continue
                rows.append(
                    {
                        level_column: level_value,
                        "comparator": comparator,
                        "metric": metric,
                        "difference_definition": "causal SCM minus comparator",
                        **paired_mean_contrast(difference),
                    }
                )
    return pd.DataFrame(rows)


def run_redundancy_sweep(
    config: kernel.ExperimentConfig,
    models: Sequence[Model],
    *,
    renal_effect_sd: float | None = None,
    heart_failure_levels_sd: tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sweep the heart-failure path at a fixed strong renal distortion.

    Renal distortion stays at the locked strong level so NT-proBNP is already
    contaminated; sweeping the PTFV1 heart-failure path then creates genuine
    redundancy — two nuisance causes jointly reproducing the atrial signature.
    The sweep has its own salted seed root, so it can never perturb, and is
    never confused with, the proposal-locked recovery ledger.

    The renal-only data-generating oracle is intentionally absent from sweep
    artifacts: with a nonzero HF path it would no longer be the true-DGP
    ceiling.
    """

    config.validate()
    registered = tuple(models)
    if not registered:
        raise ValueError("At least one fitted recovery model must be registered.")
    model_names = tuple(model.name for model in registered)
    if len(set(model_names)) != len(model_names):
        raise ValueError("Registered recovery model names must be unique.")
    assert_truth_free_fit_interfaces(registered)

    fixed_renal = (
        renal_effect_sd
        if renal_effect_sd is not None
        else config.null_renal_effect_sd
    )
    levels = (
        heart_failure_levels_sd
        if heart_failure_levels_sd is not None
        else config.simulation.heart_failure_effect_levels_sd
    )
    ledgers = redundancy_sweep_seed_ledger(
        config.master_seed,
        len(levels),
        config.repeats_per_level,
    )
    tasks: list[tuple] = []
    for heart_failure_effect_sd, seeds in zip(levels, ledgers, strict=True):
        for repeat_index, seed in enumerate(seeds):
            tasks.append(
                (
                    repeat_index,
                    seed,
                    fixed_renal,
                    config,
                    registered,
                    heart_failure_effect_sd,
                )
            )

    results = ordered_map(_run_registry_repeat, tasks, config.workers)
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


def registry_validation_checks(
    raw_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    parameters: pd.DataFrame,
    raw_null: pd.DataFrame,
    config: kernel.ExperimentConfig,
    models: Sequence[Model],
) -> dict[str, object]:
    """Retain legacy checks while deriving recovery row counts from the registry."""

    checks = kernel.validation_checks(
        raw_metrics,
        diagnostics,
        parameters,
        raw_null,
        config,
    )
    level_count = len(config.simulation.renal_effect_levels_sd)
    checks["metric_row_count_matches"] = raw_metrics.shape[0] == (
        level_count * config.repeats_per_level * (len(models) + 1)
    )
    diagnostic_method_count = diagnostics["method"].nunique()
    checks["diagnostic_row_count_matches"] = diagnostics.shape[0] == (
        level_count * config.repeats_per_level * diagnostic_method_count
    )

    redundant_sizes = raw_metrics.get("competing_subgroup_size__redundant")
    if redundant_sizes is not None:
        checks["redundant_subgroup_median_size"] = float(redundant_sizes.median())
        checks["redundant_subgroup_analyzable"] = bool(
            redundant_sizes.median() >= 30
        )

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
