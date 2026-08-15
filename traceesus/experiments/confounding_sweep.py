"""Confounding sweep: renal dysfunction as a common cause of mechanism and marker.

Scientific design
-----------------
The locked endotype-discovery experiment sets

    P(atrial | kidneys normal) = P(atrial | kidneys impaired) = 0.50

so renal status is *not* a confounder there.  It is a second parent of a shared
child (Z -> NT-proBNP <- R), and the failure it produces is misattribution at a
collider, not confounding bias.

This sweep opens the R -> Z edge while leaving R -> NT-proBNP in place:

    R -> Z            (prior shift, swept)
    R -> NT-proBNP    (direct path, fixed at the locked strong level)
    Z -> NT-proBNP, PTFV1

Renal status becomes a genuine confounder of the Z -> B relationship.  The
clinically plausible direction is *positive* — chronic kidney disease and atrial
cardiopathy share age, hypertension and diabetes as risk factors — so the sweep
runs upward from the locked 0.50.

Two opposing predictions are being tested at once:

1.  **Pooled minus causal should shrink.**  An unadjusted model calls renally
    impaired patients atrial because their NT-proBNP is high.  Once impairment
    genuinely predicts atrial mechanism, that bias partly coincides with the
    truth, so the pooled model is right for the wrong reason.

2.  **Adjusted minus causal should grow.**  Once R carries information about Z,
    an unconstrained renal coefficient on every marker can absorb genuine
    mechanism signal — over-adjustment — while the biology mask forbids it.
    This is the first regime in the package where the causal *constraint*, as
    opposed to the presence of a nuisance term, has something to win.

Both fitted nuisance-aware models already represent p(Z | R); the only
difference between them remains the path mask.  So any divergence here is
attributable to the constraint alone.

Nothing in this module touches a proposal-cited artifact.  The default
dependency level reproduces the locked configuration exactly, the output
directory and named seed root are separate from locked outputs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from traceesus.core.metrics import evaluate_binary_posterior
from traceesus.core.runner import ordered_map, spawned_uint64_seeds
from traceesus.core.seeds import CONFOUNDING_SWEEP_SEED_OFFSET
from traceesus.experiments.endotype_discovery import kernel
from traceesus.models import (
    AdjustedLatentClassModel,
    AssociativeLatentClassModel,
    BiologicallyConstrainedCausalSCM,
    DataGeneratingOracle,
)
from traceesus.simulators.two_mechanism import TwoMechanismSimulator

# 0.50 is the locked, unconfounded setting and is kept as the first level so
# every sweep contains its own control.
DEPENDENCE_LEVELS = (0.50, 0.55, 0.60, 0.70, 0.80)

POOLED = kernel.ASSOCIATIVE_LCA
ADJUSTED = kernel.ASSOCIATIVE_ADJUSTED
CAUSAL = kernel.CAUSAL_SCM
ORACLE = kernel.ORACLE


def _fitted_prior(fitted) -> float:
    """Recover the model's fitted P(atrial | kidneys impaired), or NaN."""

    for attribute in vars(fitted).values():
        table = getattr(attribute, "class_probability_by_renal", None)
        if table is not None:
            return float(np.asarray(table, dtype=float)[1, 0])
    return float("nan")


def _run_cell(task) -> list[dict[str, object]]:
    repeat_index, seed, dependence, renal_sd, config = task

    simulation = replace(
        config.simulation, atrial_probability_if_renal_impaired=float(dependence)
    )
    models = (
        AssociativeLatentClassModel(),
        AdjustedLatentClassModel(),
        BiologicallyConstrainedCausalSCM(),
        DataGeneratingOracle(renal_sd, simulation),
    )
    sequences = np.random.SeedSequence(seed).spawn(2 + len(models))
    simulator = TwoMechanismSimulator(simulation, renal_sd, 0.0)
    training = simulator.simulate(
        np.random.default_rng(sequences[0]), simulation.training_patients
    )
    test = simulator.simulate(
        np.random.default_rng(sequences[1]), simulation.test_patients
    )
    retry = replace(
        config.fitting,
        random_starts=max(config.fitting.random_starts, 8),
        maximum_em_iterations=max(config.fitting.maximum_em_iterations, 1_200),
    )

    rows: list[dict[str, object]] = []
    for index, model in enumerate(models):
        stream = np.random.default_rng(sequences[2 + index])
        fitted = model.fit(training.observed, stream, config.fitting)
        diagnostics = fitted.fit_diagnostics()
        if diagnostics is not None and not diagnostics.converged:
            fitted = model.fit(
                training.observed, np.random.default_rng(sequences[2 + index]), retry
            )
        row: dict[str, object] = {
            "repeat": repeat_index,
            "atrial_probability_if_renal_impaired": dependence,
            "renal_effect_sd": renal_sd,
            "method": model.name,
            "fitted_prior_atrial_given_renal": _fitted_prior(fitted),
        }
        row.update(
            evaluate_binary_posterior(
                fitted.posterior(test.observed),
                test.truth.mechanism,
                test.observed.covariate("renal_dysfunction"),
                config.fitting.calibration_bins,
                atrial=int(kernel.Mechanism.ATRIAL),
                competing=int(kernel.Mechanism.COMPETING),
            )
        )
        rows.append(row)
    return rows


def run_confounding_sweep(
    config: kernel.ExperimentConfig,
    *,
    dependence_levels: tuple[float, ...] = DEPENDENCE_LEVELS,
    renal_effect_sd: float | None = None,
    repeats: int = 150,
    workers: int = 4,
) -> pd.DataFrame:
    """Sweep P(atrial | kidneys impaired) at a fixed direct renal effect."""

    config.validate()
    renal_effect_sd = (
        config.simulation.renal_effect_levels_sd[-1]
        if renal_effect_sd is None
        else renal_effect_sd
    )
    root = np.random.SeedSequence(
        config.master_seed + CONFOUNDING_SWEEP_SEED_OFFSET
    )
    tasks = []
    for dependence, child in zip(
        dependence_levels, root.spawn(len(dependence_levels)), strict=True
    ):
        for repeat_index, seed in enumerate(spawned_uint64_seeds(child, repeats)):
            tasks.append((repeat_index, seed, dependence, renal_effect_sd, config))
    results = ordered_map(_run_cell, tasks, workers)
    return pd.DataFrame([row for rows in results for row in rows])


def paired_confounding_contrasts(raw: pd.DataFrame) -> pd.DataFrame:
    """Within-repeat differences against the causal model, per dependence level.

    Pairing is exact: every method in a repeat saw the identical cohort, so each
    repeat contributes one difference and the interval is a genuine paired
    comparison rather than a difference of independent means.
    """

    from scipy.stats import t

    rows: list[dict[str, object]] = []
    metrics = ("accuracy", "false_atrial_renal_competing")
    for level, block in raw.groupby("atrial_probability_if_renal_impaired"):
        for comparator in (POOLED, ADJUSTED, ORACLE):
            for metric in metrics:
                wide = block.pivot(index="repeat", columns="method", values=metric)
                if comparator not in wide or CAUSAL not in wide:
                    continue
                paired = (
                    wide[CAUSAL].to_numpy(dtype=float)
                    - wide[comparator].to_numpy(dtype=float)
                )
                paired = paired[np.isfinite(paired)]
                if paired.size < 2:
                    continue
                mean = float(np.mean(paired))
                half = float(t.ppf(0.975, paired.size - 1)) * float(
                    np.std(paired, ddof=1)
                ) / np.sqrt(paired.size)
                rows.append({
                    "atrial_probability_if_renal_impaired": level,
                    "comparator": comparator,
                    "metric": metric,
                    "difference_definition": "causal SCM minus comparator",
                    "repeat_count": int(paired.size),
                    "mean_difference": mean,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                })
    return pd.DataFrame(rows)
