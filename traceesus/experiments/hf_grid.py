"""Renal x heart-failure grid for unsupervised endotype discovery.

The runner produces the presentation dataset for five latent models per repeat.

Every cell of the 4 x 4 grid uses paired cohorts and per-model fit streams, so
differences between models are attributable to the model, never the draw.
The runner always reports the four prespecified nuisance profiles
(uncomplicated / renal-only / heart-failure-only / redundant); false-atrial
attribution within a profile is computed over its competing-mechanism patients
by the shared metric function.

The named HF-grid seed root is disjoint from every
proposal-locked ledger: this grid can be rerun freely without touching a cited
artifact.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from traceesus.core.metrics import evaluate_binary_posterior
from traceesus.core.model import Model
from traceesus.core.runner import ordered_map, spawned_uint64_seeds
from traceesus.core.seeds import (
    COHORT_SCATTER_SEED_OFFSET,
    HF_GRID_SEED_OFFSET,
    IDENTITY_DRIFT_SEED_OFFSET,
)
from traceesus.experiments.endotype_discovery import kernel
from traceesus.models import (
    TwoNuisanceCausalSCM,
    TwoNuisanceCounterfactualSCM,
)
from traceesus.models.multi_nuisance import counterfactual_view
from traceesus.registry import FULL_LADDER
from traceesus.simulators.two_mechanism import TwoMechanismSimulator

RENAL_LEVELS = (0.0, 0.5, 1.0, 1.5)
HF_LEVELS = (0.0, 0.5, 1.0, 1.5)

# Facets for the patient-level scatter: none / moderate / strong.  The two
# endpoints are the locked anchors; the midpoint is spaced evenly between them.
SCATTER_RENAL_LEVELS = (0.0, 0.75, 1.5)

BIOMARKER_COLUMNS = ("nt_probnp", "ptfv1", "competing_vascular")

def _latent_models() -> tuple[Model, ...]:
    """Return the fitted portion of the single declared R1--R6 ladder."""

    return FULL_LADDER.fitted_models


def _subgroups(observed) -> dict[str, np.ndarray]:
    renal = observed.covariate("renal_dysfunction") == 1
    heart_failure = observed.covariate("heart_failure") == 1
    return {
        "uncomplicated": ~renal & ~heart_failure,
        "renal_only": renal & ~heart_failure,
        "heart_failure_only": ~renal & heart_failure,
        "redundant": renal & heart_failure,
    }


def _evaluate(
    posterior: np.ndarray,
    truth: np.ndarray,
    observed,
    calibration_bins: int,
) -> dict[str, float]:
    return evaluate_binary_posterior(
        posterior,
        truth,
        observed.covariate("renal_dysfunction"),
        calibration_bins,
        atrial=int(kernel.Mechanism.ATRIAL),
        competing=int(kernel.Mechanism.COMPETING),
        subgroups=_subgroups(observed),
    )


# --------------------------------------------------------------------------
# Latent (endotype discovery) grid
# --------------------------------------------------------------------------

def _run_latent_cell(task) -> list[dict[str, object]]:
    repeat_index, seed, renal_sd, hf_sd, config = task
    models = _latent_models()
    sequences = np.random.SeedSequence(seed).spawn(2 + len(models))
    simulator = TwoMechanismSimulator(config.simulation, renal_sd, hf_sd)
    training = simulator.simulate(
        np.random.default_rng(sequences[0]), config.simulation.training_patients
    )
    test = simulator.simulate(
        np.random.default_rng(sequences[1]), config.simulation.test_patients
    )
    retry = replace(
        config.fitting,
        random_starts=max(config.fitting.random_starts, 8),
        maximum_em_iterations=max(config.fitting.maximum_em_iterations, 1_200),
    )
    rows: list[dict[str, object]] = []
    two_path_fit = None
    for model_index, model in enumerate(models):
        if isinstance(model, TwoNuisanceCounterfactualSCM):
            if two_path_fit is None:
                raise RuntimeError("R5 must follow R4 in the model ladder.")
            fitted = counterfactual_view(two_path_fit)
        else:
            fitted = model.fit(
                training.observed,
                np.random.default_rng(sequences[2 + model_index]),
                config.fitting,
            )
        diagnostics = fitted.fit_diagnostics()
        if (
            diagnostics is not None
            and not diagnostics.converged
            and not isinstance(model, TwoNuisanceCounterfactualSCM)
        ):
            fitted = model.fit(
                training.observed,
                np.random.default_rng(sequences[2 + model_index]),
                retry,
            )
        if isinstance(model, TwoNuisanceCausalSCM):
            two_path_fit = fitted
        row: dict[str, object] = {
            "repeat": repeat_index,
            "renal_effect_sd": renal_sd,
            "heart_failure_effect_sd": hf_sd,
            "method": model.name,
        }
        row.update(
            _evaluate(
                fitted.posterior(test.observed),
                test.truth.mechanism,
                test.observed,
                config.fitting.calibration_bins,
            )
        )
        rows.append(row)
    return rows


def run_latent_grid(
    config: kernel.ExperimentConfig,
    *,
    renal_levels: tuple[float, ...] = RENAL_LEVELS,
    hf_levels: tuple[float, ...] = HF_LEVELS,
    repeats: int = 100,
    workers: int = 4,
) -> pd.DataFrame:
    """Unsupervised recovery across the full renal x heart-failure grid."""

    config.validate()
    tasks = []
    root = np.random.SeedSequence(config.master_seed + HF_GRID_SEED_OFFSET)
    cells = [(r, h) for r in renal_levels for h in hf_levels]
    ledgers = [spawned_uint64_seeds(child, repeats) for child in root.spawn(len(cells))]
    for (renal_sd, hf_sd), seeds in zip(cells, ledgers, strict=True):
        for repeat_index, seed in enumerate(seeds):
            tasks.append((repeat_index, seed, renal_sd, hf_sd, config))
    results = ordered_map(_run_latent_cell, tasks, workers)
    return pd.DataFrame([row for rows in results for row in rows])


# --------------------------------------------------------------------------
# Patient-level cohort sample (figure input only)
# --------------------------------------------------------------------------

DRIFT_RENAL_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

# The four true cells a discovered class can draw its members from.
TRUE_CELLS = (
    ("atrial_normal_kidneys", "atrial", 0),
    ("atrial_impaired_kidneys", "atrial", 1),
    ("competing_impaired_kidneys", "competing", 1),
    ("competing_normal_kidneys", "competing", 0),
)


def _agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Agreement between two binary labellings, invariant to label switching."""

    match = float(np.mean(left == right))
    return max(match, 1.0 - match)


def identity_drift_sweep(
    sim_config,
    fitting_config,
    *,
    master_seed: int,
    renal_levels: tuple[float, ...] = DRIFT_RENAL_LEVELS,
    heart_failure_effect_sd: float = 1.5,
    repeats: int = 40,
    workers: int = 4,
) -> pd.DataFrame:
    """Track what the pooled model's latent class comes to *mean*.

    A latent class has no intrinsic identity — it is whatever the likelihood
    rewards.  This sweep asks, at each renal strength, whether the unadjusted
    associative model's split corresponds to the mechanism contrast or to
    kidney status, and reports the true composition of the class the model
    would read as "atrial-like" (the one with higher mean NT-proBNP).

    Its own named root shares no seed with any cited artifact.
    """

    root = np.random.SeedSequence(master_seed + IDENTITY_DRIFT_SEED_OFFSET)
    tasks = []
    for renal_sd, child in zip(renal_levels, root.spawn(len(renal_levels)), strict=True):
        for repeat_index, seed in enumerate(spawned_uint64_seeds(child, repeats)):
            tasks.append((
                repeat_index, seed, renal_sd, heart_failure_effect_sd,
                sim_config, fitting_config,
            ))
    return pd.DataFrame(ordered_map(_run_drift_cell, tasks, workers))


def _run_drift_cell(task) -> dict[str, object]:
    """Fit on training and score held-out; children are train, fit, evaluation."""

    repeat_index, seed, renal_sd, hf_sd, sim_config, fitting_config = task
    training_draw, fit_stream, evaluation_draw = np.random.SeedSequence(seed).spawn(3)
    simulator = TwoMechanismSimulator(sim_config, renal_sd, hf_sd)
    training = simulator.simulate(
        np.random.default_rng(training_draw), sim_config.training_patients
    )
    evaluation = simulator.simulate(
        np.random.default_rng(evaluation_draw), sim_config.test_patients
    )

    fitted = AssociativeLatentClassModel().fit(
        training.observed, np.random.default_rng(fit_stream), fitting_config
    )
    training_assignment = np.argmax(fitted.posterior(training.observed), axis=1)
    marker = int(kernel.Biomarker.NT_PROBNP)
    high = int(np.argmax([
        training.observed.biomarkers[training_assignment == component, marker].mean()
        if np.any(training_assignment == component) else -np.inf
        for component in (0, 1)
    ]))
    observed = evaluation.observed
    assignment = np.argmax(fitted.posterior(observed), axis=1)
    called_atrial = assignment == high

    atrial = evaluation.truth.mechanism == int(kernel.Mechanism.ATRIAL)
    renal = observed.covariate("renal_dysfunction") == 1
    row: dict[str, object] = {
        "repeat": repeat_index,
        "renal_effect_sd": renal_sd,
        "heart_failure_effect_sd": hf_sd,
        "agreement_with_mechanism": _agreement(called_atrial, atrial),
        "agreement_with_kidney_status": _agreement(called_atrial, renal),
        "called_atrial_size": int(called_atrial.sum()),
    }
    size = max(int(called_atrial.sum()), 1)
    for name, mechanism, impaired in TRUE_CELLS:
        member = (atrial if mechanism == "atrial" else ~atrial) & (
            renal if impaired else ~renal
        )
        row[f"composition__{name}"] = float(np.sum(called_atrial & member) / size)
    return row


def cohort_scatter_sample(
    sim_config,
    fitting_config,
    *,
    master_seed: int,
    renal_levels: tuple[float, ...] = SCATTER_RENAL_LEVELS,
    heart_failure_effect_sd: float = 1.5,
    patients: int = 4_000,
) -> pd.DataFrame:
    """Emit one labelled cohort per renal level, plus what an LCM finds in it.

    This is *figure input*, deliberately outside every repeat loop: it draws
    from its own named root and must never enter a summarized
    estimate.  Truth is included because the figure's whole purpose is to show
    the labels alongside the discovered split — something no real cohort can do.

    ``discovered_class`` is the hard assignment of an unadjusted associative
    latent class model fitted to that cohort.  Latent labels are arbitrary, so
    they are oriented mechanically: class A is always the component with the
    higher mean NT-proBNP.  The rule never consults the truth, so it cannot
    flatter or penalise the model.
    """

    root = np.random.SeedSequence(master_seed + COHORT_SCATTER_SEED_OFFSET)
    frames: list[pd.DataFrame] = []
    for renal_sd, child in zip(renal_levels, root.spawn(len(renal_levels)), strict=True):
        draw, fit_stream = child.spawn(2)
        simulator = TwoMechanismSimulator(sim_config, renal_sd, heart_failure_effect_sd)
        cohort = simulator.simulate(np.random.default_rng(draw), patients)
        observed, biomarkers = cohort.observed, cohort.observed.biomarkers

        fitted = AssociativeLatentClassModel().fit(
            observed, np.random.default_rng(fit_stream), fitting_config
        )
        assignment = np.argmax(fitted.posterior(observed), axis=1)
        marker = int(kernel.Biomarker.NT_PROBNP)
        high_component = int(np.argmax([
            biomarkers[assignment == component, marker].mean()
            if np.any(assignment == component) else -np.inf
            for component in (0, 1)
        ]))

        frame = pd.DataFrame(biomarkers, columns=list(BIOMARKER_COLUMNS))
        frame.insert(0, "renal_effect_sd", renal_sd)
        frame["heart_failure_effect_sd"] = heart_failure_effect_sd
        frame["renal_dysfunction"] = observed.covariate("renal_dysfunction")
        frame["heart_failure"] = observed.covariate("heart_failure")
        frame["true_mechanism"] = np.where(
            cohort.truth.mechanism == int(kernel.Mechanism.ATRIAL),
            "atrial",
            "competing",
        )
        frame["discovered_class"] = np.where(
            assignment == high_component, "class A", "class B"
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
