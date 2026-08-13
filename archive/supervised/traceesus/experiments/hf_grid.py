"""Renal x heart-failure grid for endotype discovery and model comparison.

One runner produces both presentation datasets:

- ``run_latent_grid``     — unsupervised discovery (5 models per repeat)
- ``run_supervised_grid`` — supervised comparison (5 models per repeat)

Every cell of the 4 x 4 grid uses paired cohorts and per-model fit streams, so
differences between models are attributable to the model, never the draw.
Both runners always report the four prespecified nuisance profiles
(uncomplicated / renal-only / heart-failure-only / redundant); false-atrial
attribution within a profile is computed over its competing-mechanism patients
by the shared metric function.

Seed roots are salted (+737_270 latent, +747_270 supervised) and therefore
disjoint from every proposal-locked ledger: this grid can be rerun freely
without touching a cited artifact.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from traceesus.core.metrics import evaluate_binary_posterior
from traceesus.core.model import Model
from traceesus.core.runner import ordered_map, spawned_uint64_seeds
from traceesus.experiments.endotype_discovery import kernel
from traceesus.experiments.endotype_discovery import multi_nuisance as mn
from traceesus.experiments.model_comparison import kernel as comparison_kernel
from traceesus.models import AdjustedLatentClassModel, AssociativeLatentClassModel
from traceesus.simulators.two_mechanism import TwoMechanismSimulator

RENAL_LEVELS = (0.0, 0.5, 1.0, 1.5)
HF_LEVELS = (0.0, 0.5, 1.0, 1.5)

# Facets for the patient-level scatter: none / moderate / strong.  The two
# endpoints are the locked anchors; the midpoint is spaced evenly between them.
SCATTER_RENAL_LEVELS = (0.0, 0.75, 1.5)

BIOMARKER_COLUMNS = ("nt_probnp", "ptfv1", "competing_vascular")

SUPERVISED_LOGISTIC_BIOMARKERS = "Logistic regression (biomarkers only)"
SUPERVISED_LOGISTIC_RENAL = "Logistic regression (+ kidney status)"
SUPERVISED_LOGISTIC_BOTH = "Logistic regression (+ kidney + heart failure)"
SUPERVISED_SCM_TWO_PATH = "Supervised two-path SCM (posterior)"
SUPERVISED_SCM_COUNTERFACTUAL = "Supervised two-path SCM (counterfactual query)"


def _latent_models() -> tuple[Model, ...]:
    return (
        AssociativeLatentClassModel(),
        AdjustedLatentClassModel(),
        mn.TwoNuisanceAdjustedLCM(),
        mn.TwoNuisanceCausalSCM(),
        mn.TwoNuisanceCounterfactualSCM(),
    )


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
    for model_index, model in enumerate(models):
        fitted = model.fit(
            training.observed,
            np.random.default_rng(sequences[2 + model_index]),
            config.fitting,
        )
        diagnostics = fitted.fit_diagnostics()
        if diagnostics is not None and not diagnostics.converged:
            fitted = model.fit(
                training.observed,
                np.random.default_rng(sequences[2 + model_index]),
                retry,
            )
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
    root = np.random.SeedSequence(config.master_seed + 737_270)
    cells = [(r, h) for r in renal_levels for h in hf_levels]
    ledgers = [spawned_uint64_seeds(child, repeats) for child in root.spawn(len(cells))]
    for (renal_sd, hf_sd), seeds in zip(cells, ledgers, strict=True):
        for repeat_index, seed in enumerate(seeds):
            tasks.append((repeat_index, seed, renal_sd, hf_sd, config))
    results = ordered_map(_run_latent_cell, tasks, workers)
    return pd.DataFrame([row for rows in results for row in rows])


# --------------------------------------------------------------------------
# Supervised (model comparison) grid
# --------------------------------------------------------------------------

def _supervised_two_path_fit(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    heart_failure: np.ndarray,
    mechanism: np.ndarray,
    config: comparison_kernel.ComparisonConfig,
) -> mn.MultiNuisanceLatentFit:
    """Supervised structural fit expressed as a MultiNuisanceLatentFit.

    Least-squares structural equations on the design
    ``[atrial, competing, renal, heart_failure]`` — the direct two-nuisance
    generalization of the locked supervised SCM.  Returning the shared fit
    dataclass lets the supervised posterior and counterfactual queries reuse
    the exact latent-side arithmetic, so the two experiments answer the query
    question with one implementation.
    """

    design = np.column_stack(
        (
            (mechanism == kernel.Mechanism.ATRIAL).astype(float),
            (mechanism == kernel.Mechanism.COMPETING).astype(float),
            renal.astype(float),
            heart_failure.astype(float),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, biomarkers, rcond=None)
    residual = biomarkers - design @ coefficients
    degrees_of_freedom = biomarkers.shape[0] - design.shape[1]
    noise_sd = np.sqrt(np.sum(residual**2, axis=0) / degrees_of_freedom)
    noise_sd = np.maximum(noise_sd, config.variance_floor)

    smoothing = config.scm_prior_smoothing
    class_probability_by_renal = np.zeros((2, 2), dtype=float)
    for renal_value in (0, 1):
        stratum = renal == renal_value
        count = int(np.sum(stratum))
        atrial = int(np.sum(stratum & (mechanism == kernel.Mechanism.ATRIAL)))
        p_atrial = (atrial + smoothing) / (count + 2.0 * smoothing)
        class_probability_by_renal[renal_value] = (p_atrial, 1.0 - p_atrial)

    return mn.MultiNuisanceLatentFit(
        class_probability_by_renal=class_probability_by_renal,
        class_means_at_reference=coefficients[:2, :],
        nuisance_effects=coefficients[2:, :],
        nuisance_path_mask=np.ones((2, biomarkers.shape[1]), dtype=bool),
        biomarker_variance=noise_sd**2,
        log_likelihood=float("nan"),
        converged=True,
        iterations=1,
        best_start=0,
        effective_class_fraction=np.array([
            float(np.mean(mechanism == kernel.Mechanism.ATRIAL)),
            float(np.mean(mechanism == kernel.Mechanism.COMPETING)),
        ]),
        anchor_margin=float("nan"),
    )


def _run_supervised_cell(task) -> list[dict[str, object]]:
    repeat_index, seed, renal_sd, hf_sd, sim_config, cmp_config, train_n, test_n = task
    sequences = np.random.SeedSequence(seed).spawn(2)
    training = kernel.simulate_two_mechanism_cohort(
        np.random.default_rng(sequences[0]), train_n, renal_sd, sim_config,
        heart_failure_effect_sd=hf_sd,
    )
    test = kernel.simulate_two_mechanism_cohort(
        np.random.default_rng(sequences[1]), test_n, renal_sd, sim_config,
        heart_failure_effect_sd=hf_sd,
    )
    atrial_label = (training.true_mechanism == kernel.Mechanism.ATRIAL).astype(float)
    nuisances_test = np.column_stack(
        (test.renal_dysfunction.astype(float), test.heart_failure.astype(float))
    )

    def logistic_posterior(train_features, test_features):
        model = comparison_kernel.fit_logistic_regression(
            train_features, atrial_label, cmp_config
        )
        p = comparison_kernel.logistic_atrial_probability(model, test_features)
        return np.column_stack((p, 1.0 - p))

    train_b, test_b = training.biomarkers, test.biomarkers
    train_r = training.renal_dysfunction.astype(float)[:, None]
    test_r = test.renal_dysfunction.astype(float)[:, None]
    train_h = training.heart_failure.astype(float)[:, None]
    test_h = test.heart_failure.astype(float)[:, None]

    scm_fit = _supervised_two_path_fit(
        train_b,
        training.renal_dysfunction,
        training.heart_failure,
        training.true_mechanism,
        cmp_config,
    )
    scm_posterior = mn.multi_nuisance_posterior(
        scm_fit, test_b, test.renal_dysfunction, nuisances_test
    )
    scm_scores = mn.multi_nuisance_counterfactual_scores(
        scm_fit, test_b, test.renal_dysfunction, nuisances_test
    )
    posteriors = {
        SUPERVISED_LOGISTIC_BIOMARKERS: logistic_posterior(train_b, test_b),
        SUPERVISED_LOGISTIC_RENAL: logistic_posterior(
            np.hstack((train_b, train_r)), np.hstack((test_b, test_r))
        ),
        SUPERVISED_LOGISTIC_BOTH: logistic_posterior(
            np.hstack((train_b, train_r, train_h)), np.hstack((test_b, test_r, test_h))
        ),
        SUPERVISED_SCM_TWO_PATH: scm_posterior,
        SUPERVISED_SCM_COUNTERFACTUAL: mn._row_normalize(scm_scores["combined"]),
    }

    class _Observed:
        """Minimal covariate view matching the metric function's interface."""

        def covariate(self, name: str) -> np.ndarray:
            if name == "renal_dysfunction":
                return test.renal_dysfunction
            if name == "heart_failure":
                return test.heart_failure
            raise KeyError(name)

    observed = _Observed()
    rows: list[dict[str, object]] = []
    for method, posterior in posteriors.items():
        row: dict[str, object] = {
            "repeat": repeat_index,
            "renal_effect_sd": renal_sd,
            "heart_failure_effect_sd": hf_sd,
            "method": method,
        }
        row.update(
            _evaluate(posterior, test.true_mechanism, observed, 10)
        )
        rows.append(row)
    return rows


def run_supervised_grid(
    sim_config,
    cmp_config: comparison_kernel.ComparisonConfig,
    *,
    master_seed: int,
    renal_levels: tuple[float, ...] = RENAL_LEVELS,
    hf_levels: tuple[float, ...] = HF_LEVELS,
    repeats: int = 100,
    workers: int = 4,
    training_patients: int = 3_000,
    test_patients: int = 1_000,
) -> pd.DataFrame:
    """Supervised comparison across the full renal x heart-failure grid."""

    tasks = []
    root = np.random.SeedSequence(master_seed + 747_270)
    cells = [(r, h) for r in renal_levels for h in hf_levels]
    ledgers = [spawned_uint64_seeds(child, repeats) for child in root.spawn(len(cells))]
    for (renal_sd, hf_sd), seeds in zip(cells, ledgers, strict=True):
        for repeat_index, seed in enumerate(seeds):
            tasks.append(
                (
                    repeat_index,
                    seed,
                    renal_sd,
                    hf_sd,
                    sim_config,
                    cmp_config,
                    training_patients,
                    test_patients,
                )
            )
    results = ordered_map(_run_supervised_cell, tasks, workers)
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

    Its own salted root (+767_270); shares no seed with any cited artifact.
    """

    root = np.random.SeedSequence(master_seed + 767_270)
    tasks = []
    for renal_sd, child in zip(renal_levels, root.spawn(len(renal_levels)), strict=True):
        for repeat_index, seed in enumerate(spawned_uint64_seeds(child, repeats)):
            tasks.append((
                repeat_index, seed, renal_sd, heart_failure_effect_sd,
                sim_config, fitting_config,
            ))
    return pd.DataFrame(ordered_map(_run_drift_cell, tasks, workers))


def _run_drift_cell(task) -> dict[str, object]:
    repeat_index, seed, renal_sd, hf_sd, sim_config, fitting_config = task
    draw, fit_stream = np.random.SeedSequence(seed).spawn(2)
    simulator = TwoMechanismSimulator(sim_config, renal_sd, hf_sd)
    cohort = simulator.simulate(
        np.random.default_rng(draw), sim_config.training_patients
    )
    observed = cohort.observed

    fitted = AssociativeLatentClassModel().fit(
        observed, np.random.default_rng(fit_stream), fitting_config
    )
    assignment = np.argmax(fitted.posterior(observed), axis=1)
    marker = int(kernel.Biomarker.NT_PROBNP)
    biomarkers = observed.biomarkers
    high = int(np.argmax([
        biomarkers[assignment == component, marker].mean()
        if np.any(assignment == component) else -np.inf
        for component in (0, 1)
    ]))
    called_atrial = assignment == high

    atrial = cohort.truth.mechanism == int(kernel.Mechanism.ATRIAL)
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
    from its own salted root (+757_270) and must never enter a summarized
    estimate.  Truth is included because the figure's whole purpose is to show
    the labels alongside the discovered split — something no real cohort can do.

    ``discovered_class`` is the hard assignment of an unadjusted associative
    latent class model fitted to that cohort.  Latent labels are arbitrary, so
    they are oriented mechanically: class A is always the component with the
    higher mean NT-proBNP.  The rule never consults the truth, so it cannot
    flatter or penalise the model.
    """

    root = np.random.SeedSequence(master_seed + 757_270)
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
