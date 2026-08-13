"""Posterior metrics shared by hidden-mechanism experiments."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def adjusted_rand_index(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute ARI with the legacy contingency and reduction order."""

    truth_values, truth_inverse = np.unique(truth, return_inverse=True)
    predicted_values, predicted_inverse = np.unique(prediction, return_inverse=True)
    contingency = np.zeros((truth_values.size, predicted_values.size), dtype=np.int64)
    np.add.at(contingency, (truth_inverse, predicted_inverse), 1)

    def choose_two(values: np.ndarray) -> np.ndarray:
        return values * (values - 1) / 2.0

    patient_count = truth.size
    total_pairs = patient_count * (patient_count - 1) / 2.0
    if total_pairs == 0:
        return 1.0
    sum_cells = float(np.sum(choose_two(contingency)))
    sum_truth = float(np.sum(choose_two(np.sum(contingency, axis=1))))
    sum_prediction = float(np.sum(choose_two(np.sum(contingency, axis=0))))
    expected = sum_truth * sum_prediction / total_pairs
    maximum = 0.5 * (sum_truth + sum_prediction)
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0 if np.array_equal(truth, prediction) else 0.0
    return (sum_cells - expected) / denominator


def expected_calibration_error(
    probability: np.ndarray,
    outcome: np.ndarray,
    bin_count: int,
) -> float:
    """Compute the historical equal-width, observed-mass-weighted calibration error."""

    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    bin_index = np.minimum(
        np.digitize(probability, bin_edges[1:-1], right=False),
        bin_count - 1,
    )
    error = 0.0
    for current_bin in range(bin_count):
        members = bin_index == current_bin
        if not np.any(members):
            continue
        error += np.mean(members) * abs(
            np.mean(probability[members]) - np.mean(outcome[members])
        )
    return float(error)


def _safe_rate(values: np.ndarray) -> float:
    """Return a mean, or NaN when the subgroup is empty.

    Sparse nuisance profiles — renal AND heart failure at realistic prevalence
    is roughly 2% of patients — will produce empty cells in some repeats.
    A NaN degrades to a smaller effective repeat count downstream; raising
    would kill the whole run for one thin cell.
    """

    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def evaluate_binary_posterior(
    posterior: np.ndarray,
    truth: np.ndarray,
    renal: np.ndarray,
    calibration_bins: int,
    *,
    atrial: int = 0,
    competing: int = 1,
    subgroups: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Evaluate recovery, the legacy renal/competing subgroup, and named profiles.

    ``subgroups`` maps a name to a boolean mask over patients defined by
    *nuisance profile only* — do not pre-filter by mechanism. False-atrial
    attribution is a competing-mechanism quantity, so the competing filter is
    applied here; accuracy is reported over every patient in the profile.
    Keeping that split inside this function stops callers from silently
    disagreeing about what a subgroup denominator means.
    """

    prediction = np.argmax(posterior, axis=1).astype(np.int8)
    is_atrial = (truth == atrial).astype(float)
    is_competing = truth == competing

    renal_competing = (renal == 1) & is_competing
    if not np.any(renal_competing):
        raise RuntimeError("The renal-impaired competing subgroup was empty.")

    entropy = -np.sum(
        posterior * np.log(np.maximum(posterior, 1e-15)),
        axis=1,
    )

    results: dict[str, float] = {
        "accuracy": float(np.mean(prediction == truth)),
        "adjusted_rand_index": adjusted_rand_index(truth, prediction),
        "false_atrial_renal_competing": float(
            np.mean(prediction[renal_competing] == atrial)
        ),
        "brier_score": float(np.mean((posterior[:, atrial] - is_atrial) ** 2)),
        "expected_calibration_error": expected_calibration_error(
            posterior[:, atrial], is_atrial, calibration_bins
        ),
        "mean_posterior_entropy": float(np.mean(entropy)),
        "predicted_atrial_prevalence": float(np.mean(prediction == atrial)),
        "renal_competing_subgroup_size": int(np.sum(renal_competing)),
    }

    for name, profile in (subgroups or {}).items():
        profile = np.asarray(profile, dtype=bool)
        if profile.shape != truth.shape:
            raise ValueError(
                f"Subgroup {name!r} mask has shape {profile.shape}, expected {truth.shape}."
            )
        profile_competing = profile & is_competing
        results[f"accuracy__{name}"] = _safe_rate(
            prediction[profile] == truth[profile]
        )
        results[f"false_atrial__{name}"] = _safe_rate(
            prediction[profile_competing] == atrial
        )
        results[f"mean_posterior_entropy__{name}"] = _safe_rate(entropy[profile])
        results[f"subgroup_size__{name}"] = int(np.sum(profile))
        results[f"competing_subgroup_size__{name}"] = int(np.sum(profile_competing))

    return results
