"""Define plain-English meanings for every metric emitted by TRACE-ESUS tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_BASE_METRICS = (
    ("accuracy", "Fraction whose top-ranked class matches the simulated mechanism.", "All evaluated patients.", "proportion"),
    ("accuracy_any_missing", "Mechanism-ranking accuracy among patients missing at least one biomarker.", "Evaluated patients with any missing biomarker.", "proportion"),
    ("accuracy_complete_case", "Mechanism-ranking accuracy among patients with all biomarkers observed.", "Complete-case evaluated patients.", "proportion"),
    ("accuracy_electrical_missing", "Mechanism-ranking accuracy when the atrial electrical marker is missing.", "Evaluated patients missing the electrical marker.", "proportion"),
    ("adjusted_rand_index", "Chance-adjusted agreement between assigned and simulated mechanism partitions.", "All unordered pairs of evaluated patients.", "index from -1 to 1"),
    ("all_biomarkers_missing_rate", "Fraction of patients with every biomarker missing.", "All evaluated patients.", "proportion"),
    ("biomarker_missing_fraction", "Fraction of individual biomarker measurements that are missing.", "All patient-by-biomarker measurements.", "proportion"),
    ("brier_score", "Mean squared error of the atrial posterior probability.", "All evaluated patients.", "squared probability"),
    ("expected_calibration_error", "Weighted mean absolute gap between predicted and observed atrial frequency across calibration bins.", "All evaluated patients, weighted by bin size.", "probability"),
    ("false_atrial_confounded_competing", "Fraction called atrial among renal-impaired patients with a competing mechanism.", "Renal-impaired competing-mechanism patients.", "proportion"),
    ("false_atrial_renal_competing", "Fraction called atrial among renal-impaired patients with a competing mechanism.", "Renal-impaired competing-mechanism patients.", "proportion"),
    ("mean_posterior_entropy", "Mean uncertainty of each two-class posterior distribution.", "All evaluated patients.", "nats"),
    ("predicted_atrial_prevalence", "Mean posterior probability assigned to the atrial mechanism.", "All evaluated patients.", "proportion"),
    ("true_mechanism_accuracy", "Fraction whose top-ranked class matches the simulated mechanism.", "All evaluated patients.", "proportion"),
    ("agreement_with_kidney_status", "Label-invariant agreement between the discovered class and renal status.", "All patients in the identity-drift cohort.", "proportion"),
    ("agreement_with_mechanism", "Label-invariant agreement between the discovered class and simulated mechanism.", "All patients in the identity-drift cohort.", "proportion"),
    ("composition__atrial_impaired_kidneys", "Share of the atrial-like discovered class with atrial mechanism and impaired kidneys.", "Patients assigned to the atrial-like class.", "proportion"),
    ("composition__atrial_normal_kidneys", "Share of the atrial-like discovered class with atrial mechanism and normal kidneys.", "Patients assigned to the atrial-like class.", "proportion"),
    ("composition__competing_impaired_kidneys", "Share of the atrial-like discovered class with competing mechanism and impaired kidneys.", "Patients assigned to the atrial-like class.", "proportion"),
    ("composition__competing_normal_kidneys", "Share of the atrial-like discovered class with competing mechanism and normal kidneys.", "Patients assigned to the atrial-like class.", "proportion"),
    ("fitted_prior_atrial_given_renal", "Fitted probability of atrial mechanism among renal-impaired patients.", "Renal-impaired patients represented by the fitted model.", "probability"),
)

_SUBGROUPS = (
    ("uncomplicated", "patients with normal kidneys and no heart failure"),
    ("renal_only", "patients with renal impairment and no heart failure"),
    ("heart_failure_only", "patients with heart failure and normal kidneys"),
    ("redundant", "patients with both renal impairment and heart failure"),
)

_SUBGROUP_METRICS = (
    ("accuracy", "Mechanism-ranking accuracy among {group}.", "Patients in the subgroup.", "proportion"),
    ("competing_subgroup_size", "Number of competing-mechanism {group}.", "One simulated cohort.", "patients"),
    ("false_atrial", "Fraction called atrial among competing-mechanism {group}.", "Competing-mechanism patients in the subgroup.", "proportion"),
    ("mean_posterior_entropy", "Mean posterior uncertainty among {group}.", "Patients in the subgroup.", "nats"),
    ("subgroup_size", "Number of {group}.", "One simulated cohort.", "patients"),
)


def data_dictionary() -> pd.DataFrame:
    """Return one axis-label-ready row per emitted metric name."""

    rows = list(_BASE_METRICS)
    for subgroup, group_description in _SUBGROUPS:
        for metric, definition, denominator, units in _SUBGROUP_METRICS:
            rows.append(
                (
                    f"{metric}__{subgroup}",
                    definition.format(group=group_description),
                    denominator,
                    units,
                )
            )
    return pd.DataFrame(rows, columns=("metric", "definition", "denominator", "units"))


def write_data_dictionary(path: Path) -> None:
    """Write the canonical metric dictionary as a stable CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data_dictionary().sort_values("metric").to_csv(path, index=False)


__all__ = ["data_dictionary", "write_data_dictionary"]
