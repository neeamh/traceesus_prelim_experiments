"""Build the reported-rate and paired-contrast delta audit for EM unification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = ROOT / "outputs_locked"
V2_ROOT = ROOT / "outputs_locked_v2"
REPORT_PATH = ROOT / "reports" / "em_unification_delta.csv"

RATE_METRICS = (
    "accuracy",
    "accuracy_any_missing",
    "accuracy_complete_case",
    "accuracy_electrical_missing",
    "all_biomarkers_missing_rate",
    "biomarker_missing_fraction",
    "false_atrial_confounded_competing",
    "false_atrial_renal_competing",
    "false_k2_rate",
    "k2_convergence_rate",
    "predicted_atrial_prevalence",
    "true_mechanism_accuracy",
)

NATIVE_REPORTED_DIGITS = {
    "adjusted_rand_index": 3,
    "brier_score": 4,
    "expected_calibration_error": 4,
    "median_bic_k2_minus_k1": 2,
    "median_delta_bic_k2_minus_k1": 2,
}


@dataclass(frozen=True)
class TableSpec:
    """Describe one manuscript-rate or percentage-point source table."""

    file: str
    level_columns: tuple[str, ...]
    method_columns: tuple[str, ...]
    metric_column: str | None
    value_column: str
    ci_columns: tuple[str, str] | None = None
    fixed_metric: str | None = None
    contrasts: bool = False


TABLES = (
    TableSpec("outputs_latent_endotyping/recovery_summary.csv", ("renal_effect_sd",), ("method",), "metric", "mean"),
    TableSpec("outputs_latent_endotyping/paired_contrasts.csv", ("renal_effect_sd",), ("difference_definition", "comparator"), "metric", "mean_difference", ("ci95_low", "ci95_high"), contrasts=True),
    TableSpec("outputs_latent_endotyping/k1_null_summary.csv", (), ("method",), None, "false_k2_rate", ("wilson_ci95_low", "wilson_ci95_high"), "false_k2_rate"),
    TableSpec("outputs_latent_endotyping/k1_null_summary.csv", (), ("method",), None, "median_delta_bic_k2_minus_k1", fixed_metric="median_delta_bic_k2_minus_k1"),
    TableSpec("outputs_latent_endotyping/k1_null_summary.csv", (), ("method",), None, "k2_convergence_rate", fixed_metric="k2_convergence_rate"),
    TableSpec("outputs_transportability/transport_summary.csv", ("shift",), ("method",), "metric", "mean"),
    TableSpec("outputs_transportability/paired_transport_contrasts.csv", ("shift",), ("difference_definition", "comparator"), "metric", "mean_difference", ("ci95_low", "ci95_high"), contrasts=True),
    TableSpec("outputs_transportability/transport_degradation.csv", (), ("method",), None, "mean_difference", ("ci95_low", "ci95_high"), "accuracy", True),
    TableSpec("outputs_transportability/ablations/ablation_summary.csv", ("shift",), ("method",), "metric", "mean"),
    TableSpec("outputs_transportability/ablations/paired_ablation_contrasts.csv", ("shift",), ("difference_definition", "comparator"), "metric", "mean_difference", ("ci95_low", "ci95_high"), contrasts=True),
    TableSpec("outputs_transportability/ablations/ablation_accuracy_changes.csv", ("shift",), ("method",), None, "mean_difference", ("ci95_low", "ci95_high"), "accuracy", True),
    TableSpec("outputs_transportability/exact_no_shift_control/summary.csv", ("shift",), ("method",), "metric", "mean"),
    TableSpec("outputs_transportability/exact_no_shift_control/paired_contrasts.csv", ("shift",), ("difference_definition", "comparator"), "metric", "mean_difference", ("ci95_low", "ci95_high"), contrasts=True),
    TableSpec("outputs/main_simulation_summary.csv", ("renal_effect_sd",), ("method",), "metric", "mean"),
    TableSpec("outputs/paired_method_differences.csv", ("renal_effect_sd",), ("contrast",), "metric", "mean_difference", ("ci_low", "ci_high"), contrasts=True),
    TableSpec("outputs/k1_null_summary.csv", ("truth_k",), ("comparison",), None, "false_k2_rate", ("wilson_ci_low", "wilson_ci_high"), "false_k2_rate"),
    TableSpec("outputs/k1_null_summary.csv", ("truth_k",), ("comparison",), None, "median_bic_k2_minus_k1", fixed_metric="median_bic_k2_minus_k1"),
    TableSpec("outputs/k1_null_summary.csv", ("truth_k",), ("comparison",), None, "k2_convergence_rate", fixed_metric="k2_convergence_rate"),
)


def _label(frame: pd.DataFrame, columns: tuple[str, ...], empty: str) -> pd.Series:
    """Serialize stable key columns without depending on row order."""

    if not columns:
        return pd.Series(np.repeat(empty, len(frame)), index=frame.index)
    parts = [column + "=" + frame[column].astype(str) for column in columns]
    result = parts[0]
    for part in parts[1:]:
        result = result + " | " + part
    return result


def _tidy(path: Path, spec: TableSpec) -> pd.DataFrame:
    """Extract uniquely keyed reported rates from one output table."""

    frame = pd.read_csv(path)
    metric = (
        frame[spec.metric_column].astype(str)
        if spec.metric_column
        else pd.Series(np.repeat(spec.fixed_metric, len(frame)), index=frame.index)
    )
    result = pd.DataFrame(
        {
            "level": _label(frame, spec.level_columns, "overall"),
            "method": _label(frame, spec.method_columns, "all methods"),
            "metric": metric,
            "value": frame[spec.value_column].astype(float),
        }
    )
    if spec.ci_columns:
        result["ci_low"] = frame[spec.ci_columns[0]].astype(float)
        result["ci_high"] = frame[spec.ci_columns[1]].astype(float)
    if result.duplicated(("level", "method", "metric")).any():
        raise ValueError(f"Non-unique report key in {spec.file}.")
    return result


def _paired(
    spec: TableSpec,
    before_root: Path,
    after_root: Path,
    before_label: str,
    after_label: str,
) -> pd.DataFrame:
    """Join baseline versions by scientific keys and reject dropped estimands."""

    before = _tidy(before_root / spec.file, spec)
    after = _tidy(after_root / spec.file, spec)
    paired = before.merge(
        after,
        on=("level", "method", "metric"),
        how="outer",
        validate="one_to_one",
        suffixes=(f"_{before_label}", f"_{after_label}"),
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError(
            f"{before_label}/{after_label} estimand mismatch in {spec.file}."
        )
    return paired.drop(columns="_merge")


def _reported_columns(
    paired: pd.DataFrame,
    contrasts: bool,
    before_label: str,
    after_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply documented manuscript precision only to quantities actually reported."""

    before = np.full(len(paired), np.nan)
    after = np.full(len(paired), np.nan)
    eligible = np.zeros(len(paired), dtype=bool)
    before_column = f"value_{before_label}"
    after_column = f"value_{after_label}"
    for row, metric in enumerate(paired["metric"]):
        if metric in RATE_METRICS:
            digits = 2 if contrasts else 1
            before[row] = np.round(100.0 * paired.iloc[row][before_column], digits)
            after[row] = np.round(100.0 * paired.iloc[row][after_column], digits)
            eligible[row] = True
        elif metric in NATIVE_REPORTED_DIGITS:
            digits = NATIVE_REPORTED_DIGITS[metric]
            before[row] = np.round(paired.iloc[row][before_column], digits)
            after[row] = np.round(paired.iloc[row][after_column], digits)
            eligible[row] = True
    return before, after, eligible & (before != after)


def build_report(
    before_root: Path = V1_ROOT,
    after_root: Path = V2_ROOT,
    before_label: str = "v1",
    after_label: str = "v2",
) -> pd.DataFrame:
    """Build one row per reported percentage or percentage-point estimand."""

    rows: list[pd.DataFrame] = []
    for spec in TABLES:
        paired = _paired(
            spec, before_root, after_root, before_label, after_label
        )
        paired.insert(0, "file", spec.file)
        before_value = f"value_{before_label}"
        after_value = f"value_{after_label}"
        paired["absolute_difference"] = np.abs(
            paired[after_value] - paired[before_value]
        )
        before, after, changed = _reported_columns(
            paired, spec.contrasts, before_label, after_label
        )
        before_reported = f"reported_{before_label}"
        after_reported = f"reported_{after_label}"
        paired[before_reported] = before
        paired[after_reported] = after
        paired["changes_at_reported_precision"] = changed
        rows.append(paired[[
            "file", "level", "method", "metric", before_value, after_value,
            "absolute_difference", before_reported, after_reported,
            "changes_at_reported_precision",
        ]])
    return pd.concat(rows, ignore_index=True).sort_values(
        ["file", "level", "method", "metric"], ignore_index=True
    )


def confidence_interval_crossings(
    before_root: Path = V1_ROOT,
    after_root: Path = V2_ROOT,
    before_label: str = "v1",
    after_label: str = "v2",
) -> pd.DataFrame:
    """Return contrasts whose 95% interval changed zero-inclusion status."""

    rows: list[pd.DataFrame] = []
    for spec in TABLES:
        if not spec.ci_columns:
            continue
        paired = _paired(
            spec, before_root, after_root, before_label, after_label
        )
        before = (paired[f"ci_low_{before_label}"] <= 0.0) & (
            paired[f"ci_high_{before_label}"] >= 0.0
        )
        after = (paired[f"ci_low_{after_label}"] <= 0.0) & (
            paired[f"ci_high_{after_label}"] >= 0.0
        )
        changed = paired.loc[before != after].copy()
        if not changed.empty:
            changed.insert(0, "file", spec.file)
            changed[f"included_zero_{before_label}"] = before.loc[changed.index]
            changed[f"included_zero_{after_label}"] = after.loc[changed.index]
            rows.append(changed)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    """Write the audited CSV and print its three decision-relevant counts."""

    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    changed = report["absolute_difference"] > 0.0
    reported = report["changes_at_reported_precision"]
    crossings = confidence_interval_crossings()
    print(f"rows={len(report)} changed={int(changed.sum())}")
    print(f"maximum_absolute_difference={report['absolute_difference'].max():.17g}")
    print(f"reported_precision_changes={int(reported.sum())}")
    print(f"ci_zero_inclusion_changes={len(crossings)}")


if __name__ == "__main__":
    main()
