"""Build the v2-to-v3 delta audit for conditional-EM unification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from scripts.em_unification_delta import build_report, confidence_interval_crossings
except ModuleNotFoundError:
    from em_unification_delta import build_report, confidence_interval_crossings


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "outputs_locked_v2"
V3_ROOT = ROOT / "outputs_locked_v3"
REPORT_PATH = ROOT / "reports" / "conditional_em_unification_delta.csv"


def conditional_report() -> pd.DataFrame:
    """Return the scientific-keyed v2-to-v3 delta table."""

    return build_report(V2_ROOT, V3_ROOT, "v2", "v3")


def conditional_crossings() -> pd.DataFrame:
    """Return confidence intervals whose zero-inclusion status changed."""

    return confidence_interval_crossings(V2_ROOT, V3_ROOT, "v2", "v3")


def main() -> None:
    """Write the audit and print its three decision-relevant findings."""

    report = conditional_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    changed = report["absolute_difference"] > 0.0
    reported = report["changes_at_reported_precision"]
    crossings = conditional_crossings()
    print(f"rows={len(report)} changed={int(changed.sum())}")
    print(f"maximum_absolute_difference={report['absolute_difference'].max():.17g}")
    print(f"reported_precision_changes={int(reported.sum())}")
    print(f"ci_zero_inclusion_changes={len(crossings)}")


if __name__ == "__main__":
    main()
