"""Compare the exploratory in-sample and held-out identity-drift sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LEVEL_COLUMNS = ("renal_effect_sd", "heart_failure_effect_sd")


def build_report(old_path: Path, new_path: Path) -> pd.DataFrame:
    """Return mean old, new, and signed change for every recorded drift metric."""

    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    if old.shape != new.shape or list(old.columns) != list(new.columns):
        raise ValueError("Old and new drift files must have the same schema and shape.")
    metrics = tuple(
        column
        for column in old.columns
        if column not in ("repeat", "called_atrial_size", *LEVEL_COLUMNS)
    )
    rows: list[dict[str, object]] = []
    for levels, old_block in old.groupby(list(LEVEL_COLUMNS), sort=True):
        new_block = new[
            (new[LEVEL_COLUMNS[0]] == levels[0])
            & (new[LEVEL_COLUMNS[1]] == levels[1])
        ]
        for metric in metrics:
            old_mean = float(old_block[metric].mean())
            new_mean = float(new_block[metric].mean())
            rows.append(
                {
                    LEVEL_COLUMNS[0]: levels[0],
                    LEVEL_COLUMNS[1]: levels[1],
                    "metric": metric,
                    "in_sample_mean": old_mean,
                    "held_out_mean": new_mean,
                    "held_out_minus_in_sample": new_mean - old_mean,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Parse explicit input paths and write the comparison table."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_report(arguments.old, arguments.new)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(arguments.out, index=False)
    print(f"rows={len(report)} max_abs_change={report['held_out_minus_in_sample'].abs().max():.17g}")


if __name__ == "__main__":
    main()
