"""Run the two prespecified 500-repeat ARCADIA-calibrated discovery streams."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.arcadia_calibrated import CONFIG
from traceesus.core.io import write_json, write_manifest
from traceesus.core.stats import monte_carlo_summary
from traceesus.experiments.endotype_discovery import kernel
from traceesus.experiments.endotype_discovery.recovery import (
    paired_registry_contrasts,
    run_model_registry,
    run_redundancy_sweep,
)
from traceesus.registry import MODEL_LADDER, full_ladder_for_config


def _summary(raw: pd.DataFrame, level_column: str) -> pd.DataFrame:
    """Summarize every finite repeat-level metric within level and method."""

    identifiers = {
        "repeat",
        "renal_effect_sd",
        "heart_failure_effect_sd",
        "method",
    }
    metrics = [
        column
        for column in raw.columns
        if column not in identifiers and raw[column].dtype.kind in "fi"
    ]
    rows: list[dict[str, object]] = []
    for (level, method), block in raw.groupby([level_column, "method"], sort=True):
        for metric in metrics:
            finite = block[metric].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size < 2:
                continue
            rows.append(
                {
                    level_column: level,
                    "method": method,
                    "metric": metric,
                    **monte_carlo_summary(finite),
                }
            )
    return pd.DataFrame(rows)


def run(output_directory: Path, workers: int) -> None:
    """Execute renal and redundancy streams and write additive artifacts."""

    started = perf_counter()
    output_directory.mkdir(parents=True, exist_ok=True)
    config = replace(CONFIG, workers=workers)
    model_set = full_ladder_for_config(config)
    models = model_set.fitted_models
    r4_name = models[3].name

    print("stream=renal status=started repeats=500", flush=True)
    renal_started = perf_counter()
    renal_raw, renal_diagnostics, renal_parameters = run_model_registry(config, models)
    renal_summary = kernel.summarize_repeated_metrics(renal_raw)
    renal_contrasts = paired_registry_contrasts(
        renal_raw,
        models,
        reference_method=r4_name,
    )
    renal_raw.to_csv(output_directory / "arcadia_renal_sweep_raw.csv", index=False)
    renal_summary.to_csv(
        output_directory / "arcadia_renal_sweep_summary.csv", index=False
    )
    renal_contrasts.to_csv(
        output_directory / "arcadia_renal_sweep_paired_contrasts.csv", index=False
    )
    renal_diagnostics.to_csv(
        output_directory / "arcadia_renal_sweep_fit_diagnostics.csv", index=False
    )
    renal_parameters.to_csv(
        output_directory / "arcadia_renal_sweep_parameters.csv", index=False
    )
    print(
        f"stream=renal status=completed rows={len(renal_raw)} "
        f"seconds={perf_counter() - renal_started:.3f}",
        flush=True,
    )

    print("stream=redundancy status=started repeats=500", flush=True)
    redundancy_started = perf_counter()
    redundancy_raw, redundancy_diagnostics, redundancy_parameters = (
        run_redundancy_sweep(config, models)
    )
    redundancy_summary = _summary(redundancy_raw, "heart_failure_effect_sd")
    redundancy_contrasts = paired_registry_contrasts(
        redundancy_raw,
        models,
        level_column="heart_failure_effect_sd",
        reference_method=r4_name,
    )
    redundancy_raw.to_csv(
        output_directory / "arcadia_redundancy_sweep_raw.csv", index=False
    )
    redundancy_summary.to_csv(
        output_directory / "arcadia_redundancy_sweep_summary.csv", index=False
    )
    redundancy_contrasts.to_csv(
        output_directory / "arcadia_redundancy_sweep_paired_contrasts.csv",
        index=False,
    )
    redundancy_diagnostics.to_csv(
        output_directory / "arcadia_redundancy_sweep_fit_diagnostics.csv",
        index=False,
    )
    redundancy_parameters.to_csv(
        output_directory / "arcadia_redundancy_sweep_parameters.csv", index=False
    )
    print(
        f"stream=redundancy status=completed rows={len(redundancy_raw)} "
        f"seconds={perf_counter() - redundancy_started:.3f}",
        flush=True,
    )

    runtime = perf_counter() - started
    write_json(
        output_directory / "arcadia_confirmatory_metadata.json",
        {
            "status": "confirmatory",
            "repeats_per_level": config.repeats_per_level,
            "seed_root": config.master_seed,
            "workers": config.workers,
            "model_ladder": [list(row) for row in MODEL_LADDER],
            "r4_reference": r4_name,
            "simulation_config": asdict(config.simulation),
            "left_atrial_size_channel": "not represented",
            "renal_ptfv1_mask_approximation": (
                "The measured -0.095 renal loading on PTFV1 is treated as near-noise."
            ),
        },
    )
    write_manifest(
        output_directory,
        experiment="arcadia_calibrated_discovery_confirmatory",
        config=config,
        master_seed=config.master_seed,
        wall_clock_runtime_seconds=runtime,
    )
    print(f"status=completed seconds={runtime:.3f}", flush=True)


def main() -> None:
    """Parse the additive output location and worker count."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs_arcadia_calibrated"),
    )
    parser.add_argument("--workers", type=int, default=CONFIG.workers)
    arguments = parser.parse_args()
    run(arguments.out, arguments.workers)


if __name__ == "__main__":
    main()
