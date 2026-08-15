"""OOP orchestration for proposal-locked latent endotype discovery."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from traceesus.core.experiment import Experiment
from traceesus.core.io import write_json, write_manifest, write_standard_tables
from traceesus.registry import EndotypeModelSet, FULL_LADDER

from . import kernel
from .recovery import (
    paired_registry_contrasts,
    registry_validation_checks,
    run_model_registry,
)


@dataclass
class EndotypeDiscoveryArtifacts:
    """All repeat-level and summarized artifacts from one configured run."""

    raw_metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    parameters: pd.DataFrame
    raw_null: pd.DataFrame
    example: dict[str, object]
    summary: pd.DataFrame | None = None
    contrasts: pd.DataFrame | None = None
    null_summary: pd.DataFrame | None = None
    validation_checks: dict[str, object] | None = None
    metadata: dict[str, object] | None = None
    manifest: dict[str, object] | None = None


def _standard_tables(
    artifacts: EndotypeDiscoveryArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reshape legacy discovery results into the additive tidy contract."""

    identifiers = ["renal_effect_sd", "repeat", "method"]
    metrics = [
        column for column in artifacts.raw_metrics.columns if column not in identifiers
    ]
    raw_long = artifacts.raw_metrics.melt(
        id_vars=identifiers,
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    summary = artifacts.summary[
        [
            "renal_effect_sd", "method", "metric", "mean", "ci95_low",
            "ci95_high", "monte_carlo_standard_error", "repeat_count",
        ]
    ].copy()
    contrasts = artifacts.contrasts.copy()
    contrasts["contrast"] = (
        contrasts["difference_definition"] + ": " + contrasts["comparator"]
    )
    contrast_columns = [
        "renal_effect_sd", "metric", "contrast", "mean_difference",
        "ci95_low", "ci95_high", "repeat_count",
    ]
    return raw_long, summary, contrasts[contrast_columns]


def _example_table(example: dict[str, object]) -> pd.DataFrame:
    """Flatten the illustrative patient into a CSV-only plotting input."""

    observed = example["observed_biomarkers"]
    contribution = example["causal_estimated_renal_contribution"]
    neutralized = example["causal_renal_neutralized_biomarkers"]
    rows = []
    for biomarker in observed:
        rows.append(
            {
                "biomarker": biomarker,
                "observed_value": observed[biomarker],
                "renal_contribution": contribution[biomarker],
                "renal_neutralized_value": neutralized[biomarker],
                "associative_atrial_probability": example["associative_atrial_probability"],
                "causal_atrial_probability": example["causal_atrial_probability"],
                "renal_effect_sd": example["renal_effect_sd"],
            }
        )
    return pd.DataFrame(rows)


class EndotypeDiscoveryExperiment(Experiment):
    """Compare unlabeled associative and causal latent models on paired cohorts.

    The model registry is explicit so a future source-of-gain ablation can add a
    ``Model`` subclass as a row without changing simulator or metric plumbing.
    The registry drives the paired repeat loop; frozen kernels retain the exact
    simulator and EM arithmetic that produced the cited numerical artifacts.
    """

    name = "endotype_discovery"

    def __init__(
        self,
        config: kernel.ExperimentConfig,
        output_directory: Path | str,
        model_set: EndotypeModelSet = FULL_LADDER,
    ) -> None:
        self.config = config
        self.output_directory = Path(output_directory)
        self.model_set = model_set
        self.models = model_set.fitted_models
        self.artifacts: EndotypeDiscoveryArtifacts | None = None

    def configure(self) -> kernel.ExperimentConfig:
        """Validate every frozen numerical control before spawning seeds."""

        self.config.validate()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return self.config

    def run(self) -> EndotypeDiscoveryArtifacts:
        """Run recovery, null, and example streams using their legacy seed roots."""

        raw_metrics, diagnostics, parameters = run_model_registry(
            self.config, self.models
        )
        raw_null = kernel.run_k1_null_experiment(self.config)
        example = kernel.build_example_patient(self.config)
        self.artifacts = EndotypeDiscoveryArtifacts(
            raw_metrics=raw_metrics,
            diagnostics=diagnostics,
            parameters=parameters,
            raw_null=raw_null,
            example=example,
        )
        return self.artifacts

    def summarize(self) -> EndotypeDiscoveryArtifacts:
        """Apply the exact legacy group order, reductions, CIs, and validation."""

        if self.artifacts is None:
            raise RuntimeError("run() must precede summarize().")
        artifacts = self.artifacts
        artifacts.summary = kernel.summarize_repeated_metrics(artifacts.raw_metrics)
        artifacts.contrasts = paired_registry_contrasts(
            artifacts.raw_metrics,
            self.models,
            reference_method=self.model_set.contrast_reference,
        )
        artifacts.null_summary = kernel.summarize_k1_null(artifacts.raw_null)
        artifacts.validation_checks = registry_validation_checks(
            artifacts.raw_metrics,
            artifacts.diagnostics,
            artifacts.parameters,
            artifacts.raw_null,
            self.config,
            self.models,
            require_legacy_parameter_check=self.model_set.uses_legacy_parameter_check,
        )
        experiment_config = asdict(self.config)
        # HF-grid controls are additive extension settings, not inputs to the
        # proposal-locked discovery run.  Keep the legacy metadata byte shape
        # exact while leaving those controls available to the extension.
        simulation_config = experiment_config["simulation"]
        simulation_config.pop("heart_failure_prevalence", None)
        simulation_config.pop("heart_failure_effect_levels_sd", None)
        artifacts.metadata = {
            "experiment_config": experiment_config,
            "model_parameter_counts_k2": dict(
                zip(
                    (model.name for model in self.models),
                    self.model_set.parameter_counts,
                    strict=True,
                )
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "truth_usage": (
                "True mechanisms are generated and stored by the simulator but are "
                "never passed to a fit function; they are used only by evaluate_posterior."
            ),
            "label_orientation": (
                "Latent labels are oriented using the prespecified electrical-minus-"
                "competing biomarker anchor; the misleading NT-proBNP-like marker and "
                "the simulated truth labels are excluded from orientation."
            ),
        }
        return artifacts

    def plot(self) -> None:
        """Do nothing; figures are rendered by ``notebooks/figures.ipynb``."""

    def write(self) -> dict[str, Any]:
        """Write exact legacy tables/JSON plus an additive checksum manifest."""
        if self.artifacts is None or self.artifacts.summary is None:
            raise RuntimeError("summarize() must precede write().")
        artifacts = self.artifacts
        artifacts.raw_metrics.to_csv(
            self.output_directory / "raw_recovery_metrics.csv", index=False
        )
        artifacts.summary.to_csv(
            self.output_directory / "recovery_summary.csv", index=False
        )
        artifacts.contrasts.to_csv(
            self.output_directory / "paired_contrasts.csv", index=False
        )
        artifacts.diagnostics.to_csv(
            self.output_directory / "fit_diagnostics.csv", index=False
        )
        artifacts.parameters.to_csv(
            self.output_directory / "parameter_recovery.csv", index=False
        )
        artifacts.raw_null.to_csv(
            self.output_directory / "k1_null_raw.csv", index=False
        )
        artifacts.null_summary.to_csv(
            self.output_directory / "k1_null_summary.csv", index=False
        )
        raw_long, summary, contrasts = _standard_tables(artifacts)
        write_standard_tables(
            self.output_directory, self.name, raw_long, summary, contrasts
        )
        _example_table(artifacts.example).to_csv(
            self.output_directory / "endotype_discovery_example_patient.csv",
            index=False,
        )
        write_json(self.output_directory / "example_patient.json", artifacts.example)
        write_json(
            self.output_directory / "validation_checks.json",
            artifacts.validation_checks,
        )
        write_json(self.output_directory / "metadata.json", artifacts.metadata)
        artifacts.manifest = write_manifest(
            self.output_directory,
            experiment=self.name,
            config=artifacts.metadata["experiment_config"],
            master_seed=self.config.master_seed,
            wall_clock_runtime_seconds=self.wall_clock_runtime_seconds,
        )
        return {
            "raw_metrics": artifacts.raw_metrics,
            "summary": artifacts.summary,
            "contrasts": artifacts.contrasts,
            "diagnostics": artifacts.diagnostics,
            "parameters": artifacts.parameters,
            "raw_null": artifacts.raw_null,
            "null_summary": artifacts.null_summary,
            "example": artifacts.example,
            "validation_checks": artifacts.validation_checks,
            "metadata": artifacts.metadata,
            "manifest": artifacts.manifest,
        }
