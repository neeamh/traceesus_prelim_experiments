"""OOP orchestration for the proposal-locked known-SCM experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from traceesus.core.experiment import Experiment
from traceesus.core.data_dictionary import write_data_dictionary
from traceesus.core.io import write_json, write_manifest, write_standard_tables
from archive.counterfactual.known_scm import (
    KnownKidneyBlindPosteriorModel,
    KnownStructuralCausalModel,
)

from . import kernel


@dataclass
class CounterfactualArtifacts:
    """Repeat-level and summarized products from one known-SCM query run."""

    raw_metrics: pd.DataFrame
    null_results: pd.DataFrame
    summary: pd.DataFrame | None = None
    paired_differences: pd.DataFrame | None = None
    null_summary: pd.DataFrame | None = None
    metadata: dict[str, object] | None = None
    manifest: dict[str, object] | None = None


def _standard_tables(
    artifacts: CounterfactualArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rename the already-tidy known-SCM outputs to the common contract."""

    raw_long = artifacts.raw_metrics[
        [
            "strength_index", "renal_effect_sd", "repeat", "method", "metric",
            "value", "denominator",
        ]
    ].copy()
    summary = artifacts.summary.rename(
        columns={
            "n_repeats": "repeat_count",
            "monte_carlo_se": "monte_carlo_standard_error",
            "ci_low": "ci95_low",
            "ci_high": "ci95_high",
        }
    )
    summary_columns = [
        "strength_index", "renal_effect_sd", "method", "metric", "mean",
        "ci95_low", "ci95_high", "monte_carlo_standard_error", "repeat_count",
    ]
    contrasts = artifacts.paired_differences.rename(
        columns={
            "n_repeats": "repeat_count",
            "ci_low": "ci95_low",
            "ci_high": "ci95_high",
        }
    )
    contrast_columns = [
        "strength_index", "renal_effect_sd", "metric", "contrast",
        "mean_difference", "ci95_low", "ci95_high", "repeat_count",
    ]
    return raw_long, summary[summary_columns], contrasts[contrast_columns]


class CounterfactualExperiment(Experiment):
    """Compare kidney-blind and kidney-aware queries of a known toy SCM.

    The SCM parameters are prespecified in ``ExperimentConfig`` and are the
    parameters that generated the synthetic cohorts.  ``run`` therefore makes
    no fitting call.  The kidney-aware posterior is retained as a fairness
    diagnostic: in the symmetric K=2 specification, normalized sufficiency and
    disablement are monotone transformations of that posterior, so any gain
    over the kidney-blind baseline comes from modeling the renal path rather
    than from a causal query beating a Bayes classifier.
    """

    name = "counterfactual"

    def __init__(
        self,
        config: kernel.ExperimentConfig,
        output_directory: Path | str,
    ) -> None:
        self.config = config
        self.output_directory = Path(output_directory)
        # Factories expose the uniform Model interface without pretending that
        # a single renal-effect value applies across all four simulation levels.
        self.model_factories = (
            KnownKidneyBlindPosteriorModel,
            KnownStructuralCausalModel,
        )
        self.artifacts: CounterfactualArtifacts | None = None

    def configure(self) -> kernel.ExperimentConfig:
        """Validate frozen controls before constructing the legacy seed tree."""

        self.config.validate()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return self.config

    def run(self) -> CounterfactualArtifacts:
        """Execute the exact paired main and K=1-null RNG streams without fitting."""

        raw_metrics = kernel.run_main_simulation(self.config)
        null_results = kernel.run_k1_null_experiment(self.config)
        self.artifacts = CounterfactualArtifacts(
            raw_metrics=raw_metrics,
            null_results=null_results,
        )
        return self.artifacts

    def summarize(self) -> CounterfactualArtifacts:
        """Apply historical dataframe ordering and exact scalar reductions."""

        if self.artifacts is None:
            raise RuntimeError("run() must precede summarize().")
        artifacts = self.artifacts
        artifacts.summary = kernel.summarize_repeated_simulation(
            artifacts.raw_metrics
        )
        artifacts.paired_differences = kernel.summarize_paired_differences(
            artifacts.raw_metrics
        )
        artifacts.null_summary = kernel.summarize_k1_null(artifacts.null_results)
        artifacts.metadata = {
            "config": asdict(self.config),
            "software_versions": kernel.software_versions(),
            "primary_estimand": (
                "Expected top-1 true-mechanism ranking accuracy across repeated "
                "simulated studies."
            ),
            "subgroup_estimand": (
                "Expected false atrial classification among renal-impaired "
                "patients whose true mechanism is competing."
            ),
            "ci_definition": (
                "Two-sided 95% t confidence interval for the mean across paired "
                "simulation repeats; empirical 2.5th and 97.5th repeat quantiles "
                "are saved separately."
            ),
            "null_definition": (
                "False K=2 selection rate when truth is one homogeneous Gaussian "
                "regime after subtracting the prespecified renal biomarker path."
            ),
        }
        return artifacts

    def plot(self) -> None:
        """Do nothing; figures are rendered by ``notebooks/figures.ipynb``."""

    def write(self) -> dict[str, Any]:
        """Write five exact CSVs, exact metadata, and an additive manifest."""

        if self.artifacts is None or self.artifacts.summary is None:
            raise RuntimeError("summarize() must precede write().")
        artifacts = self.artifacts
        artifacts.raw_metrics.to_csv(
            self.output_directory / "main_simulation_raw_metrics.csv", index=False
        )
        artifacts.summary.to_csv(
            self.output_directory / "main_simulation_summary.csv", index=False
        )
        artifacts.paired_differences.to_csv(
            self.output_directory / "paired_method_differences.csv", index=False
        )
        artifacts.null_results.to_csv(
            self.output_directory / "k1_null_raw_results.csv", index=False
        )
        artifacts.null_summary.to_csv(
            self.output_directory / "k1_null_summary.csv", index=False
        )
        raw_long, summary, contrasts = _standard_tables(artifacts)
        write_standard_tables(
            self.output_directory, self.name, raw_long, summary, contrasts
        )
        write_json(self.output_directory / "run_metadata.json", artifacts.metadata)
        write_json(self.output_directory / "metadata.json", artifacts.metadata)
        write_data_dictionary(self.output_directory / "data_dictionary.csv")
        artifacts.manifest = write_manifest(
            self.output_directory,
            experiment=self.name,
            config=self.config,
            master_seed=self.config.seed,
            wall_clock_runtime_seconds=self.wall_clock_runtime_seconds,
        )
        return {
            "raw_metrics": artifacts.raw_metrics,
            "summary": artifacts.summary,
            "paired_differences": artifacts.paired_differences,
            "null_results": artifacts.null_results,
            "null_summary": artifacts.null_summary,
            "metadata": artifacts.metadata,
            "manifest": artifacts.manifest,
        }

    def console_summary(self) -> str:
        """Render the historical compact diagnostics after summarization."""

        if self.artifacts is None or self.artifacts.summary is None:
            raise RuntimeError("summarize() must precede console_summary().")
        return kernel._format_console_summary(
            {
                "summary": self.artifacts.summary,
                "null_summary": self.artifacts.null_summary,
            }
        )
