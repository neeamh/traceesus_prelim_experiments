"""OOP orchestration for the proposal-locked known-SCM experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from traceesus.core.experiment import Experiment
from traceesus.core.io import write_json, write_manifest
from traceesus.models.known_scm import (
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
        """Regenerate the historical Figure P1 PNG and PDF from the summary."""

        if self.artifacts is None or self.artifacts.summary is None:
            raise RuntimeError("summarize() must precede plot().")
        kernel.plot_primary_figure(
            self.artifacts.summary,
            self.config,
            self.output_directory / "figure_P1.png",
        )

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
        write_json(self.output_directory / "run_metadata.json", artifacts.metadata)
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
            "figure_png": self.output_directory / "figure_P1.png",
            "figure_pdf": self.output_directory / "figure_P1.pdf",
            "metadata": artifacts.metadata,
            "manifest": artifacts.manifest,
        }

    def figures_only(self) -> None:
        """Regenerate figures from the existing compatibility summary table."""

        summary = pd.read_csv(
            self.output_directory / "main_simulation_summary.csv"
        )
        kernel.plot_primary_figure(
            summary,
            self.config,
            self.output_directory / "figure_P1.png",
        )

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
