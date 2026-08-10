"""OOP orchestration for the proposal-locked supervised comparison."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from traceesus.core.experiment import Experiment
from traceesus.core.io import write_json, write_manifest
from traceesus.core.model import Model
from traceesus.models.causal_scm import SupervisedStructuralCausalModel
from traceesus.models.logistic import (
    BiomarkersOnlyLogisticModel,
    KidneyAdjustedLogisticModel,
)

from . import kernel


@dataclass
class ModelComparisonArtifacts:
    """Repeat-level and summarized products of one supervised comparison run."""

    raw_metrics: pd.DataFrame
    fit_diagnostics: pd.DataFrame
    summary: pd.DataFrame | None = None
    paired_contrasts: pd.DataFrame | None = None
    metadata: dict[str, object] | None = None
    manifest: dict[str, object] | None = None


class ModelComparisonExperiment(Experiment):
    """Compare three supervised classifiers on identical paired cohorts.

    All models, including the structural causal model, use the known synthetic
    mechanism labels during fitting.  This experiment measures supervised
    classification under renal-marker confounding; it does not discover latent
    endotypes.  The explicit model registry exposes the common ``Model``
    interface without changing the historical kernel's seed or tie-breaking
    order.
    """

    name = "model_comparison"

    def __init__(
        self,
        config: kernel.ComparisonConfig,
        output_directory: Path | str,
    ) -> None:
        self.config = config
        self.output_directory = Path(output_directory)
        self.models: tuple[Model, ...] = (
            BiomarkersOnlyLogisticModel(),
            KidneyAdjustedLogisticModel(),
            SupervisedStructuralCausalModel(),
        )
        self.artifacts: ModelComparisonArtifacts | None = None

    def configure(self) -> kernel.ComparisonConfig:
        """Validate controls and the fixed supervised model-row ordering."""

        self.config.validate()
        registered_names = tuple(model.name for model in self.models)
        if registered_names != kernel.METHODS:
            raise AssertionError(
                "The model registry must preserve the proposal-locked row order."
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return self.config

    def run(self) -> ModelComparisonArtifacts:
        """Execute the exact legacy seed tree, fits, and random tie breaking."""

        raw_metrics, fit_diagnostics = kernel.run_comparison(self.config)
        self.artifacts = ModelComparisonArtifacts(
            raw_metrics=raw_metrics,
            fit_diagnostics=fit_diagnostics,
        )
        return self.artifacts

    def summarize(self) -> ModelComparisonArtifacts:
        """Apply the historical reduction and paired-contrast operation order."""

        if self.artifacts is None:
            raise RuntimeError("run() must precede summarize().")
        artifacts = self.artifacts
        artifacts.summary = kernel.summarize_metrics(artifacts.raw_metrics)
        artifacts.paired_contrasts = kernel.summarize_paired_contrasts(
            artifacts.raw_metrics
        )
        artifacts.metadata = {
            "config": asdict(self.config),
            "models": {
                kernel.ASSOCIATIVE_BIOMARKERS: (
                    "L2-penalized logistic regression fit to the three continuous "
                    "biomarkers; no graph and no renal-status input."
                ),
                kernel.ASSOCIATIVE_ADJUSTED: (
                    "The same logistic regression fit to the three biomarkers plus "
                    "observed renal status; no graph and no interventions."
                ),
                kernel.SCM_COUNTERFACTUAL: (
                    "Fitted prespecified SCM with mechanism and renal biomarker "
                    "equations; posterior-integrated disablement and sufficiency."
                ),
            },
            "training_label_boundary": (
                "All three models use known synthetic mechanism labels during "
                "training. This is supervised classification, not unsupervised "
                "endotype discovery."
            ),
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "matplotlib": plt.matplotlib.__version__,
            },
            "ci_definition": (
                "Two-sided 95% t confidence interval for the mean across 500 paired "
                "train/test simulation repeats."
            ),
        }
        return artifacts

    def plot(self) -> None:
        """Regenerate the historical PNG and PDF from the in-memory summary."""

        if self.artifacts is None or self.artifacts.summary is None:
            raise RuntimeError("summarize() must precede plot().")
        kernel.plot_comparison(
            self.artifacts.summary,
            self.config,
            self.output_directory / "figure_P1_associative_vs_scm.png",
        )

    def write(self) -> dict[str, Any]:
        """Write exact compatibility outputs plus an additive checksum manifest."""

        if self.artifacts is None or self.artifacts.summary is None:
            raise RuntimeError("summarize() must precede write().")
        artifacts = self.artifacts
        artifacts.raw_metrics.to_csv(
            self.output_directory / "raw_metrics.csv", index=False
        )
        artifacts.fit_diagnostics.to_csv(
            self.output_directory / "fit_diagnostics.csv", index=False
        )
        artifacts.summary.to_csv(self.output_directory / "summary.csv", index=False)
        artifacts.paired_contrasts.to_csv(
            self.output_directory / "paired_contrasts.csv", index=False
        )
        write_json(self.output_directory / "metadata.json", artifacts.metadata)
        artifacts.manifest = write_manifest(
            self.output_directory,
            experiment=self.name,
            config=self.config,
            master_seed=self.config.seed,
            wall_clock_runtime_seconds=self.wall_clock_runtime_seconds,
        )
        return {
            "raw_metrics": artifacts.raw_metrics,
            "fit_diagnostics": artifacts.fit_diagnostics,
            "summary": artifacts.summary,
            "paired_contrasts": artifacts.paired_contrasts,
            "figure_png": self.output_directory
            / "figure_P1_associative_vs_scm.png",
            "figure_pdf": self.output_directory
            / "figure_P1_associative_vs_scm.pdf",
            "metadata": artifacts.metadata,
            "manifest": artifacts.manifest,
        }

    def figures_only(self) -> None:
        """Regenerate figures from the existing compatibility summary table."""

        summary = pd.read_csv(self.output_directory / "summary.csv")
        kernel.plot_comparison(
            summary,
            self.config,
            self.output_directory / "figure_P1_associative_vs_scm.png",
        )

    def console_summary(self) -> str:
        """Render the original compact diagnostic summary after execution."""

        if self.artifacts is None or self.artifacts.summary is None:
            raise RuntimeError("summarize() must precede console_summary().")
        return kernel.console_summary(
            {
                "summary": self.artifacts.summary,
                "fit_diagnostics": self.artifacts.fit_diagnostics,
            }
        )
