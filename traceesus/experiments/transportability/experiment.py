"""OOP orchestration for proposal-locked cross-hospital transportability."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from traceesus.core.experiment import Experiment
from traceesus.core.io import write_json, write_manifest
from traceesus.core.model import Model, assert_truth_free_fit_interfaces
from traceesus.models.modular_causal_scm import (
    FrozenCausalSCM,
    ModularCausalSCM,
    PooledAssociativeTransportModel,
    TargetAdjustedAssociativeModel,
    TargetTransportOracle,
)

from . import kernel


@dataclass
class TransportRunArtifacts:
    """Raw and summarized products for one target-hospital specification set."""

    config: kernel.TransportExperimentConfig
    raw_metrics: pd.DataFrame
    fit_diagnostics: pd.DataFrame
    target_diagnostics: pd.DataFrame
    maximum_assay_error: float
    summary: pd.DataFrame | None = None
    contrasts: pd.DataFrame | None = None
    degradation: pd.DataFrame | None = None
    changes: pd.DataFrame | None = None
    negative_control: dict[str, object] | None = None
    validation_checks: dict[str, object] | None = None
    metadata: dict[str, object] | None = None


@dataclass
class TransportabilityArtifacts:
    """Main curve, shift ablation, and literal no-shift control artifacts."""

    main: TransportRunArtifacts
    ablation: TransportRunArtifacts
    exact_no_shift: TransportRunArtifacts
    manifest: dict[str, object] | None = None


def _exact_no_shift_config(
    config: kernel.TransportExperimentConfig,
) -> kernel.TransportExperimentConfig:
    """Recreate the historical identical-distribution control specification."""

    baseline = replace(kernel.TARGET_HOSPITALS[0], name="Exact no shift")
    identical_sources = tuple(
        replace(baseline, name=f"Identical source {label}")
        for label in ("A", "B", "C")
    )
    return replace(
        config,
        source_hospitals=identical_sources,
        target_hospitals=(baseline,),
    )


def _run_specification(
    config: kernel.TransportExperimentConfig,
) -> TransportRunArtifacts:
    """Execute only numerical kernels so lifecycle side effects stay separated."""

    raw, diagnostics, target, assay_error = kernel.run_transport_experiment(config)
    return TransportRunArtifacts(
        config=config,
        raw_metrics=raw,
        fit_diagnostics=diagnostics,
        target_diagnostics=target,
        maximum_assay_error=assay_error,
    )


def _summarize_specification(
    artifacts: TransportRunArtifacts,
    *,
    degradation: bool = False,
    ablation: bool = False,
) -> None:
    """Apply the exact legacy grouping, reduction, and validation order."""

    artifacts.summary = kernel.summarize_metrics(artifacts.raw_metrics)
    artifacts.contrasts = kernel.paired_contrasts(artifacts.raw_metrics)
    if degradation:
        artifacts.degradation = kernel.transport_degradation(artifacts.raw_metrics)
    if ablation:
        artifacts.changes = kernel.ablation_accuracy_changes(artifacts.raw_metrics)
    artifacts.negative_control = kernel.negative_control_check(
        artifacts.contrasts,
        artifacts.config,
    )
    artifacts.validation_checks = kernel.validation_checks(
        artifacts.raw_metrics,
        artifacts.fit_diagnostics,
        artifacts.target_diagnostics,
        artifacts.maximum_assay_error,
        artifacts.negative_control,
        artifacts.config,
    )


class TransportabilityExperiment(Experiment):
    """Evaluate unlabeled transport under paired hospital-shift scenarios.

    The main curve, one-component shift ablation, and literal identical-source-
    target control all use the frozen repeat kernel. That kernel deliberately
    preserves ``master_seed + 404404``, twelve child streams per repeat, target
    seeds from children 9 and 10 reset at every shift, and unused child 11.
    Model objects document the interchangeable posterior contract; the frozen
    repeat kernel remains authoritative for cited numbers and RNG consumption.
    """

    name = "transportability"

    def __init__(
        self,
        config: kernel.TransportExperimentConfig,
        output_directory: Path | str,
    ) -> None:
        self.config = config
        self.output_directory = Path(output_directory)
        self.models: tuple[Model, ...] = (
            PooledAssociativeTransportModel(),
            TargetAdjustedAssociativeModel(),
            FrozenCausalSCM(),
            ModularCausalSCM(),
            TargetTransportOracle(config.simulation),
        )
        self.artifacts: TransportabilityArtifacts | None = None

    def configure(self) -> kernel.TransportExperimentConfig:
        """Validate controls and preserve the fixed method-row ordering."""

        self.config.validate()
        registered_names = tuple(model.name for model in self.models)
        if registered_names != kernel.ALL_METHODS:
            raise AssertionError(
                "The transport model registry must preserve legacy row order."
            )
        assert_truth_free_fit_interfaces(self.models)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        (self.output_directory / "ablations").mkdir(parents=True, exist_ok=True)
        (self.output_directory / "exact_no_shift_control").mkdir(
            parents=True,
            exist_ok=True,
        )
        return self.config

    def run(self) -> TransportabilityArtifacts:
        """Run all three paired designs with their unchanged independent seed roots."""

        ablation_config = replace(
            self.config,
            target_hospitals=kernel.ABLATION_TARGETS,
        )
        exact_config = _exact_no_shift_config(self.config)
        self.artifacts = TransportabilityArtifacts(
            main=_run_specification(self.config),
            ablation=_run_specification(ablation_config),
            exact_no_shift=_run_specification(exact_config),
        )
        return self.artifacts

    def summarize(self) -> TransportabilityArtifacts:
        """Create legacy summaries, controls, checks, and metadata verbatim."""

        if self.artifacts is None:
            raise RuntimeError("run() must precede summarize().")
        artifacts = self.artifacts
        _summarize_specification(artifacts.main, degradation=True)
        _summarize_specification(artifacts.ablation, ablation=True)
        _summarize_specification(artifacts.exact_no_shift)

        artifacts.main.metadata = {
            "experiment_config": asdict(self.config),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "truth_usage": (
                "True mechanism labels are stored by the simulator and used only "
                "inside evaluate_posterior after all fitting and prediction."
            ),
            "target_calibration": (
                "Assay metadata plus renal status, background inflammation, and "
                "unlabeled biomarkers from the target calibration cohort; no "
                "mechanism labels are used."
            ),
            "assay_boundary": (
                "Assay offset and scale metadata are assumed known and applied "
                "equally to every method. Unknown calibration is not identified by "
                "this experiment."
            ),
        }
        artifacts.exact_no_shift.metadata = {
            "experiment_config": asdict(artifacts.exact_no_shift.config),
            "purpose": (
                "Literal no-shift negative control: all source and target hospitals "
                "share identical generating parameters."
            ),
            "truth_usage": (
                "True mechanism labels are used only after prediction for "
                "simulation evaluation."
            ),
        }
        return artifacts

    def plot(self) -> None:
        """Regenerate both main control figures and the shift-ablation figure."""

        if self.artifacts is None or self.artifacts.main.summary is None:
            raise RuntimeError("summarize() must precede plot().")
        artifacts = self.artifacts
        kernel.plot_transport_figure(
            artifacts.main.summary,
            self.config,
            self.output_directory,
        )
        kernel.plot_transport_controls(
            artifacts.main.summary,
            artifacts.main.degradation,
            self.config,
            self.output_directory,
        )
        kernel.plot_ablation_figure(
            artifacts.ablation.changes,
            self.output_directory / "ablations",
        )

    def write(self) -> dict[str, Any]:
        """Write exact legacy filenames and JSON shapes plus one root manifest."""

        if self.artifacts is None or self.artifacts.main.summary is None:
            raise RuntimeError("summarize() must precede write().")
        artifacts = self.artifacts
        self._write_main(artifacts.main)
        self._write_ablation(artifacts.ablation)
        self._write_exact(artifacts.exact_no_shift)
        artifacts.manifest = write_manifest(
            self.output_directory,
            experiment=self.name,
            config=self.config,
            master_seed=self.config.master_seed,
            wall_clock_runtime_seconds=self.wall_clock_runtime_seconds,
        )
        return {
            "raw_metrics": artifacts.main.raw_metrics,
            "summary": artifacts.main.summary,
            "contrasts": artifacts.main.contrasts,
            "degradation": artifacts.main.degradation,
            "fit_diagnostics": artifacts.main.fit_diagnostics,
            "target_diagnostics": artifacts.main.target_diagnostics,
            "negative_control": artifacts.main.negative_control,
            "validation_checks": artifacts.main.validation_checks,
            "metadata": artifacts.main.metadata,
            "ablation": artifacts.ablation,
            "exact_no_shift_control": artifacts.exact_no_shift,
            "manifest": artifacts.manifest,
        }

    def _write_main(self, artifacts: TransportRunArtifacts) -> None:
        """Persist main-curve compatibility products in their historical shape."""

        artifacts.raw_metrics.to_csv(
            self.output_directory / "raw_transport_metrics.csv", index=False
        )
        artifacts.summary.to_csv(
            self.output_directory / "transport_summary.csv", index=False
        )
        artifacts.contrasts.to_csv(
            self.output_directory / "paired_transport_contrasts.csv", index=False
        )
        artifacts.degradation.to_csv(
            self.output_directory / "transport_degradation.csv", index=False
        )
        artifacts.fit_diagnostics.to_csv(
            self.output_directory / "fit_diagnostics.csv", index=False
        )
        artifacts.target_diagnostics.to_csv(
            self.output_directory / "target_calibration_diagnostics.csv", index=False
        )
        write_json(
            self.output_directory / "negative_control.json",
            artifacts.negative_control,
        )
        write_json(
            self.output_directory / "validation_checks.json",
            artifacts.validation_checks,
        )
        write_json(self.output_directory / "metadata.json", artifacts.metadata)

    def _write_ablation(self, artifacts: TransportRunArtifacts) -> None:
        """Persist one-factor ablation products without inventing new legacy JSON."""

        directory = self.output_directory / "ablations"
        artifacts.raw_metrics.to_csv(directory / "raw_ablation_metrics.csv", index=False)
        artifacts.summary.to_csv(directory / "ablation_summary.csv", index=False)
        artifacts.contrasts.to_csv(
            directory / "paired_ablation_contrasts.csv", index=False
        )
        artifacts.changes.to_csv(
            directory / "ablation_accuracy_changes.csv", index=False
        )
        artifacts.fit_diagnostics.to_csv(directory / "fit_diagnostics.csv", index=False)
        artifacts.target_diagnostics.to_csv(
            directory / "target_calibration_diagnostics.csv", index=False
        )
        write_json(directory / "negative_control.json", artifacts.negative_control)
        write_json(directory / "validation_checks.json", artifacts.validation_checks)

    def _write_exact(self, artifacts: TransportRunArtifacts) -> None:
        """Persist the literal identical-distribution negative control products."""

        directory = self.output_directory / "exact_no_shift_control"
        artifacts.raw_metrics.to_csv(directory / "raw_metrics.csv", index=False)
        artifacts.summary.to_csv(directory / "summary.csv", index=False)
        artifacts.contrasts.to_csv(directory / "paired_contrasts.csv", index=False)
        artifacts.fit_diagnostics.to_csv(directory / "fit_diagnostics.csv", index=False)
        artifacts.target_diagnostics.to_csv(
            directory / "target_calibration_diagnostics.csv", index=False
        )
        write_json(directory / "negative_control.json", artifacts.negative_control)
        write_json(directory / "validation_checks.json", artifacts.validation_checks)
        write_json(directory / "metadata.json", artifacts.metadata)

    def figures_only(self) -> None:
        """Regenerate all transport figures from existing compatibility tables."""

        summary = pd.read_csv(self.output_directory / "transport_summary.csv")
        degradation = pd.read_csv(
            self.output_directory / "transport_degradation.csv"
        )
        changes = pd.read_csv(
            self.output_directory / "ablations" / "ablation_accuracy_changes.csv"
        )
        kernel.plot_transport_figure(summary, self.config, self.output_directory)
        kernel.plot_transport_controls(
            summary,
            degradation,
            self.config,
            self.output_directory,
        )
        kernel.plot_ablation_figure(changes, self.output_directory / "ablations")


__all__ = ["TransportabilityExperiment"]
