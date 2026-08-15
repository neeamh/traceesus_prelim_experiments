"""Heart-failure redundancy sweep built on the locked recovery machinery.

Scientific design
-----------------
Renal distortion is fixed at the locked strong level (1.5 SD on NT-proBNP)
while the heart-failure path on PTFV1 sweeps 0.0 -> 1.5 SD.  With both
nuisances active, the pair can jointly reproduce the full atrial signature:
NT-proBNP elevated by kidneys, PTFV1 elevated by heart failure.  That is the
redundancy regime in which posterior resemblance and counterfactual
sufficiency/disablement are expected to separate.

Results are reported inside four prespecified nuisance profiles —
uncomplicated, renal-only, heart-failure-only, and redundant (renal AND HF) —
with false-atrial attribution always computed over the competing-mechanism
patients of a profile.

This experiment writes to its own output directory and draws from its own
salted seed root.  It shares no seed and no output file with the
proposal-locked artifacts, so it can be rerun or reconfigured freely without
touching a cited number.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from traceesus.core.experiment import Experiment
from traceesus.core.io import write_json, write_manifest
from traceesus.core.stats import monte_carlo_summary
from traceesus.core.seeds import (
    REDUNDANCY_QUADRANT_SEED_OFFSET,
    REDUNDANCY_SWEEP_SEED_ROOT,
)
from traceesus.models import TwoNuisanceCounterfactualSCM
from traceesus.registry import FULL_LADDER
from traceesus.simulators.two_mechanism import TwoMechanismSimulator

from . import kernel
from .recovery import _nuisance_subgroups, paired_registry_contrasts, run_redundancy_sweep

_SUBGROUP_NAMES = ("uncomplicated", "renal_only", "heart_failure_only", "redundant")


@dataclass
class RedundancySweepArtifacts:
    """Repeat-level and summarized artifacts from one sweep run."""

    raw_metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    parameters: pd.DataFrame
    summary: pd.DataFrame | None = None
    contrasts: pd.DataFrame | None = None
    validation_checks: dict[str, object] | None = None
    manifest: dict[str, object] | None = None


class RedundancySweepExperiment(Experiment):
    """Sweep the PTFV1 heart-failure path at fixed strong renal distortion."""

    name = "redundancy_sweep"

    def __init__(
        self,
        config: kernel.ExperimentConfig,
        output_directory: Path | str,
    ) -> None:
        self.config = config
        self.output_directory = Path(output_directory)
        # The fourth row is the decisive comparison: the same constrained SCM,
        # fitted by the same EM on the same cohort and paired seed, answered by
        # sufficiency/disablement instead of the posterior.  Any difference
        # between rows three and four is attributable to the query alone.
        self.models = FULL_LADDER.fitted_models
        self.artifacts: RedundancySweepArtifacts | None = None

    def configure(self) -> kernel.ExperimentConfig:
        """Validate the frozen controls before spawning the salted sweep seeds."""

        self.config.validate()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return self.config

    def run(self) -> RedundancySweepArtifacts:
        """Execute the sweep on paired cohorts with the dedicated seed root."""

        raw_metrics, diagnostics, parameters = run_redundancy_sweep(
            self.config,
            self.models,
        )
        self.artifacts = RedundancySweepArtifacts(
            raw_metrics=raw_metrics,
            diagnostics=diagnostics,
            parameters=parameters,
        )
        return self.artifacts

    def summarize(self) -> RedundancySweepArtifacts:
        """Reduce repeats with the shared Monte Carlo summary, per HF level."""

        artifacts = self._require_artifacts()
        raw = artifacts.raw_metrics
        value_columns = [
            column
            for column in raw.columns
            if column not in ("repeat", "renal_effect_sd", "heart_failure_effect_sd", "method")
            and raw[column].dtype.kind in "fi"
        ]
        summary_rows: list[dict[str, object]] = []
        for (hf_level, method), block in raw.groupby(
            ["heart_failure_effect_sd", "method"], sort=True
        ):
            for column in value_columns:
                values = block[column].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size < 2:
                    continue
                summary_rows.append(
                    {
                        "heart_failure_effect_sd": hf_level,
                        "method": method,
                        "metric": column,
                        **monte_carlo_summary(finite),
                    }
                )
        artifacts.summary = pd.DataFrame(summary_rows)
        artifacts.contrasts = paired_registry_contrasts(
            raw,
            self.models,
            level_column="heart_failure_effect_sd",
        )
        artifacts.validation_checks = self._validation_checks(raw)
        return artifacts

    def plot(self) -> None:
        """Figures are produced by the dedicated figure session, not this run."""

    def write(self) -> dict[str, object]:
        """Persist sweep artifacts and an additive manifest."""

        artifacts = self._require_artifacts()
        out = self.output_directory
        artifacts.raw_metrics.to_csv(out / "raw_sweep_metrics.csv", index=False)
        artifacts.diagnostics.to_csv(out / "fit_diagnostics.csv", index=False)
        artifacts.parameters.to_csv(out / "parameter_recovery.csv", index=False)
        if artifacts.summary is not None:
            artifacts.summary.to_csv(out / "sweep_summary.csv", index=False)
        if artifacts.contrasts is not None:
            artifacts.contrasts.to_csv(out / "paired_sweep_contrasts.csv", index=False)
        if artifacts.validation_checks is not None:
            write_json(out / "validation_checks.json", artifacts.validation_checks)
        self.quadrant_sample().to_csv(out / "quadrant_sample.csv", index=False)
        metadata = {
            "experiment": self.name,
            "design": (
                "Fixed strong renal distortion on NT-proBNP; swept heart-failure "
                "path on PTFV1; four prespecified nuisance-profile subgroups."
            ),
            "renal_effect_sd_fixed": self.config.null_renal_effect_sd,
            "heart_failure_effect_levels_sd": list(
                self.config.simulation.heart_failure_effect_levels_sd
            ),
            "heart_failure_prevalence": self.config.simulation.heart_failure_prevalence,
            "seed_root": REDUNDANCY_SWEEP_SEED_ROOT,
            "oracle_note": (
                "The renal-only data-generating oracle is excluded: under a "
                "nonzero heart-failure path it is not the true-DGP ceiling."
            ),
            "config": asdict(self.config),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
        }
        write_json(out / "metadata.json", metadata)
        manifest = write_manifest(
            out,
            experiment=self.name,
            config=self.config,
            master_seed=self.config.master_seed,
            wall_clock_runtime_seconds=self.wall_clock_runtime_seconds,
        )
        artifacts.manifest = manifest
        return {
            "raw_metrics": artifacts.raw_metrics,
            "summary": artifacts.summary,
            "contrasts": artifacts.contrasts,
            "validation_checks": artifacts.validation_checks,
            "manifest": manifest,
        }

    def quadrant_sample(self, patients: int = 4_000) -> pd.DataFrame:
        """Emit patient-level sufficiency and disablement for the quadrant figure.

        One representative cohort per heart-failure level, fitted on a training
        cohort and scored on a larger evaluation cohort so the sparse redundant
        profile has enough patients to plot.  This is figure input, deliberately
        outside the repeat loop: it must never influence a summarized estimate.
        """

        config = self.config
        fixed_renal = config.null_renal_effect_sd
        model = TwoNuisanceCounterfactualSCM()
        rows: list[dict[str, object]] = []
        for level_index, hf_level in enumerate(
            config.simulation.heart_failure_effect_levels_sd
        ):
            sequences = np.random.SeedSequence(
                config.master_seed + REDUNDANCY_QUADRANT_SEED_OFFSET + level_index
            ).spawn(3)
            simulator = TwoMechanismSimulator(config.simulation, fixed_renal, hf_level)
            training = simulator.simulate(
                np.random.default_rng(sequences[0]),
                config.simulation.training_patients,
            )
            evaluation = simulator.simulate(
                np.random.default_rng(sequences[1]), patients
            )
            fitted = model.fit(
                training.observed, np.random.default_rng(sequences[2]), config.fitting
            )
            scores = fitted.counterfactual_scores(evaluation.observed)
            profiles = _nuisance_subgroups(evaluation.observed)
            profile_label = np.empty(patients, dtype=object)
            for name, mask in profiles.items():
                profile_label[mask] = name
            atrial = int(kernel.Mechanism.ATRIAL)
            frame = pd.DataFrame(
                {
                    "heart_failure_effect_sd": hf_level,
                    "nuisance_profile": profile_label,
                    "true_mechanism": np.where(
                        evaluation.truth.mechanism == atrial, "atrial", "competing"
                    ),
                    "sufficiency_atrial": scores["sufficiency"][:, atrial],
                    "disablement_atrial": scores["disablement"][:, atrial],
                    "posterior_atrial": scores["posterior"][:, atrial],
                }
            )
            rows.extend(frame.to_dict("records"))
        return pd.DataFrame(rows)

    def _require_artifacts(self) -> RedundancySweepArtifacts:
        if self.artifacts is None:
            raise RuntimeError("run() must complete before this lifecycle stage.")
        return self.artifacts

    def _validation_checks(self, raw: pd.DataFrame) -> dict[str, object]:
        """Report sweep-specific adequacy alongside basic shape checks."""

        levels = self.config.simulation.heart_failure_effect_levels_sd
        expected_rows = (
            len(levels) * self.config.repeats_per_level * len(self.models)
        )
        redundant_sizes = raw.get("competing_subgroup_size__redundant")
        checks: dict[str, object] = {
            "metric_row_count_matches": int(raw.shape[0]) == expected_rows,
            "hf_levels_present": sorted(set(raw["heart_failure_effect_sd"]))
            == sorted(levels),
            "oracle_absent": kernel.ORACLE not in set(raw["method"]),
            "subgroup_columns_present": all(
                f"false_atrial__{name}" in raw.columns for name in _SUBGROUP_NAMES
            ),
        }
        if redundant_sizes is not None:
            checks["redundant_subgroup_median_size"] = float(redundant_sizes.median())
            checks["redundant_subgroup_analyzable"] = bool(
                redundant_sizes.median() >= 30
            )
        checks["all_required_checks_pass"] = bool(
            checks["metric_row_count_matches"]
            and checks["hf_levels_present"]
            and checks["oracle_absent"]
            and checks["subgroup_columns_present"]
        )
        return checks
