"""Biologically constrained latent structural model adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from traceesus.core.model import FitDiagnostics, FittedModel, Model
from traceesus.core.simulator import Cohort, SupervisedCohort
from traceesus.experiments.endotype_discovery import kernel
from traceesus.experiments.model_comparison import kernel as comparison_kernel


@dataclass(frozen=True)
class FittedBiologicallyConstrainedCausalSCM(FittedModel):
    """Conditional latent fit whose direct renal path is prespecified to NT only."""

    fit_result: kernel.ConditionalLatentFit

    def posterior(self, data: Cohort) -> np.ndarray:
        """Rank latent mechanisms after modeling the permitted renal path."""

        return kernel.conditional_posterior(
            self.fit_result,
            data.biomarkers,
            data.covariate("renal_dysfunction"),
        )

    def fit_diagnostics(self) -> FitDiagnostics:
        """Expose the historical EM audit fields without changing fit arithmetic."""

        result = self.fit_result
        return FitDiagnostics(
            converged=result.converged,
            iterations=result.iterations,
            best_start=result.best_start,
            log_likelihood=result.log_likelihood,
            effective_class_fraction=result.effective_class_fraction,
            anchor_margin=result.anchor_margin,
        )

    @property
    def n_parameters(self) -> int:
        """Return the K=2 count matched to the primary associative model."""

        return 12


class BiologicallyConstrainedCausalSCM(Model):
    """Fit the latent SCM with only the prespecified renal-to-NT direct path."""

    name = kernel.CAUSAL_SCM

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: kernel.FittingConfig,
    ) -> FittedBiologicallyConstrainedCausalSCM:
        """Run the exact conditional EM sequence with the biological path mask."""

        mask = np.zeros(data.biomarkers.shape[1], dtype=bool)
        mask[kernel.Biomarker.NT_PROBNP_LIKE] = True
        result = kernel.fit_conditional_latent_model(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            mask,
            rng,
            config,
        )
        return FittedBiologicallyConstrainedCausalSCM(result)


@dataclass(frozen=True)
class FittedSupervisedStructuralCausalModel(FittedModel):
    """Fitted supervised SCM used only in the direct model comparison."""

    fit_result: comparison_kernel.FittedSCM
    config: comparison_kernel.ComparisonConfig

    def posterior(self, data: Cohort) -> np.ndarray:
        """Return the same-SCM posterior diagnostic on observed data."""

        return comparison_kernel.scm_posterior(
            self.fit_result,
            data.biomarkers,
            data.covariate("renal_dysfunction"),
        )

    def counterfactual_scores(self, data: Cohort) -> dict[str, np.ndarray]:
        """Return posterior-integrated disablement and sufficiency scores.

        In this symmetric K=2 model the scores are ranking-equivalent to the
        same-SCM posterior up to floating-point noise; they do not establish
        incremental causal-query value.
        """

        return comparison_kernel.scm_counterfactual_scores(
            self.fit_result,
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            self.config,
        )

    @property
    def n_parameters(self) -> int:
        """Count structural means, renal slopes, residual SDs, and prior terms."""

        return 14


class SupervisedStructuralCausalModel(Model):
    """Fit the prespecified SCM with true mechanism labels during training."""

    name = comparison_kernel.SCM_COUNTERFACTUAL

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: comparison_kernel.ComparisonConfig,
    ) -> FittedSupervisedStructuralCausalModel:
        """Fit structural equations from explicitly supervised training labels."""

        if not isinstance(data, SupervisedCohort):
            raise TypeError("The supervised SCM requires SupervisedCohort.")
        result = comparison_kernel.fit_structural_causal_model(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            data.require_training_labels(),
            config,
        )
        return FittedSupervisedStructuralCausalModel(result, config)
