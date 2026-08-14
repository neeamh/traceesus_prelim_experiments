"""Fit the one-path biologically constrained latent structural model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from configs.endotype_discovery import FittingConfig
from traceesus.core.markers import Biomarker
from traceesus.core.model import FitDiagnostics, FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.models.adjusted_lcm import (
    ConditionalLatentFit,
    conditional_posterior,
    fit_conditional_latent_model,
)


CAUSAL_SCM = "Biologically constrained latent SCM"


@dataclass(frozen=True)
class FittedBiologicallyConstrainedCausalSCM(FittedModel):
    """Conditional latent fit whose direct renal path is restricted to NT."""

    fit_result: ConditionalLatentFit

    def posterior(self, data: Cohort) -> np.ndarray:
        return conditional_posterior(
            self.fit_result, data.biomarkers, data.covariate("renal_dysfunction")
        )

    def fit_diagnostics(self) -> FitDiagnostics:
        result = self.fit_result
        return FitDiagnostics(
            result.converged,
            result.iterations,
            result.best_start,
            result.log_likelihood,
            result.effective_class_fraction,
            result.anchor_margin,
        )

    @property
    def n_parameters(self) -> int:
        return 12


@dataclass(frozen=True)
class BiologicallyConstrainedCausalSCM(Model):
    """Fit the latent SCM with only the prespecified renal-to-NT direct path."""

    name: str = CAUSAL_SCM

    def fit(
        self, data: Cohort, rng: Generator, config: FittingConfig
    ) -> FittedBiologicallyConstrainedCausalSCM:
        mask = np.zeros(data.biomarkers.shape[1], dtype=bool)
        mask[Biomarker.NT_PROBNP] = True
        result = fit_conditional_latent_model(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            mask,
            rng,
            config,
        )
        return FittedBiologicallyConstrainedCausalSCM(result)


__all__ = [
    "BiologicallyConstrainedCausalSCM",
    "CAUSAL_SCM",
    "FittedBiologicallyConstrainedCausalSCM",
]
