"""Renal-adjusted associative latent-class model adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from traceesus.core.model import FitDiagnostics, FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.experiments.endotype_discovery import kernel


@dataclass(frozen=True)
class FittedAdjustedLatentClassModel(FittedModel):
    """Conditional mixture with freely estimated renal slopes for every marker."""

    fit_result: kernel.ConditionalLatentFit

    def posterior(self, data: Cohort) -> np.ndarray:
        """Evaluate the fitted conditional posterior on observed data."""

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
        """Return the proposal-locked flexible K=2 count."""

        return 14


class AdjustedLatentClassModel(Model):
    """Fit p(Z|R)p(B|Z,R) with a renal association for every biomarker."""

    name = kernel.ASSOCIATIVE_ADJUSTED

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: kernel.FittingConfig,
    ) -> FittedAdjustedLatentClassModel:
        """Use the exact conditional EM kernel with an all-true path mask."""

        mask = np.ones(data.biomarkers.shape[1], dtype=bool)
        result = kernel.fit_conditional_latent_model(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            mask,
            rng,
            config,
        )
        return FittedAdjustedLatentClassModel(result)
