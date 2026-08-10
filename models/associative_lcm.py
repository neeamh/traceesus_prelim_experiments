"""Associative latent-class model adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from traceesus.core.model import FitDiagnostics, FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.experiments.endotype_discovery import kernel


@dataclass(frozen=True)
class FittedAssociativeLatentClassModel(FittedModel):
    """Expose the frozen associative EM fit through the common posterior API."""

    fit_result: kernel.AssociativeLatentClassFit

    def posterior(self, data: Cohort) -> np.ndarray:
        """Evaluate p(Z | biomarkers, renal) with the fitted associative model."""

        return kernel.associative_posterior(
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
        """Return the proposal-locked K=2 count used by BIC."""

        return 12


class AssociativeLatentClassModel(Model):
    """Fit p(Z)p(R|Z)prod_j p(B_j|Z) without mechanism labels."""

    name = kernel.ASSOCIATIVE_LCA

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: kernel.FittingConfig,
    ) -> FittedAssociativeLatentClassModel:
        """Run the exact multi-start EM sequence on observed fields only."""

        result = kernel.fit_associative_latent_class_model(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            rng,
            config,
        )
        return FittedAssociativeLatentClassModel(result)
