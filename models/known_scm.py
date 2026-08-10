"""Uniform model adapters for prespecified, unfitted preliminary SCM queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from traceesus.core.model import FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.experiments.counterfactual import kernel


@dataclass(frozen=True)
class FittedKnownKidneyBlindPosterior(FittedModel):
    """Known DGP parameters queried after deliberately omitting the renal path."""

    renal_effect_sd: float
    config: kernel.ExperimentConfig

    def posterior(self, data: Cohort) -> np.ndarray:
        """Return the exact kidney-blind resemblance posterior."""

        return kernel.kidney_blind_posterior(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            self.renal_effect_sd,
            self.config,
        )

    @property
    def n_parameters(self) -> int:
        """Return zero because this experiment estimates no model parameters."""

        return 0


@dataclass(frozen=True)
class KnownKidneyBlindPosteriorModel(Model):
    """Expose the legacy misspecified posterior through the common interface."""

    renal_effect_sd: float
    name = kernel.METHOD_POSTERIOR_BLIND

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: kernel.ExperimentConfig,
    ) -> FittedKnownKidneyBlindPosterior:
        """Return the prespecified model without consuming RNG or evaluation truth."""

        return FittedKnownKidneyBlindPosterior(self.renal_effect_sd, config)


@dataclass(frozen=True)
class FittedKnownStructuralCausalModel(FittedModel):
    """Known renal-aware SCM supporting posterior and causal-query diagnostics."""

    renal_effect_sd: float
    config: kernel.ExperimentConfig

    def posterior(self, data: Cohort) -> np.ndarray:
        """Return the exact same-SCM Bayes posterior fairness diagnostic."""

        return kernel.kidney_aware_posterior(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            self.renal_effect_sd,
            self.config,
        )

    def counterfactual_scores(self, data: Cohort) -> dict[str, np.ndarray]:
        """Return posterior-integrated sufficiency and disablement scores.

        In the symmetric K=2 toy model these normalized scores are monotone
        transformations of the posterior, which prevents interpreting them as
        an accuracy improvement over the correctly specified Bayes classifier.
        """

        return kernel.posterior_integrated_counterfactual_scores(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            self.renal_effect_sd,
            self.config,
        )

    @property
    def n_parameters(self) -> int:
        """Return zero because the data-generating SCM is prespecified, not fitted."""

        return 0


@dataclass(frozen=True)
class KnownStructuralCausalModel(Model):
    """Expose the known renal-aware DGP without implying parameter estimation."""

    renal_effect_sd: float
    name = "Known structural causal model"

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: kernel.ExperimentConfig,
    ) -> FittedKnownStructuralCausalModel:
        """Return the prespecified SCM without consuming RNG or evaluation truth."""

        return FittedKnownStructuralCausalModel(self.renal_effect_sd, config)


__all__ = [
    "FittedKnownKidneyBlindPosterior",
    "FittedKnownStructuralCausalModel",
    "KnownKidneyBlindPosteriorModel",
    "KnownStructuralCausalModel",
]
