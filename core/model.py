"""Uniform model and fitted-model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from typing import Mapping

import numpy as np
from numpy.random import Generator

from .config import ValidatedConfig
from .simulator import Cohort


@dataclass(frozen=True)
class FitDiagnostics:
    """Optional fit evidence kept outside the universal posterior contract.

    Convergence metadata is useful for the proposal-locked EM models, but it is
    not a prerequisite for comparing a new model on paired cohorts. Keeping it
    optional prevents an ablation model from having to imitate kernel-specific
    ``fit_result`` internals merely to participate in posterior evaluation.
    """

    converged: bool
    iterations: int
    best_start: int
    log_likelihood: float
    effective_class_fraction: np.ndarray
    anchor_margin: float


class FittedModel(ABC):
    """A fitted model that can answer the same posterior query contract."""

    @abstractmethod
    def posterior(self, data: Cohort) -> np.ndarray:
        """Return an ``(n, K)`` posterior without consulting evaluation truth."""

    def counterfactual_scores(self, data: Cohort) -> Mapping[str, np.ndarray]:
        """Return optional causal-query scores, or fail explicitly if unsupported."""

        raise NotImplementedError("This fitted model has no counterfactual query.")

    def fit_diagnostics(self) -> FitDiagnostics | None:
        """Return optional convergence evidence without burdening new models.

        Posterior evaluation is the common scientific interface. Models that
        do not use an iterative fit, or do not expose comparable diagnostics,
        intentionally return ``None`` and remain valid registry entries.
        """

        return None

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Report the exact free-parameter count used by model selection."""


class Model(ABC):
    """Fit a model from observed cohort fields using an assigned RNG stream."""

    name: str

    @abstractmethod
    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: ValidatedConfig,
    ) -> FittedModel:
        """Fit with validated controls and without reading simulator truth."""


def assert_truth_free_fit_interfaces(models: list[Model] | tuple[Model, ...]) -> None:
    """Prove that unsupervised fit signatures cannot accept simulator truth.

    The cohort type enforces the boundary structurally.  This second check makes
    the proposal's validation claim executable instead of relying only on a
    hard-coded JSON flag.
    """

    forbidden = {"truth", "true_mechanism", "mechanism_labels", "labels"}
    for model in models:
        parameters = set(inspect.signature(model.fit).parameters)
        leaked = sorted(parameters & forbidden)
        if leaked:
            raise AssertionError(
                f"{model.name!r} fit interface accepts truth-like fields: {leaked}"
            )
