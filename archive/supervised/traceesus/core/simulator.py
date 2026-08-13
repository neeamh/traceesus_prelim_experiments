"""Simulation contracts with structural truth separation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.random import Generator


@dataclass(frozen=True)
class Cohort:
    """Fit-facing measurements and covariates, deliberately excluding truth.

    Models receive this object.  Mechanism labels live in ``SimulationTruth`` so
    an unsupervised fit cannot accidentally consume them through a convenience
    dataframe or a broad simulator result.
    """

    biomarkers: np.ndarray
    covariates: Mapping[str, np.ndarray]
    measurement_indicators: np.ndarray | None = None

    def covariate(self, name: str) -> np.ndarray:
        """Return a named observed covariate without exposing simulator truth."""

        try:
            return self.covariates[name]
        except KeyError as error:
            raise KeyError(f"Observed covariate {name!r} is unavailable.") from error


@dataclass(frozen=True)
class SupervisedCohort(Cohort):
    """Fit-facing cohort with an explicit supervised target.

    This type is used only by the model-comparison experiment. Naming the field
    ``training_labels`` keeps the supervised scientific boundary visible while
    evaluation truth stays out of every unsupervised ``Cohort``.
    """

    training_labels: np.ndarray | None = None

    def require_training_labels(self) -> np.ndarray:
        """Return labels or reject accidental use on an unlabeled cohort."""

        if self.training_labels is None:
            raise ValueError("This supervised model requires training_labels.")
        return self.training_labels


@dataclass(frozen=True)
class SimulationTruth:
    """Simulator-only labels used after fitting for controlled evaluation."""

    mechanism: np.ndarray


@dataclass(frozen=True)
class SimulatedData:
    """Pair observed fit inputs with evaluation-only simulator truth."""

    observed: Cohort
    truth: SimulationTruth


class Simulator(ABC):
    """Generate a cohort while keeping fit inputs separate from truth labels."""

    @abstractmethod
    def simulate(self, rng: Generator, patient_count: int) -> SimulatedData:
        """Consume the supplied generator in the simulator's historical order."""
