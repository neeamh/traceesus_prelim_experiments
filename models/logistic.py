"""Supervised logistic comparators with explicit feature boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from traceesus.core.model import FittedModel, Model
from traceesus.core.simulator import Cohort, SupervisedCohort
from traceesus.experiments.model_comparison import kernel


@dataclass(frozen=True)
class FittedLogisticModel(FittedModel):
    """L2 logistic fit plus the feature contract used during training."""

    fit_result: kernel.LogisticModel
    include_kidney_status: bool

    def _features(self, data: Cohort) -> np.ndarray:
        if not self.include_kidney_status:
            return data.biomarkers
        return np.column_stack(
            (data.biomarkers, data.covariate("renal_dysfunction"))
        )

    def posterior(self, data: Cohort) -> np.ndarray:
        """Return two-class probabilities in atrial, competing column order."""

        atrial = kernel.logistic_atrial_probability(
            self.fit_result,
            self._features(data),
        )
        return np.column_stack((atrial, 1.0 - atrial))

    @property
    def n_parameters(self) -> int:
        """Count the fitted intercept and standardized feature coefficients."""

        return int(self.fit_result.coefficients.size)


class _BaseLogisticModel(Model):
    include_kidney_status: bool

    def _features(self, data: Cohort) -> np.ndarray:
        if not self.include_kidney_status:
            return data.biomarkers
        return np.column_stack(
            (data.biomarkers, data.covariate("renal_dysfunction"))
        )

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: kernel.ComparisonConfig,
    ) -> FittedLogisticModel:
        """Fit supervised labels; Newton fitting itself consumes no RNG draws."""

        if not isinstance(data, SupervisedCohort):
            raise TypeError("Supervised logistic fitting requires SupervisedCohort.")
        result = kernel.fit_logistic_regression(
            self._features(data),
            data.require_training_labels(),
            config,
        )
        return FittedLogisticModel(result, self.include_kidney_status)


class BiomarkersOnlyLogisticModel(_BaseLogisticModel):
    """Supervised associative comparator with biomarkers and no renal input."""

    name = kernel.ASSOCIATIVE_BIOMARKERS
    include_kidney_status = False


class KidneyAdjustedLogisticModel(_BaseLogisticModel):
    """Supervised associative control with biomarkers plus observed renal status."""

    name = kernel.ASSOCIATIVE_ADJUSTED
    include_kidney_status = True
