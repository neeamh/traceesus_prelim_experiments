"""Focused contract tests for the extensible endotype model registry."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.random import Generator

from configs.endotype_discovery import CONFIG
from traceesus.core.model import FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.experiments.endotype_discovery.recovery import run_model_registry
from traceesus.models.oracle import DataGeneratingOracle


class _FittedMinimalModel(FittedModel):
    """Provide only the universal fitted-model contract, with no diagnostics."""

    def posterior(self, data: Cohort) -> np.ndarray:
        """Return a deterministic valid K=2 posterior for every patient."""

        probabilities = np.empty((data.biomarkers.shape[0], 2), dtype=float)
        probabilities[:, 0] = 0.4
        probabilities[:, 1] = 0.6
        return probabilities

    @property
    def n_parameters(self) -> int:
        """Report the deliberately parameter-free test model."""

        return 0


class _MinimalModel(Model):
    """Exercise registry genericity without a kernel-specific fit_result field."""

    name = "Minimal posterior model"

    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: Any,
    ) -> _FittedMinimalModel:
        """Fit once per cohort while intentionally exposing no diagnostics."""

        self.fit_calls += 1
        return _FittedMinimalModel()


def test_minimal_model_runs_without_fit_result_or_diagnostics() -> None:
    """A conforming new row must need no undocumented adapter attributes."""

    simulation = replace(
        CONFIG.simulation,
        training_patients=50,
        test_patients=50,
        renal_effect_levels_sd=(0.0,),
        renal_effect_labels=("None",),
    )
    config = replace(
        CONFIG,
        repeats_per_level=2,
        workers=1,
        simulation=simulation,
    )
    model = _MinimalModel()

    metrics, diagnostics, parameters = run_model_registry(config, [model])

    assert model.fit_calls == config.repeats_per_level
    assert metrics["method"].tolist() == [
        model.name,
        DataGeneratingOracle.name,
        model.name,
        DataGeneratingOracle.name,
    ]
    assert diagnostics.empty
    assert parameters.columns.tolist() == ["repeat", "renal_effect_sd"]
    assert parameters["repeat"].tolist() == [0, 1]

