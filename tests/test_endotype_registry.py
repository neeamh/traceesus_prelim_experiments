"""Focused contract tests for the extensible endotype model registry."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import Generator

from configs.endotype_discovery import CONFIG
from configs.arcadia_calibrated import CONFIG as ARCADIA_CONFIG
from traceesus.core.model import FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.experiments.endotype_discovery.recovery import run_model_registry
from traceesus.models.oracle import DataGeneratingOracle
from traceesus.models.multi_nuisance import (
    BIOLOGY_MASK,
    TwoNuisanceCausalSCM,
    TwoNuisanceCounterfactualSCM,
)
from traceesus.registry import (
    FULL_LADDER,
    LOCKED_MODEL_SET,
    MODEL_LADDER,
    full_ladder_for_config,
)


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


def test_model_sets_are_ordered_and_keep_legacy_causal_distinct() -> None:
    """Lock the requested R1--R6 ladder and the separate four-row v3 set."""

    assert tuple(code for code, _ in MODEL_LADDER) == (
        "R1", "R2", "R3", "R4", "R5", "R6"
    )
    assert len(FULL_LADDER.names) == 6
    assert len(LOCKED_MODEL_SET.names) == 4
    assert LOCKED_MODEL_SET.names[:2] == FULL_LADDER.names[:2]
    assert LOCKED_MODEL_SET.names[-1] == FULL_LADDER.names[-1]
    assert LOCKED_MODEL_SET.names[2] not in FULL_LADDER.names


def test_biology_mask_is_config_driven_without_moving_locked_defaults() -> None:
    """Use the historical mask by default and the ARCADIA override only there."""

    assert CONFIG.simulation.biology_path_mask == BIOLOGY_MASK
    locked = full_ladder_for_config(CONFIG)
    arcadia = full_ladder_for_config(ARCADIA_CONFIG)
    locked_r4, locked_r5 = locked.fitted_models[3:5]
    arcadia_r4, arcadia_r5 = arcadia.fitted_models[3:5]
    assert isinstance(locked_r4, TwoNuisanceCausalSCM)
    assert isinstance(locked_r5, TwoNuisanceCounterfactualSCM)
    assert locked_r4.biology_path_mask == BIOLOGY_MASK
    assert locked_r5.biology_path_mask == BIOLOGY_MASK
    assert arcadia_r4.biology_path_mask == ARCADIA_CONFIG.simulation.biology_path_mask
    assert arcadia_r5.biology_path_mask == ARCADIA_CONFIG.simulation.biology_path_mask
    assert arcadia_r4.biology_path_mask == (
        (True, False, False),
        (True, True, False),
    )
    assert locked.parameter_counts[3:5] == (14, 14)
    assert arcadia.parameter_counts[3:5] == (15, 15)


def test_spawning_appended_children_preserves_existing_streams() -> None:
    """Prove NumPy child streams 0..7 are invariant when four are appended."""

    for seed in (0, CONFIG.master_seed, 2**63 + 17):
        original = np.random.SeedSequence(seed).spawn(8)
        extended = np.random.SeedSequence(seed).spawn(12)
        for left, right in zip(original, extended[:8], strict=True):
            np.testing.assert_array_equal(
                left.generate_state(16, dtype=np.uint64),
                right.generate_state(16, dtype=np.uint64),
            )


def test_full_ladder_preserves_shared_rows_and_only_adds_new_names() -> None:
    """Compare both model sets through the same paired runner on a small design."""

    simulation = replace(
        CONFIG.simulation,
        training_patients=80,
        test_patients=90,
        renal_effect_levels_sd=(0.5,),
        renal_effect_labels=("Weak",),
    )
    config = replace(
        CONFIG,
        repeats_per_level=2,
        workers=1,
        simulation=simulation,
    )
    locked, _, _ = run_model_registry(config, LOCKED_MODEL_SET.fitted_models)
    full, _, _ = run_model_registry(config, FULL_LADDER.fitted_models)
    shared = set(LOCKED_MODEL_SET.names) & set(FULL_LADDER.names)
    keys = ["repeat", "renal_effect_sd", "method"]
    locked_shared = locked[locked["method"].isin(shared)].sort_values(keys)
    full_shared = full[full["method"].isin(shared)].sort_values(keys)
    pd.testing.assert_frame_equal(
        locked_shared.reset_index(drop=True),
        full_shared.reset_index(drop=True),
        check_exact=True,
    )
    added = set(full["method"]) - set(locked["method"])
    assert added == set(FULL_LADDER.names) - set(LOCKED_MODEL_SET.names)
