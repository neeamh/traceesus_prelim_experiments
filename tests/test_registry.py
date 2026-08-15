"""Contracts for named seeds, design rows, and the Monte Carlo target."""

from __future__ import annotations

import numpy as np
import pytest

from traceesus.core import seeds
from traceesus.core.stats import (
    TARGET_MONTE_CARLO_STANDARD_ERROR,
    required_repeats_for_target_mcse,
)
from traceesus.registry import (
    ACTIVE_EXPERIMENT_DESIGNS,
    EXPERIMENT_DESIGNS,
    design_table,
)


def test_seed_roots_are_named_and_unique() -> None:
    """Mirror the import-time collision guard with a readable failure."""

    roots = tuple(
        value
        for name, value in vars(seeds).items()
        if name.endswith("_SEED_ROOT")
    )
    assert len(roots) == 10
    assert len(roots) == len(set(roots))


def test_design_table_is_generated_from_registry() -> None:
    """Keep code and documentation at the same experiment grain."""

    table = design_table()
    assert table["experiment"].tolist() == [
        design.name for design in EXPERIMENT_DESIGNS
    ]
    assert set(table.columns) == {
        "experiment", "config", "n_train", "n_test", "repeats", "seed_root",
        "evaluation_cohort", "status", "output_directory",
    }
    assert tuple(design.name for design in ACTIVE_EXPERIMENT_DESIGNS) == (
        "endotype_discovery", "transportability"
    )
    drift = table.loc[table["experiment"] == "identity_drift"].iloc[0]
    assert drift["evaluation_cohort"] == "held-out"
    assert drift["status"] == "exploratory"


def test_required_repeats_uses_the_single_mcse_target() -> None:
    """Convert pilot variance to repeats without altering configured designs."""

    assert TARGET_MONTE_CARLO_STANDARD_ERROR == 0.005
    assert required_repeats_for_target_mcse(0.01) == 400
    assert required_repeats_for_target_mcse(0.0) == 2
    with pytest.raises(ValueError):
        required_repeats_for_target_mcse(np.nan)
