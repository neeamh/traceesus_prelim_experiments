"""Declare the model ladder and reproducible experiment designs in one place."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from configs.endotype_discovery import CONFIG as ENDOTYPE_CONFIG
from configs.transportability import CONFIG as TRANSPORT_CONFIG
from traceesus.core.config import ValidatedConfig
from traceesus.core.model import Model
from traceesus.core.seeds import (
    CONFOUNDING_SWEEP_SEED_ROOT,
    ENDOTYPE_RECOVERY_SEED_ROOT,
    HF_GRID_SEED_ROOT,
    IDENTITY_DRIFT_SEED_ROOT,
    REDUNDANCY_SWEEP_SEED_ROOT,
    TRANSPORT_SEED_ROOT,
)
from traceesus.models import (
    AdjustedLatentClassModel,
    AssociativeLatentClassModel,
    BiologicallyConstrainedCausalSCM,
    TwoNuisanceAdjustedLCM,
    TwoNuisanceCausalSCM,
    TwoNuisanceCounterfactualSCM,
)
from traceesus.models.oracle import ORACLE


@dataclass(frozen=True)
class EndotypeModelSet:
    """Ordered fitted rows followed by the deterministic oracle ceiling."""

    fitted_models: tuple[Model, ...]
    parameter_counts: tuple[int, ...]
    contrast_reference: str
    uses_legacy_parameter_check: bool

    def __post_init__(self) -> None:
        if len(self.fitted_models) != len(self.parameter_counts):
            raise ValueError("Each fitted model requires one parameter count.")

    @property
    def names(self) -> tuple[str, ...]:
        """Return the exact output order, including the oracle last."""

        return tuple(model.name for model in self.fitted_models) + (ORACLE,)


LOCKED_MODEL_SET = EndotypeModelSet(
    (
        AssociativeLatentClassModel(),
        AdjustedLatentClassModel(),
        BiologicallyConstrainedCausalSCM(),
    ),
    (12, 14, 12),
    BiologicallyConstrainedCausalSCM.name,
    True,
)

FULL_LADDER = EndotypeModelSet(
    (
        AssociativeLatentClassModel(),
        AdjustedLatentClassModel(),
        TwoNuisanceAdjustedLCM(),
        TwoNuisanceCausalSCM(),
        TwoNuisanceCounterfactualSCM(),
    ),
    (12, 14, 18, 14, 14),
    TwoNuisanceCausalSCM.name,
    False,
)

MODEL_LADDER = tuple(zip(("R1", "R2", "R3", "R4", "R5", "R6"), FULL_LADDER.names))


@dataclass(frozen=True)
class ExperimentDesign:
    """One auditable experiment row used by execution and documentation."""

    name: str
    config: ValidatedConfig
    n_train: int
    n_test: int
    repeats: int
    seed_root: int
    evaluation_cohort: str
    status: str
    output_directory: str
    description: str

    def __post_init__(self) -> None:
        if self.evaluation_cohort not in {"held-out", "in-sample"}:
            raise ValueError("evaluation_cohort must be held-out or in-sample.")
        if self.status not in {"locked", "exploratory"}:
            raise ValueError("status must be locked or exploratory.")


EXPERIMENT_DESIGNS = (
    ExperimentDesign(
        "endotype_discovery",
        ENDOTYPE_CONFIG,
        ENDOTYPE_CONFIG.simulation.training_patients,
        ENDOTYPE_CONFIG.simulation.test_patients,
        ENDOTYPE_CONFIG.repeats_per_level,
        ENDOTYPE_RECOVERY_SEED_ROOT,
        "held-out",
        "locked",
        "outputs_latent_endotyping",
        "Unsupervised latent endotype recovery plus the K=1 null control.",
    ),
    ExperimentDesign(
        "transportability",
        TRANSPORT_CONFIG,
        len(TRANSPORT_CONFIG.source_hospitals)
        * TRANSPORT_CONFIG.simulation.source_patients_per_hospital,
        TRANSPORT_CONFIG.simulation.target_test_patients,
        TRANSPORT_CONFIG.repeats,
        TRANSPORT_SEED_ROOT,
        "held-out",
        "locked",
        "outputs_transportability",
        "Unlabeled cross-hospital transport, shifts, and ablations.",
    ),
    ExperimentDesign(
        "redundancy_sweep", ENDOTYPE_CONFIG,
        ENDOTYPE_CONFIG.simulation.training_patients,
        ENDOTYPE_CONFIG.simulation.test_patients,
        ENDOTYPE_CONFIG.repeats_per_level,
        REDUNDANCY_SWEEP_SEED_ROOT, "held-out", "exploratory",
        "outputs_redundancy_sweep", "Heart-failure redundancy sweep.",
    ),
    ExperimentDesign(
        "hf_grid", ENDOTYPE_CONFIG,
        ENDOTYPE_CONFIG.simulation.training_patients,
        ENDOTYPE_CONFIG.simulation.test_patients,
        100, HF_GRID_SEED_ROOT,
        "held-out", "exploratory", "outputs_hf_grid",
        "Renal by heart-failure latent recovery grid.",
    ),
    ExperimentDesign(
        "identity_drift", ENDOTYPE_CONFIG,
        ENDOTYPE_CONFIG.simulation.training_patients,
        ENDOTYPE_CONFIG.simulation.test_patients,
        40,
        IDENTITY_DRIFT_SEED_ROOT, "held-out", "exploratory", "outputs_hf_grid",
        "Pooled-class identity drift under renal distortion.",
    ),
    ExperimentDesign(
        "confounding_sweep", ENDOTYPE_CONFIG,
        ENDOTYPE_CONFIG.simulation.training_patients,
        ENDOTYPE_CONFIG.simulation.test_patients,
        150,
        CONFOUNDING_SWEEP_SEED_ROOT, "held-out", "exploratory",
        "outputs_confounding", "Renal-to-mechanism confounding sweep.",
    ),
)

ACTIVE_EXPERIMENT_DESIGNS = tuple(
    design for design in EXPERIMENT_DESIGNS if design.status == "locked"
)


def design_table() -> pd.DataFrame:
    """Generate the design summary directly from immutable registry rows."""

    return pd.DataFrame(
        {
            "experiment": design.name,
            "config": design.config.__class__.__module__,
            "n_train": design.n_train,
            "n_test": design.n_test,
            "repeats": design.repeats,
            "seed_root": design.seed_root,
            "evaluation_cohort": design.evaluation_cohort,
            "status": design.status,
            "output_directory": design.output_directory,
        }
        for design in EXPERIMENT_DESIGNS
    )


__all__ = (
    "ACTIVE_EXPERIMENT_DESIGNS",
    "EndotypeModelSet",
    "EXPERIMENT_DESIGNS",
    "ExperimentDesign",
    "FULL_LADDER",
    "LOCKED_MODEL_SET",
    "MODEL_LADDER",
    "design_table",
)
