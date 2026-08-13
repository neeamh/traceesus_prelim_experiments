"""Scientific defaults for the supervised associative-versus-SCM comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traceesus.core.config import ValidatedConfig


@dataclass(frozen=True)
class ComparisonConfig(ValidatedConfig):
    """Freeze every supervised-comparison control that can alter cited results.

    Keeping these values together makes the scientific boundary explicit: all
    three methods are trained with known mechanism labels, so this experiment
    evaluates supervised classification rather than endotype discovery.
    """

    seed: int = 20_260_729
    training_patients: int = 3_000
    test_patients: int = 1_000
    repeats_per_level: int = 500
    confounding_strengths_sd: tuple[float, ...] = (0.0, 0.75, 1.50, 2.25)
    confounding_labels: tuple[str, ...] = (
        "None\n(0 SD)",
        "Weak\n(0.75 SD)",
        "Moderate\n(1.50 SD)",
        "Strong\n(2.25 SD)",
    )
    logistic_l2_penalty: float = 1.0
    logistic_max_iter: int = 50
    logistic_tolerance: float = 1e-8
    scm_prior_smoothing: float = 0.5
    variance_floor: float = 1e-6
    counterfactual_disablement_weight: float = 0.50
    counterfactual_sufficiency_weight: float = 0.50

    def __post_init__(self) -> None:
        """Reject invalid controls before any proposal-locked seed is spawned."""

        self.validate()

    def validate(self) -> None:
        """Fail before seed derivation when a control violates the locked design."""

        if self.training_patients < 10 or self.test_patients < 10:
            raise ValueError("Training and test cohorts must each contain at least 10 patients.")
        if self.repeats_per_level < 2:
            raise ValueError("At least two repeats are required.")
        if len(self.confounding_strengths_sd) != len(self.confounding_labels):
            raise ValueError("Each renal-effect level requires one label.")
        if self.logistic_l2_penalty < 0:
            raise ValueError("The L2 penalty cannot be negative.")
        if self.scm_prior_smoothing <= 0:
            raise ValueError("SCM prior smoothing must be positive.")
        weights = (
            self.counterfactual_disablement_weight,
            self.counterfactual_sufficiency_weight,
        )
        if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("Counterfactual weights must be nonnegative and sum to one.")


# This immutable object reproduces the proposal-cited 500-repeat configuration.
CONFIG = ComparisonConfig()

__all__ = ["CONFIG", "ComparisonConfig"]
