"""Scientific defaults for the preliminary known-SCM query experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traceesus.core.config import ValidatedConfig
from traceesus.core.seeds import ENDOTYPE_RECOVERY_SEED_ROOT


@dataclass(frozen=True)
class ExperimentConfig(ValidatedConfig):
    """All prespecified parameters for the preliminary experiment."""

    seed: int = ENDOTYPE_RECOVERY_SEED_ROOT
    patients_per_repeat: int = 1_000
    repeats_per_level: int = 500
    renal_prevalence: float = 0.30
    atrial_log_odds_when_renal_normal: float = 0.0
    renal_to_atrial_log_odds: float = -0.40
    confounding_strengths_sd: tuple[float, ...] = (0.0, 0.75, 1.50, 2.25)
    confounding_labels: tuple[str, ...] = (
        "None\n(0 SD)",
        "Weak\n(0.75 SD)",
        "Moderate\n(1.50 SD)",
        "Strong\n(2.25 SD)",
    )
    # Rows: atrial mechanism, competing mechanism.
    # Columns: NT-proBNP-like, atrial electrical, competing-mechanism marker.
    mechanism_effects: tuple[tuple[float, ...], ...] = (
        (1.20, 0.80, 0.00),
        (0.00, 0.00, 1.00),
    )
    biomarker_noise_sd: tuple[float, ...] = (1.00, 1.00, 1.00)
    counterfactual_disablement_weight: float = 0.50
    counterfactual_sufficiency_weight: float = 0.50
    null_repeats: int = 500
    null_patients_per_repeat: int = 1_000
    null_renal_effect_sd: float = 2.25
    null_gmm_starts: int = 5
    null_gmm_max_iter: int = 400
    null_min_component_weight: float = 0.10
    gmm_variance_floor: float = 0.05

    def __post_init__(self) -> None:
        """Reject invalid controls before any proposal-locked seed is spawned."""

        self.validate()

    def validate(self) -> None:
        """Reject invalid controls before they can alter a locked RNG trajectory."""

        if self.patients_per_repeat < 2:
            raise ValueError("patients_per_repeat must be at least 2.")
        if self.repeats_per_level < 2:
            raise ValueError("repeats_per_level must be at least 2.")
        if self.null_repeats < 2:
            raise ValueError("null_repeats must be at least 2.")
        if len(self.confounding_strengths_sd) != len(self.confounding_labels):
            raise ValueError("Each confounding strength needs one display label.")
        if not 0.0 < self.renal_prevalence < 1.0:
            raise ValueError("renal_prevalence must be strictly between 0 and 1.")
        weights = (
            self.counterfactual_disablement_weight,
            self.counterfactual_sufficiency_weight,
        )
        if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("Counterfactual weights must be nonnegative and sum to 1.")
        effects = np.asarray(self.mechanism_effects, dtype=float)
        noise = np.asarray(self.biomarker_noise_sd, dtype=float)
        if effects.shape != (2, 3) or noise.shape != (3,):
            raise ValueError("This preliminary experiment requires 2 mechanisms and 3 biomarkers.")
        if np.any(noise <= 0):
            raise ValueError("All biomarker noise standard deviations must be positive.")


# This immutable object reproduces the proposal-cited main and K=1-null runs.
CONFIG = ExperimentConfig()

__all__ = ["CONFIG", "ExperimentConfig"]
