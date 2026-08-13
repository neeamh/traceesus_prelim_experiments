"""Scientific defaults for unsupervised latent endotype discovery."""

from __future__ import annotations

from dataclasses import dataclass

from traceesus.core.config import ValidatedConfig


_BIOMARKER_COUNT = 3


@dataclass(frozen=True)
class SimulationConfig(ValidatedConfig):
    """Parameters of the transparent synthetic data-generating process.

    Effects are expressed in within-biomarker residual standard deviations.
    The chosen values create moderate, imperfect separation at zero renal
    distortion; the Bayes error is deliberately nonzero.
    """

    training_patients: int = 800
    test_patients: int = 1_000
    renal_dysfunction_prevalence: float = 0.30
    atrial_probability_if_renal_normal: float = 0.50
    atrial_probability_if_renal_impaired: float = 0.50
    atrial_path_effects_sd: tuple[float, float, float] = (1.25, 1.00, 0.00)
    competing_path_effects_sd: tuple[float, float, float] = (0.00, 0.00, 1.00)
    biomarker_noise_sd: tuple[float, float, float] = (1.00, 1.00, 1.00)
    renal_effect_levels_sd: tuple[float, ...] = (0.00, 0.50, 1.00, 1.50)
    renal_effect_labels: tuple[str, ...] = (
        "None",
        "Weak",
        "Moderate",
        "Strong",
    )
    heart_failure_prevalence: float = 0.07
    heart_failure_effect_levels_sd: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)

    def __post_init__(self) -> None:
        """Validate before any simulator can consume an RNG stream."""

        self.validate()

    def validate(self) -> None:
        """Reject invalid DGP controls before a cohort consumes any random draw."""

        if self.training_patients < 50 or self.test_patients < 50:
            raise ValueError("Training and test cohorts must each contain >= 50 patients.")
        probabilities = (
            self.renal_dysfunction_prevalence,
            self.atrial_probability_if_renal_normal,
            self.atrial_probability_if_renal_impaired,
        )
        if any(not 0.0 < value < 1.0 for value in probabilities):
            raise ValueError("All simulation probabilities must lie strictly between 0 and 1.")
        if len(self.renal_effect_levels_sd) != len(self.renal_effect_labels):
            raise ValueError("Each renal-effect level requires one display label.")
        for name, vector in (
            ("atrial_path_effects_sd", self.atrial_path_effects_sd),
            ("competing_path_effects_sd", self.competing_path_effects_sd),
            ("biomarker_noise_sd", self.biomarker_noise_sd),
        ):
            if len(vector) != _BIOMARKER_COUNT:
                raise ValueError(f"{name} must contain one value per biomarker.")
        if any(value <= 0.0 for value in self.biomarker_noise_sd):
            raise ValueError("Residual biomarker standard deviations must be positive.")
        if any(value < 0.0 for value in self.renal_effect_levels_sd):
            raise ValueError("Renal-effect strengths cannot be negative.")
        if not 0.0 < self.heart_failure_prevalence < 1.0:
            raise ValueError(
                "heart_failure_prevalence must lie strictly between 0 and 1."
            )
        if any(value < 0.0 for value in self.heart_failure_effect_levels_sd):
            raise ValueError("Heart-failure effect strengths cannot be negative.")


@dataclass(frozen=True)
class FittingConfig(ValidatedConfig):
    """Numerical controls shared by the latent-model fits."""

    random_starts: int = 4
    maximum_em_iterations: int = 300
    relative_log_likelihood_tolerance: float = 1e-6
    variance_floor: float = 0.05**2
    probability_floor: float = 1e-5
    beta_prior_pseudocount: float = 0.5
    minimum_effective_class_fraction: float = 0.02
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        """Reject invalid numerical controls at immutable construction time."""

        self.validate()

    def validate(self) -> None:
        """Guard the exact EM floors, starts, and stopping rule used for cited fits."""

        if self.random_starts < 2:
            raise ValueError("At least two EM starts are required for a latent model.")
        if self.maximum_em_iterations < 10:
            raise ValueError("maximum_em_iterations must be at least 10.")
        if not 0.0 < self.relative_log_likelihood_tolerance < 1.0:
            raise ValueError("The EM tolerance must lie between 0 and 1.")
        if self.variance_floor <= 0.0:
            raise ValueError("The variance floor must be positive.")
        if not 0.0 < self.probability_floor < 0.5:
            raise ValueError("The probability floor must lie between 0 and 0.5.")
        if self.beta_prior_pseudocount <= 0.0:
            raise ValueError("The beta-prior pseudocount must be positive.")
        if not 0.0 < self.minimum_effective_class_fraction < 0.5:
            raise ValueError("The minimum class fraction must lie between 0 and 0.5.")
        if self.calibration_bins < 2:
            raise ValueError("At least two calibration bins are required.")


@dataclass(frozen=True)
class ExperimentConfig(ValidatedConfig):
    """Complete reproducible experiment specification."""

    master_seed: int = 20_260_728
    repeats_per_level: int = 500
    null_repeats: int = 500
    null_renal_effect_sd: float = 1.50
    workers: int = 1
    simulation: SimulationConfig = SimulationConfig()
    fitting: FittingConfig = FittingConfig()

    def __post_init__(self) -> None:
        """Validate the complete nested design before repeat seeds are derived."""

        self.validate()

    def validate(self) -> None:
        """Validate nested controls before deriving the paired repeat seed ledger."""

        if self.repeats_per_level < 2 or self.null_repeats < 2:
            raise ValueError("At least two repeats are required for interval estimation.")
        if self.null_renal_effect_sd < 0.0:
            raise ValueError("The null renal-effect strength cannot be negative.")
        if self.workers < 1:
            raise ValueError("workers must be at least 1.")
        self.simulation.validate()
        self.fitting.validate()


# Four workers reproduce the configuration recorded in the proposal-cited output.
CONFIG = ExperimentConfig(workers=4)

__all__ = ["CONFIG", "ExperimentConfig", "FittingConfig", "SimulationConfig"]
