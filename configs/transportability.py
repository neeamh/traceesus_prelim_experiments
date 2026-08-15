"""Scientific defaults for unlabeled cross-hospital transportability."""

from __future__ import annotations

from dataclasses import dataclass, replace

from configs.endotype_discovery import FittingConfig
from traceesus.core.config import ValidatedConfig
from traceesus.core.seeds import ENDOTYPE_RECOVERY_SEED_ROOT


_BIOMARKER_COUNT = 3


@dataclass(frozen=True)
class HospitalSpec(ValidatedConfig):
    """Observed-data nuisance parameters for one hospital."""

    name: str
    renal_prevalence: float
    renal_effect_nt_sd: float
    inflammation_prevalence: float
    inflammation_effect_competing_sd: float
    assay_offset: tuple[float, float, float]
    assay_scale: tuple[float, float, float]
    missingness_base: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Reject invalid hospital shifts before they enter any RNG stream."""

        self.validate()

    def validate(self) -> None:
        """Reject impossible nuisance, assay, or missingness specifications."""

        if not 0.0 < self.renal_prevalence < 1.0:
            raise ValueError(f"{self.name}: renal_prevalence must lie in (0, 1).")
        if not 0.0 < self.inflammation_prevalence < 1.0:
            raise ValueError(
                f"{self.name}: inflammation_prevalence must lie in (0, 1)."
            )
        if self.renal_effect_nt_sd < 0.0:
            raise ValueError(f"{self.name}: renal effect cannot be negative.")
        if self.inflammation_effect_competing_sd < 0.0:
            raise ValueError(f"{self.name}: inflammation effect cannot be negative.")
        if any(value <= 0.0 for value in self.assay_scale):
            raise ValueError(f"{self.name}: assay scales must be positive.")
        if any(not 0.0 <= value < 1.0 for value in self.missingness_base):
            raise ValueError(f"{self.name}: missingness probabilities are invalid.")
        for field_name, values in (
            ("assay_offset", self.assay_offset),
            ("assay_scale", self.assay_scale),
            ("missingness_base", self.missingness_base),
        ):
            if len(values) != _BIOMARKER_COUNT:
                raise ValueError(
                    f"{self.name}: {field_name} must have {_BIOMARKER_COUNT} values."
                )


SOURCE_HOSPITALS = (
    HospitalSpec(
        name="Source A",
        renal_prevalence=0.25,
        renal_effect_nt_sd=0.80,
        inflammation_prevalence=0.20,
        inflammation_effect_competing_sd=0.70,
        assay_offset=(-0.10, 0.05, 0.00),
        assay_scale=(0.98, 1.02, 1.00),
        missingness_base=(0.05, 0.04, 0.05),
    ),
    HospitalSpec(
        name="Source B",
        renal_prevalence=0.30,
        renal_effect_nt_sd=1.00,
        inflammation_prevalence=0.25,
        inflammation_effect_competing_sd=0.80,
        assay_offset=(0.00, 0.00, 0.00),
        assay_scale=(1.00, 1.00, 1.00),
        missingness_base=(0.07, 0.05, 0.06),
    ),
    HospitalSpec(
        name="Source C",
        renal_prevalence=0.35,
        renal_effect_nt_sd=1.20,
        inflammation_prevalence=0.30,
        inflammation_effect_competing_sd=0.90,
        assay_offset=(0.10, -0.05, 0.05),
        assay_scale=(1.02, 0.98, 1.03),
        missingness_base=(0.09, 0.06, 0.08),
    ),
)

TARGET_HOSPITALS = (
    HospitalSpec(
        name="No shift",
        renal_prevalence=0.30,
        renal_effect_nt_sd=1.00,
        inflammation_prevalence=0.25,
        inflammation_effect_competing_sd=0.80,
        assay_offset=(0.00, 0.00, 0.00),
        assay_scale=(1.00, 1.00, 1.00),
        missingness_base=(0.07, 0.05, 0.06),
    ),
    HospitalSpec(
        name="Mild shift",
        renal_prevalence=0.40,
        renal_effect_nt_sd=1.25,
        inflammation_prevalence=0.35,
        inflammation_effect_competing_sd=0.80,
        assay_offset=(0.15, -0.10, 0.08),
        assay_scale=(1.05, 0.98, 1.03),
        missingness_base=(0.12, 0.10, 0.10),
    ),
    HospitalSpec(
        name="Moderate shift",
        renal_prevalence=0.50,
        renal_effect_nt_sd=1.50,
        inflammation_prevalence=0.45,
        inflammation_effect_competing_sd=0.80,
        assay_offset=(0.30, -0.20, 0.16),
        assay_scale=(1.10, 0.96, 1.06),
        missingness_base=(0.22, 0.18, 0.17),
    ),
    HospitalSpec(
        name="Strong shift",
        renal_prevalence=0.60,
        renal_effect_nt_sd=1.80,
        inflammation_prevalence=0.55,
        inflammation_effect_competing_sd=0.80,
        assay_offset=(0.45, -0.30, 0.24),
        assay_scale=(1.15, 0.94, 1.09),
        missingness_base=(0.35, 0.30, 0.28),
    ),
)

ABLATION_TARGETS = (
    TARGET_HOSPITALS[0],
    replace(
        TARGET_HOSPITALS[0],
        name="Kidney only",
        renal_prevalence=TARGET_HOSPITALS[-1].renal_prevalence,
        renal_effect_nt_sd=TARGET_HOSPITALS[-1].renal_effect_nt_sd,
    ),
    replace(
        TARGET_HOSPITALS[0],
        name="Inflammation only",
        inflammation_prevalence=TARGET_HOSPITALS[-1].inflammation_prevalence,
    ),
    replace(
        TARGET_HOSPITALS[0],
        name="Assay only",
        assay_offset=TARGET_HOSPITALS[-1].assay_offset,
        assay_scale=TARGET_HOSPITALS[-1].assay_scale,
    ),
    replace(
        TARGET_HOSPITALS[0],
        name="Missingness only",
        missingness_base=TARGET_HOSPITALS[-1].missingness_base,
    ),
    replace(TARGET_HOSPITALS[-1], name="Combined strong"),
)


@dataclass(frozen=True)
class TransportSimulationConfig(ValidatedConfig):
    """Stable biology, sample sizes, and missingness mechanisms."""

    source_patients_per_hospital: int = 600
    target_calibration_patients: int = 150
    target_test_patients: int = 1_000
    atrial_probability: float = 0.50
    atrial_path_effects_sd: tuple[float, float, float] = (1.25, 1.00, 0.00)
    competing_path_effects_sd: tuple[float, float, float] = (0.00, 0.00, 1.00)
    biomarker_noise_sd: tuple[float, float, float] = (1.00, 1.00, 1.00)
    renal_missingness_log_odds_nt: float = -0.60
    inflammation_missingness_log_odds_competing: float = -0.60
    assay_metadata_known: bool = True

    def __post_init__(self) -> None:
        """Validate immutable simulation controls at construction time."""

        self.validate()

    def validate(self) -> None:
        """Reject sample sizes and biological vectors outside the fixed design."""

        if self.source_patients_per_hospital < 100:
            raise ValueError("Each source hospital requires at least 100 patients.")
        if self.target_calibration_patients < 50:
            raise ValueError("The target calibration cohort requires at least 50 patients.")
        if self.target_test_patients < 100:
            raise ValueError("The target test cohort requires at least 100 patients.")
        if not 0.0 < self.atrial_probability < 1.0:
            raise ValueError("atrial_probability must lie in (0, 1).")
        for vector in (
            self.atrial_path_effects_sd,
            self.competing_path_effects_sd,
            self.biomarker_noise_sd,
        ):
            if len(vector) != _BIOMARKER_COUNT:
                raise ValueError("Biological vectors require one value per biomarker.")
        if any(value <= 0.0 for value in self.biomarker_noise_sd):
            raise ValueError("Biomarker noise SDs must be positive.")


@dataclass(frozen=True)
class TransportExperimentConfig(ValidatedConfig):
    """Exact repeat, seed, fitting, and hospital controls for transportability."""

    master_seed: int = ENDOTYPE_RECOVERY_SEED_ROOT
    repeats: int = 500
    workers: int = 1
    equivalence_margin_accuracy: float = 0.01
    simulation: TransportSimulationConfig = TransportSimulationConfig()
    fitting: FittingConfig = FittingConfig()
    source_hospitals: tuple[HospitalSpec, ...] = SOURCE_HOSPITALS
    target_hospitals: tuple[HospitalSpec, ...] = TARGET_HOSPITALS

    def __post_init__(self) -> None:
        """Validate every nested control before per-repeat seeds are derived."""

        self.validate()

    def validate(self) -> None:
        """Guard paired-repeat, hospital, and nested-model configuration invariants."""

        if self.repeats < 2:
            raise ValueError("At least two repeats are required.")
        if self.workers < 1:
            raise ValueError("workers must be at least one.")
        if not 0.0 < self.equivalence_margin_accuracy < 0.10:
            raise ValueError("The equivalence margin must be in (0, 0.10).")
        if len(self.source_hospitals) < 2:
            raise ValueError("At least two source hospitals are required.")
        if len(self.target_hospitals) < 1:
            raise ValueError("At least one target hospital is required.")
        self.simulation.validate()
        self.fitting.validate()
        for hospital in (*self.source_hospitals, *self.target_hospitals):
            hospital.validate()


# Four ordered workers match the proposal-cited transportability run metadata.
CONFIG = TransportExperimentConfig(workers=4)

__all__ = [
    "ABLATION_TARGETS",
    "CONFIG",
    "HospitalSpec",
    "SOURCE_HOSPITALS",
    "TARGET_HOSPITALS",
    "TransportExperimentConfig",
    "TransportSimulationConfig",
]
