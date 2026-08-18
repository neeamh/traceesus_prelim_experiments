"""ARCADIA-calibrated defaults for semi-synthetic latent endotype discovery.

Purpose
-------
The proposal-locked discovery configuration uses a deliberately severe renal
distortion (up to 1.50 residual SD) and a heart-failure path that touches only
PTFV1.  Neither matches the trial the score will actually be derived in.  This
module supplies the same data-generating process with every nuisance parameter
set to a value measured in ARCADIA, so recovery experiments run at realistic
distortion rather than at a level chosen to make the contrast visible.

Nothing here modifies the locked configuration.  This is an additive,
exploratory design with its own seed root and output directory.

Provenance
----------
All values below derive from ``arcadia (1).dta`` restricted to the 1,015
randomized participants (``randomized == 1``; the file also contains 2,730
screened-but-not-randomized patients).  Marker values were standardized within
the randomized cohort before effects were estimated, so every effect is already
expressed in the residual-SD units the simulator consumes.

Renal dysfunction is defined as CKD-EPI 2021 (race-free) eGFR < 60, computed
from creatinine, age, and sex.  Heart failure uses the cleaned baseline
indicator; the raw column contains encoding corruption (values such as ``300``
and control characters) and was coerced to {0, 1, missing}.

Measured quantities
~~~~~~~~~~~~~~~~~~~
=========================== ========= =====================================
Quantity                    Value     Note
=========================== ========= =====================================
Renal prevalence (eGFR<60)  0.199     n = 993 with computable eGFR
Heart-failure prevalence    0.070     71 / 1010 after cleaning
Renal -> log NT-proBNP      +0.587    SD
Renal -> |PTFV1|            -0.056    SD, effectively null
Renal -> LA diameter        +0.233    SD
HF -> log NT-proBNP         +0.647    SD
HF -> |PTFV1|               -0.149    SD
HF -> LA diameter           +0.620    SD
Complete 3-marker coverage  90.5%     randomized; 0% have no marker
=========================== ========= =====================================

Scientific consequences
~~~~~~~~~~~~~~~~~~~~~~~
1.  The renal path is roughly one third of the locked "Strong" level.  Any gain
    demonstrated at 1.50 SD is being shown in a regime ARCADIA never reaches.
2.  Renal and heart failure load on the *same* two markers (NT-proBNP and LA
    size) while leaving PTFV1 comparatively clean.  This is genuine nuisance
    redundancy measured in real data, not a hypothetical, and it is the regime
    in which evidence ablation could plausibly diverge from the posterior.
3.  Missingness is mild (90.5% complete), not the 28-35% used in transport
    experiments.  Discovery cohorts are complete by construction, so this
    module records the fact without simulating it.

Known limitations - do not overstate what "calibrated" means
------------------------------------------------------------
*   The generator works in standardized SD units.  ARCADIA markers are on raw
    clinical scales with heavy skew (NT-proBNP skew 7.16).  Calibration matches
    standardized effect sizes, not literal ng/L values.
*   ARCADIA has no competing-vascular marker comparable to the simulator's
    third biomarker.  Only the NT-proBNP-like and PTFV1-like channels are
    ARCADIA-calibrated; the competing channel retains its synthetic
    specification.  This gap is itself relevant to the identifiability
    question: three atrial markers cannot separate four mechanism classes.
*   Left-atrial size is not a channel in this three-marker generator.  The
    clinical HF-to-LA association therefore remains outside this experiment.
*   Mechanism path effects CANNOT be calibrated from ARCADIA, because no
    mechanism labels exist.  They are inherited unchanged from the locked
    configuration and remain an explicit modelling assumption.
*   SIGN CONVENTION REQUIRES CLINICAL CONFIRMATION.  In the extract, raw PTFV1
    and log NT-proBNP correlate +0.435, i.e. less abnormal PTFV1 accompanies
    higher NT-proBNP, which is opposite to the expected direction.  Magnitudes
    below are taken from ARCADIA; the *sign* of the atrial loading is retained
    from the biological model.  Dr. Khan should confirm the PTFV1 sign
    convention in the source file before these values are cited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from configs.endotype_discovery import (
    ExperimentConfig as DiscoveryExperimentConfig,
    FittingConfig,
    SimulationConfig,
)
from traceesus.core.seeds import ARCADIA_CALIBRATED_SEED_ROOT


# Biomarker channel order is fixed by the simulator: 0 NT-proBNP-like,
# 1 PTFV1-like, 2 competing-vascular.
ARCADIA_RENAL_PREVALENCE = 0.199
ARCADIA_HEART_FAILURE_PREVALENCE = 0.070

# Measured renal loading, normalized so the swept strength equals the observed
# NT-proBNP effect: (0.587, -0.056, 0.0) / 0.587.  The PTFV1 term is retained
# at its measured near-null value rather than being forced to zero.
ARCADIA_RENAL_LOADING = (1.0, -0.095, 0.0)

# Measured heart-failure loading, normalized against the same NT-proBNP scale:
# (0.647, -0.149, --) / 0.587.  The third entry is the competing channel, which
# ARCADIA cannot inform and is therefore left at zero.
ARCADIA_HEART_FAILURE_LOADING = (1.102, -0.254, 0.0)

# Renal's -0.095 PTFV1 loading is treated as near-noise.  HF retains both
# non-negligible representable paths: NT-proBNP and PTFV1.  The clinical HF-to-LA
# path cannot be encoded because this generator has no LA-size channel.
ARCADIA_BIOLOGY_PATH_MASK = (
    (True, False, False),
    (True, True, False),
)

# Observed renal effect on log NT-proBNP, in residual SD.  The sweep brackets
# it so the operating curve either side of the real value is visible.
ARCADIA_RENAL_EFFECT_SD = 0.587
ARCADIA_RENAL_LEVELS_SD = (0.00, 0.30, 0.59, 0.90)
ARCADIA_RENAL_LABELS = (
    "None",
    "Half ARCADIA",
    "ARCADIA observed",
    "Above ARCADIA",
)

# Observed heart-failure effect on log NT-proBNP, in residual SD, plus a sweep.
ARCADIA_HEART_FAILURE_EFFECT_SD = 0.647
ARCADIA_HEART_FAILURE_LEVELS_SD = (0.00, 0.32, 0.65, 1.00)

# Recorded for provenance and for the semi-synthetic missingness work; the
# discovery generator produces complete cohorts, so this is not consumed here.
ARCADIA_COMPLETE_THREE_MARKER_FRACTION = 0.905


@dataclass(frozen=True)
class ArcadiaSimulationConfig(SimulationConfig):
    """Discovery DGP with every nuisance parameter measured in ARCADIA.

    Mechanism path effects and biomarker noise are inherited unchanged from the
    locked configuration because ARCADIA supplies no mechanism labels.
    """

    training_patients: int = 800
    test_patients: int = 1_015
    renal_dysfunction_prevalence: float = ARCADIA_RENAL_PREVALENCE
    renal_effect_levels_sd: tuple[float, ...] = ARCADIA_RENAL_LEVELS_SD
    renal_effect_labels: tuple[str, ...] = ARCADIA_RENAL_LABELS
    renal_path_effects_sd: tuple[float, float, float] = ARCADIA_RENAL_LOADING
    heart_failure_prevalence: float = ARCADIA_HEART_FAILURE_PREVALENCE
    heart_failure_effect_levels_sd: tuple[float, ...] = (
        ARCADIA_HEART_FAILURE_LEVELS_SD
    )
    heart_failure_path_effects_sd: tuple[float, float, float] = (
        ARCADIA_HEART_FAILURE_LOADING
    )
    biology_path_mask: tuple[
        tuple[bool, bool, bool], tuple[bool, bool, bool]
    ] = ARCADIA_BIOLOGY_PATH_MASK


@dataclass(frozen=True)
class ArcadiaExperimentConfig(DiscoveryExperimentConfig):
    """Reproducible ARCADIA-calibrated discovery specification."""

    master_seed: int = ARCADIA_CALIBRATED_SEED_ROOT
    null_renal_effect_sd: float = ARCADIA_RENAL_EFFECT_SD
    simulation: SimulationConfig = field(default_factory=ArcadiaSimulationConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)


CONFIG = ArcadiaExperimentConfig(workers=4)

__all__ = [
    "ARCADIA_BIOLOGY_PATH_MASK",
    "ARCADIA_COMPLETE_THREE_MARKER_FRACTION",
    "ARCADIA_HEART_FAILURE_EFFECT_SD",
    "ARCADIA_HEART_FAILURE_LOADING",
    "ARCADIA_HEART_FAILURE_PREVALENCE",
    "ARCADIA_RENAL_EFFECT_SD",
    "ARCADIA_RENAL_LOADING",
    "ARCADIA_RENAL_PREVALENCE",
    "ArcadiaExperimentConfig",
    "ArcadiaSimulationConfig",
    "CONFIG",
]
