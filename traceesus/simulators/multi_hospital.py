"""Store transport cohorts and deterministic assay-calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from traceesus.core.simulator import (
    Cohort,
    SimulationTruth,
)
from configs.transportability import HospitalSpec


@dataclass(frozen=True)
class MultiHospitalCohort(Cohort):
    """Observed hospital cohort carrying known assay calibration metadata.

    Hospital offset and scale are observed metadata in this experiment. The
    class deliberately has no mechanism-label or complete-biomarker field, so
    a model cannot acquire simulator truth through its standard fit input.
    """

    hospital: HospitalSpec | None = None


@dataclass(frozen=True)
class SourceHospitalPool(Cohort):
    """Ordered source-hospital blocks used by transport model adapters.

    Keeping the blocks preserves the legacy per-hospital nuisance regressions;
    pooling them prematurely would change both estimates and floating-point
    reduction order.
    """

    hospital_cohorts: tuple[MultiHospitalCohort, ...] = ()


@dataclass(frozen=True)
class TransportSimulationTruth(SimulationTruth):
    """Evaluation-only mechanism labels and pre-missingness calibrated values."""

    complete_calibrated_biomarkers: np.ndarray | None = None


def assay_calibrate_observed(data: Cohort) -> np.ndarray:
    """Invert known assay metadata without consulting simulation truth."""

    if not isinstance(data, MultiHospitalCohort) or data.hospital is None:
        return data.biomarkers
    offset = np.asarray(data.hospital.assay_offset, dtype=float)
    scale = np.asarray(data.hospital.assay_scale, dtype=float)
    return (data.biomarkers - offset) / scale


def pool_source_hospitals(
    cohorts: Sequence[MultiHospitalCohort],
) -> SourceHospitalPool:
    """Pool observed sources while retaining their exact ordered block boundaries."""

    ordered = tuple(cohorts)
    if not ordered:
        raise ValueError("At least one source hospital is required.")
    calibrated = [assay_calibrate_observed(cohort) for cohort in ordered]
    return SourceHospitalPool(
        biomarkers=np.vstack(calibrated),
        covariates={
            "renal_dysfunction": np.concatenate(
                [cohort.covariate("renal_dysfunction") for cohort in ordered]
            ),
            "background_inflammation": np.concatenate(
                [cohort.covariate("background_inflammation") for cohort in ordered]
            ),
        },
        measurement_indicators=np.vstack(
            [cohort.measurement_indicators for cohort in ordered]
        ),
        hospital_cohorts=ordered,
    )


__all__ = (
    "MultiHospitalCohort",
    "SourceHospitalPool",
    "TransportSimulationTruth",
    "assay_calibrate_observed",
    "pool_source_hospitals",
)
