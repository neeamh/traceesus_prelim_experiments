"""Generate every retained two-mechanism cohort without changing RNG order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.random import Generator
from scipy.special import expit, logit

from configs.counterfactual import ExperimentConfig as CounterfactualConfig
from configs.endotype_discovery import SimulationConfig as DiscoveryConfig
from configs.transportability import (
    HospitalSpec,
    TransportSimulationConfig,
)
from traceesus.core.simulator import Cohort, SimulatedData, SimulationTruth, Simulator
from traceesus.simulators.multi_hospital import (
    MultiHospitalCohort,
    TransportSimulationTruth,
)


SimulationConfig: TypeAlias = (
    DiscoveryConfig | CounterfactualConfig | TransportSimulationConfig
)
_NT_PROBNP = 0
_PTFV1 = 1
_COMPETING_MARKER = 2


def _class_effects(
    atrial: tuple[float, ...],
    competing: tuple[float, ...],
) -> np.ndarray:
    """Return the two mechanism-effect rows in their historical order."""

    return np.asarray((atrial, competing), dtype=float)


def _standard_result(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    heart_failure: np.ndarray,
    mechanism: np.ndarray,
) -> SimulatedData:
    """Separate observed arrays from mechanism truth for non-hospital cohorts."""

    observed = Cohort(
        biomarkers=biomarkers,
        covariates={
            "renal_dysfunction": renal,
            "heart_failure": heart_failure,
        },
    )
    return SimulatedData(observed, SimulationTruth(mechanism))


def _transport_calibrated(
    config: TransportSimulationConfig,
    site: HospitalSpec,
    mechanism: np.ndarray,
    renal: np.ndarray,
    inflammation: np.ndarray,
    renal_effect_sd: float,
) -> np.ndarray:
    """Apply mechanism, renal, and inflammation paths before random noise."""

    calibrated = _class_effects(
        config.atrial_path_effects_sd,
        config.competing_path_effects_sd,
    )[mechanism].copy()
    calibrated[:, _NT_PROBNP] += renal_effect_sd * renal
    calibrated[:, _COMPETING_MARKER] += (
        site.inflammation_effect_competing_sd * inflammation
    )
    return calibrated


def _missing_log_odds(
    config: TransportSimulationConfig,
    site: HospitalSpec,
    shape: tuple[int, int],
    renal: np.ndarray,
    inflammation: np.ndarray,
) -> np.ndarray:
    """Build the unchanged site-specific missingness probability surface."""

    values = np.broadcast_to(logit(np.asarray(site.missingness_base)), shape).copy()
    values[:, _NT_PROBNP] += config.renal_missingness_log_odds_nt * renal
    values[:, _COMPETING_MARKER] += (
        config.inflammation_missingness_log_odds_competing * inflammation
    )
    return values


def _counterfactual_mean(
    config: CounterfactualConfig,
    mechanism: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
) -> np.ndarray:
    """Build the known-SCM biomarker mean before its normal draw."""

    renal_contribution = np.zeros((renal.size, 3), dtype=float)
    renal_contribution[:, _NT_PROBNP] = renal_effect_sd * renal
    return np.asarray(config.mechanism_effects, dtype=float)[mechanism] + renal_contribution


def _discovery_mean(
    config: DiscoveryConfig,
    mechanism: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
) -> np.ndarray:
    """Build the discovery biomarker mean before its normal draw."""

    renal_effect = renal_effect_sd * np.asarray(
        getattr(config, "renal_path_effects_sd", (1.0, 0.0, 0.0)), dtype=float
    )
    effects = _class_effects(
        config.atrial_path_effects_sd,
        config.competing_path_effects_sd,
    )
    return effects[mechanism] + renal[:, None] * renal_effect


@dataclass(frozen=True)
class TwoMechanismSimulator(Simulator):
    """Generate discovery, transport, or known-SCM cohorts from explicit config."""

    config: SimulationConfig
    renal_effect_sd: float
    heart_failure_effect_sd: float = 0.0
    site: HospitalSpec | None = None

    def simulate(self, rng: Generator, patient_count: int) -> SimulatedData:
        """Draw in each design's locked order.

        Discovery: renal, mechanism, biomarker, heart failure. Counterfactual:
        renal, mechanism, biomarker. Transport: renal, inflammation, mechanism,
        biomarker, missingness. Disabled paths consume no random draw.
        """

        if self.site is not None:
            if not isinstance(self.config, TransportSimulationConfig):
                raise TypeError("A hospital site requires TransportSimulationConfig.")
            site = self.site
            renal = rng.binomial(1, site.renal_prevalence, patient_count).astype(np.int8)
            inflammation = rng.binomial(1, site.inflammation_prevalence, patient_count).astype(np.int8)
            mechanism = np.where(rng.random(patient_count) < self.config.atrial_probability, 0, 1).astype(np.int8)
            calibrated = _transport_calibrated(
                self.config, site, mechanism, renal, inflammation, self.renal_effect_sd
            )
            calibrated += rng.normal(0.0, np.asarray(self.config.biomarker_noise_sd), size=calibrated.shape)
            raw = np.asarray(site.assay_offset) + np.asarray(site.assay_scale) * calibrated
            missing_log_odds = _missing_log_odds(
                self.config, site, raw.shape, renal, inflammation
            )
            missing = rng.random(raw.shape) < expit(missing_log_odds)
            return self._transport_result(raw, ~missing, renal, inflammation, mechanism, calibrated)

        if isinstance(self.config, CounterfactualConfig):
            renal = rng.binomial(1, self.config.renal_prevalence, size=patient_count).astype(int)
            log_odds = self.config.atrial_log_odds_when_renal_normal + self.config.renal_to_atrial_log_odds * renal
            prior = expit(log_odds)
            mechanism = np.where(rng.random(patient_count) < prior, 0, 1)
            mean = _counterfactual_mean(self.config, mechanism, renal, self.renal_effect_sd)
            biomarkers = mean + rng.normal(size=(patient_count, 3)) * np.asarray(
                self.config.biomarker_noise_sd, dtype=float
            )
            zeros = np.zeros(patient_count, dtype=np.int8)
            return _standard_result(biomarkers, renal, zeros, mechanism)

        if not isinstance(self.config, DiscoveryConfig):
            raise TypeError("Unsupported two-mechanism simulation config.")
        renal = rng.binomial(
            1, self.config.renal_dysfunction_prevalence, size=patient_count
        ).astype(np.int8)
        prior = np.where(
            renal == 1,
            self.config.atrial_probability_if_renal_impaired,
            self.config.atrial_probability_if_renal_normal,
        )
        mechanism = np.where(rng.random(patient_count) < prior, 0, 1).astype(np.int8)
        mean = _discovery_mean(self.config, mechanism, renal, self.renal_effect_sd)
        biomarkers = mean + rng.normal(
            0.0, np.asarray(self.config.biomarker_noise_sd), size=(patient_count, 3)
        )
        heart_failure = rng.binomial(
            1, self.config.heart_failure_prevalence, size=patient_count
        ).astype(np.int8)
        biomarkers = biomarkers + heart_failure[:, None] * self._heart_failure_effect()
        return _standard_result(biomarkers, renal, heart_failure, mechanism)

    def simulate_null(self, rng: Generator, patient_count: int) -> SimulatedData:
        """Draw renal then biomarker normal, while filling all labels with zeros.

        Heart failure, inflammation, and mechanism are never drawn in the null
        cohort; their arrays are filled with zeros after the two historical
        draws. This preserves the complete K=1 RNG sequence exactly.
        """

        if isinstance(self.config, CounterfactualConfig):
            renal = rng.binomial(1, self.config.renal_prevalence, size=patient_count)
            renal_contribution = np.zeros((patient_count, 3), dtype=float)
            renal_contribution[:, _NT_PROBNP] = self.renal_effect_sd * renal
            biomarkers = renal_contribution + rng.normal(size=(patient_count, 3))
            zeros = np.zeros(patient_count, dtype=np.int8)
            return _standard_result(biomarkers, renal, zeros, zeros)

        if not isinstance(self.config, DiscoveryConfig):
            raise TypeError("Null cohorts require discovery or counterfactual config.")
        renal = rng.binomial(
            1, self.config.renal_dysfunction_prevalence, size=patient_count
        ).astype(np.int8)
        renal_effect = np.zeros(3, dtype=float)
        renal_effect[_NT_PROBNP] = self.renal_effect_sd
        biomarkers = renal[:, None] * renal_effect + rng.normal(
            0.0, np.asarray(self.config.biomarker_noise_sd), size=(patient_count, 3)
        )
        zeros = np.zeros(patient_count, dtype=np.int8)
        return _standard_result(biomarkers, renal, zeros, zeros)

    def _heart_failure_effect(self) -> np.ndarray:
        """Return the configured heart-failure loading, scaled by its strength.

        The default loading is PTFV1-only, which reproduces the locked cohorts
        exactly.  An ARCADIA-calibrated configuration supplies a multi-marker
        vector so that renal and heart failure contaminate overlapping markers.
        """

        loading = np.asarray(
            getattr(self.config, "heart_failure_path_effects_sd", (0.0, 1.0, 0.0)),
            dtype=float,
        )
        return self.heart_failure_effect_sd * loading

    def _transport_result(
        self,
        raw: np.ndarray,
        observed: np.ndarray,
        renal: np.ndarray,
        inflammation: np.ndarray,
        mechanism: np.ndarray,
        calibrated: np.ndarray,
    ) -> SimulatedData:
        """Mask missing values and keep pre-missingness values evaluation-only."""

        if self.site is None:
            raise RuntimeError("Transport output requires hospital metadata.")
        raw_with_missing = raw.copy()
        raw_with_missing[~observed] = np.nan
        cohort = MultiHospitalCohort(
            biomarkers=raw_with_missing,
            covariates={
                "renal_dysfunction": renal,
                "background_inflammation": inflammation,
            },
            measurement_indicators=observed,
            hospital=self.site,
        )
        truth = TransportSimulationTruth(
            mechanism=mechanism,
            complete_calibrated_biomarkers=calibrated,
        )
        return SimulatedData(cohort, truth)


__all__ = ("TwoMechanismSimulator",)
