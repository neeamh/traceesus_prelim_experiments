"""Two-mechanism simulator using the proposal-locked draw order."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from traceesus.core.simulator import Cohort, SimulatedData, SimulationTruth, Simulator
from traceesus.experiments.endotype_discovery import kernel


@dataclass(frozen=True)
class TwoMechanismSimulator(Simulator):
    """Generate latent-discovery cohorts without altering a single RNG draw.

    ``heart_failure_effect_sd`` defaults to zero, which reproduces the locked
    outputs bit-for-bit: the HF draw happens after every historical draw, and a
    zero effect leaves every biomarker untouched while the covariate is still
    emitted for nuisance-profile subgrouping.
    """

    config: kernel.SimulationConfig
    renal_effect_sd: float
    heart_failure_effect_sd: float = 0.0

    def simulate(self, rng: Generator, patient_count: int) -> SimulatedData:
        """Wrap the frozen simulator and immediately separate truth from fit data."""

        generated = kernel.simulate_two_mechanism_cohort(
            rng,
            patient_count,
            self.renal_effect_sd,
            self.config,
            heart_failure_effect_sd=self.heart_failure_effect_sd,
        )
        observed = Cohort(
            biomarkers=generated.biomarkers,
            covariates={
                "renal_dysfunction": generated.renal_dysfunction,
                "heart_failure": generated.heart_failure,
            },
        )
        return SimulatedData(
            observed=observed,
            truth=SimulationTruth(mechanism=generated.true_mechanism),
        )
