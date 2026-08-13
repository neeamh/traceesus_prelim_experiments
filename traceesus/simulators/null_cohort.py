"""Homogeneous K=1 simulator with a real renal biomarker path."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from traceesus.core.simulator import Cohort, SimulatedData, SimulationTruth, Simulator
from traceesus.experiments.endotype_discovery import kernel


@dataclass(frozen=True)
class NullCohortSimulator(Simulator):
    """Generate the exact K=1 negative-control cohort used for BIC selection."""

    config: kernel.SimulationConfig
    renal_effect_sd: float

    def simulate(self, rng: Generator, patient_count: int) -> SimulatedData:
        """Preserve binomial-then-normal consumption and separate zero truth labels."""

        generated = kernel.simulate_one_mechanism_null_cohort(
            rng, patient_count, self.renal_effect_sd, self.config
        )
        return SimulatedData(
            observed=Cohort(
                biomarkers=generated.biomarkers,
                covariates={"renal_dysfunction": generated.renal_dysfunction},
            ),
            truth=SimulationTruth(mechanism=generated.true_mechanism),
        )
