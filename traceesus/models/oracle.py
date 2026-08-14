"""Evaluate the known two-mechanism data-generating posterior as a ceiling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from configs.endotype_discovery import FittingConfig, SimulationConfig
from traceesus.core.em import e_step
from traceesus.core.markers import Biomarker
from traceesus.core.model import FittedModel, Model
from traceesus.core.simulator import Cohort


ORACLE = "Data-generating oracle (reference)"


def oracle_posterior(
    biomarkers: np.ndarray,
    renal: np.ndarray,
    renal_effect_sd: float,
    config: SimulationConfig,
) -> np.ndarray:
    """Evaluate Bayes' rule under the known simulator without consuming RNG."""

    class_probability = np.asarray(
        (
            (config.atrial_probability_if_renal_normal, 1.0 - config.atrial_probability_if_renal_normal),
            (config.atrial_probability_if_renal_impaired, 1.0 - config.atrial_probability_if_renal_impaired),
        ),
        dtype=float,
    )
    class_means = np.asarray(
        (config.atrial_path_effects_sd, config.competing_path_effects_sd),
        dtype=float,
    )
    renal_effect = np.zeros(biomarkers.shape[1], dtype=float)
    renal_effect[Biomarker.NT_PROBNP] = renal_effect_sd
    variance = np.asarray(config.biomarker_noise_sd, dtype=float) ** 2
    responsibility, _ = e_step(
        biomarkers,
        np.log(class_probability[renal]),
        class_means,
        variance,
        nuisance_contribution=renal[:, None] * renal_effect[None, :],
    )
    return responsibility


@dataclass(frozen=True)
class FittedDataGeneratingOracle(FittedModel):
    """Known DGP posterior; never a fitted or clinically available method."""

    renal_effect_sd: float
    simulation_config: SimulationConfig

    def posterior(self, data: Cohort) -> np.ndarray:
        return oracle_posterior(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            self.renal_effect_sd,
            self.simulation_config,
        )

    @property
    def n_parameters(self) -> int:
        return 0


@dataclass(frozen=True)
class DataGeneratingOracle(Model):
    """Construct the known-DGP ceiling without consuming a fitting RNG stream."""

    renal_effect_sd: float
    simulation_config: SimulationConfig
    name: str = ORACLE

    def fit(
        self, data: Cohort, rng: Generator, config: FittingConfig
    ) -> FittedDataGeneratingOracle:
        """Return immutable known parameters; data and RNG are intentionally unused."""

        return FittedDataGeneratingOracle(self.renal_effect_sd, self.simulation_config)


__all__ = ["DataGeneratingOracle", "FittedDataGeneratingOracle", "ORACLE", "oracle_posterior"]
