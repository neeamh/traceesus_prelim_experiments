"""Data-generating oracle used only as a simulation ceiling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from traceesus.core.model import FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.experiments.endotype_discovery import kernel


@dataclass(frozen=True)
class FittedDataGeneratingOracle(FittedModel):
    """Known DGP posterior; never a fitted or clinically available method."""

    renal_effect_sd: float
    simulation_config: kernel.SimulationConfig

    def posterior(self, data: Cohort) -> np.ndarray:
        """Evaluate the exact simulator posterior as a reference ceiling."""

        return kernel.oracle_posterior(
            data.biomarkers,
            data.covariate("renal_dysfunction"),
            self.renal_effect_sd,
            self.simulation_config,
        )

    @property
    def n_parameters(self) -> int:
        """Return zero because no parameters are estimated from the cohort."""

        return 0


@dataclass(frozen=True)
class DataGeneratingOracle(Model):
    """Construct the known-DGP ceiling without consuming a fitting RNG stream."""

    renal_effect_sd: float
    simulation_config: kernel.SimulationConfig
    name: str = kernel.ORACLE

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: kernel.FittingConfig,
    ) -> FittedDataGeneratingOracle:
        """Return the prespecified oracle; neither data nor RNG are used for fitting."""

        return FittedDataGeneratingOracle(self.renal_effect_sd, self.simulation_config)
