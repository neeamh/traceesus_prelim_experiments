"""Transport-model adapters for missing biomarkers and modular nuisance paths."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.random import Generator

from traceesus.core.model import FittedModel, Model
from traceesus.core.simulator import Cohort
from traceesus.experiments.transportability import kernel
from traceesus.simulators.multi_hospital import (
    MultiHospitalCohort,
    SourceHospitalPool,
    assay_calibrate_observed,
)


TransportModelConfig: TypeAlias = (
    kernel.FittingConfig | kernel.TransportExperimentConfig
)


def _fitting_config(config: TransportModelConfig) -> kernel.FittingConfig:
    """Extract unchanged EM controls from the two supported config contracts."""

    if isinstance(config, kernel.FittingConfig):
        return config
    if isinstance(config, kernel.TransportExperimentConfig):
        return config.fitting
    raise TypeError("Transport models require FittingConfig or .fitting.")


def _source_blocks(data: Cohort) -> tuple[Cohort, ...]:
    """Retain source-hospital order when per-site paths must be estimated."""

    if isinstance(data, SourceHospitalPool):
        return data.hospital_cohorts
    return (data,)


@dataclass(frozen=True)
class FittedPooledTransportMixture(FittedModel):
    """Source-pooled missing-data mixture with no nuisance adjustment."""

    fit_result: kernel.MissingGaussianMixtureFit

    def posterior(self, data: Cohort) -> np.ndarray:
        """Apply known assay calibration and the frozen missing-data likelihood."""

        return kernel.missing_gmm_posterior(
            self.fit_result,
            assay_calibrate_observed(data),
        )

    @property
    def n_parameters(self) -> int:
        """Count one free class probability, six means, and three variances."""

        return 10


class PooledAssociativeTransportModel(Model):
    """Fit the pooled associative mixture to calibrated source biomarkers."""

    name = kernel.POOLED_ASSOCIATIVE

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: TransportModelConfig,
    ) -> FittedPooledTransportMixture:
        """Run the exact missing-data EM arithmetic on the supplied source pool."""

        result = kernel.fit_missing_gaussian_mixture(
            assay_calibrate_observed(data),
            rng,
            _fitting_config(config),
        )
        return FittedPooledTransportMixture(result)


@dataclass(frozen=True)
class FittedResidualizedTransportMixture(FittedModel):
    """Biological mixture plus a replaceable observed-data nuisance module."""

    fit_result: kernel.MissingGaussianMixtureFit
    renal_path_mask: np.ndarray
    inflammation_path_mask: np.ndarray
    source_slopes: tuple[np.ndarray, ...]
    active_slopes: np.ndarray

    def posterior(self, data: Cohort) -> np.ndarray:
        """Evaluate after calibration and removal of the currently active paths."""

        biomarkers = assay_calibrate_observed(data)
        residualized = kernel.remove_nuisance_paths(
            biomarkers,
            data.covariate("renal_dysfunction"),
            data.covariate("background_inflammation"),
            self.active_slopes,
        )
        return kernel.missing_gmm_posterior(self.fit_result, residualized)

    def recalibrate(self, target_calibration: Cohort) -> "FittedResidualizedTransportMixture":
        """Replace only nuisance slopes using unlabeled target calibration data."""

        slopes = kernel.fit_nuisance_paths(
            assay_calibrate_observed(target_calibration),
            target_calibration.covariate("renal_dysfunction"),
            target_calibration.covariate("background_inflammation"),
            self.renal_path_mask,
            self.inflammation_path_mask,
        )
        return replace(self, active_slopes=slopes)

    @property
    def n_parameters(self) -> int:
        """Count mixture parameters and the permitted nuisance slopes."""

        return int(
            10
            + np.sum(self.renal_path_mask)
            + np.sum(self.inflammation_path_mask)
        )


class _ResidualizedTransportModel(Model):
    """Shared exact preprocessing for flexible and constrained path masks."""

    renal_path_mask: np.ndarray
    inflammation_path_mask: np.ndarray

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: TransportModelConfig,
    ) -> FittedResidualizedTransportMixture:
        """Fit per-source slopes in block order, then one pooled latent mixture."""

        residuals: list[np.ndarray] = []
        source_slopes: list[np.ndarray] = []
        for source in _source_blocks(data):
            biomarkers = assay_calibrate_observed(source)
            slopes = kernel.fit_nuisance_paths(
                biomarkers,
                source.covariate("renal_dysfunction"),
                source.covariate("background_inflammation"),
                self.renal_path_mask,
                self.inflammation_path_mask,
            )
            source_slopes.append(slopes)
            residuals.append(
                kernel.remove_nuisance_paths(
                    biomarkers,
                    source.covariate("renal_dysfunction"),
                    source.covariate("background_inflammation"),
                    slopes,
                )
            )
        result = kernel.fit_missing_gaussian_mixture(
            np.vstack(residuals),
            rng,
            _fitting_config(config),
        )
        frozen_slopes = np.mean(np.stack(source_slopes), axis=0)
        return FittedResidualizedTransportMixture(
            fit_result=result,
            renal_path_mask=self.renal_path_mask.copy(),
            inflammation_path_mask=self.inflammation_path_mask.copy(),
            source_slopes=tuple(source_slopes),
            active_slopes=frozen_slopes,
        )


class TargetAdjustedAssociativeModel(_ResidualizedTransportModel):
    """Associative mixture with freely estimated target nuisance regressions."""

    name = kernel.TARGET_ADJUSTED_ASSOCIATIVE
    renal_path_mask = np.ones(len(kernel.BIOMARKER_NAMES), dtype=bool)
    inflammation_path_mask = np.ones(len(kernel.BIOMARKER_NAMES), dtype=bool)


class FrozenCausalSCM(_ResidualizedTransportModel):
    """Causal-path mixture transported with the mean source nuisance slopes."""

    name = kernel.FROZEN_CAUSAL
    renal_path_mask = np.asarray((True, False, False), dtype=bool)
    inflammation_path_mask = np.asarray((False, False, True), dtype=bool)


class ModularCausalSCM(_ResidualizedTransportModel):
    """Transport invariant biology while refitting only target nuisance paths.

    Calling :meth:`FittedResidualizedTransportMixture.recalibrate` on an
    unlabeled target calibration cohort yields the modular target model. The
    biological mixture is not refit, and no mechanism label is accepted.
    """

    name = kernel.MODULAR_CAUSAL
    renal_path_mask = np.asarray((True, False, False), dtype=bool)
    inflammation_path_mask = np.asarray((False, False, True), dtype=bool)


@dataclass(frozen=True)
class FittedTargetTransportOracle(FittedModel):
    """Known-DGP target posterior retained solely as a simulation ceiling."""

    simulation: kernel.TransportSimulationConfig

    def posterior(self, data: Cohort) -> np.ndarray:
        """Evaluate the target DGP posterior without reading mechanism labels."""

        if not isinstance(data, MultiHospitalCohort) or data.hospital is None:
            raise TypeError("The transport oracle requires hospital metadata.")
        biomarkers = assay_calibrate_observed(data)
        true_slopes = np.zeros((2, len(kernel.BIOMARKER_NAMES)), dtype=float)
        true_slopes[0, kernel.Biomarker.NT_PROBNP_LIKE] = (
            data.hospital.renal_effect_nt_sd
        )
        true_slopes[1, kernel.Biomarker.COMPETING_SPECIFIC] = (
            data.hospital.inflammation_effect_competing_sd
        )
        residualized = kernel.remove_nuisance_paths(
            biomarkers,
            data.covariate("renal_dysfunction"),
            data.covariate("background_inflammation"),
            true_slopes,
        )
        class_means = np.asarray(
            (
                self.simulation.atrial_path_effects_sd,
                self.simulation.competing_path_effects_sd,
            ),
            dtype=float,
        )
        variance = np.asarray(self.simulation.biomarker_noise_sd, dtype=float) ** 2
        class_probability = np.asarray(
            (
                self.simulation.atrial_probability,
                1.0 - self.simulation.atrial_probability,
            )
        )
        posterior, _ = kernel._missing_gmm_e_step(
            residualized,
            class_probability,
            class_means,
            variance,
        )
        return posterior

    @property
    def n_parameters(self) -> int:
        """Return zero because no target-oracle parameter is estimated."""

        return 0


@dataclass(frozen=True)
class TargetTransportOracle(Model):
    """Construct the target DGP ceiling without fitting or RNG consumption."""

    simulation: kernel.TransportSimulationConfig
    name: str = kernel.ORACLE

    def fit(
        self,
        data: Cohort,
        rng: Generator,
        config: TransportModelConfig,
    ) -> FittedTargetTransportOracle:
        """Return the prespecified ceiling; observed data and RNG remain unused."""

        return FittedTargetTransportOracle(self.simulation)


__all__ = [
    "FrozenCausalSCM",
    "ModularCausalSCM",
    "PooledAssociativeTransportModel",
    "TargetAdjustedAssociativeModel",
    "TargetTransportOracle",
    "TransportModelConfig",
]
