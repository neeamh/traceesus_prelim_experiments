"""Non-numerical contracts that prevent the package facade from overclaiming."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, get_type_hints

import numpy as np
import pytest

from configs.counterfactual import CONFIG as COUNTERFACTUAL_CONFIG
from configs.counterfactual import ExperimentConfig as CounterfactualConfig
from configs.endotype_discovery import (
    CONFIG as ENDOTYPE_CONFIG,
    ExperimentConfig as EndotypeExperimentConfig,
    FittingConfig,
    SimulationConfig,
)
from configs.model_comparison import (
    CONFIG as MODEL_COMPARISON_CONFIG,
    ComparisonConfig,
)
from configs.transportability import (
    CONFIG as TRANSPORT_CONFIG,
    HospitalSpec,
    TransportExperimentConfig,
    TransportSimulationConfig,
)
from traceesus.experiments.counterfactual import kernel as counterfactual_kernel
from traceesus.experiments.endotype_discovery import kernel as endotype_kernel
from traceesus.experiments.model_comparison import kernel as comparison_kernel
from traceesus.experiments.transportability import kernel as transport_kernel
from traceesus.models.modular_causal_scm import (
    PooledAssociativeTransportModel,
    TransportModelConfig,
    _fitting_config,
)
from traceesus.plotting import panels
from traceesus.plotting.theme import DEFAULT_FONT, FIGURE_SIZES_INCHES, PALETTE
from traceesus.queries.posterior import anchor_order


@pytest.mark.parametrize(
    "config",
    (
        COUNTERFACTUAL_CONFIG,
        ENDOTYPE_CONFIG,
        MODEL_COMPARISON_CONFIG,
        TRANSPORT_CONFIG,
    ),
)
def test_top_level_configs_are_frozen_and_validate_at_construction(config: object) -> None:
    """Require immutable configs to reject mutation before a seed can be spawned."""

    config.validate()
    with pytest.raises(FrozenInstanceError):
        setattr(config, next(iter(config.__dataclass_fields__)), None)


def test_invalid_repeat_overrides_fail_during_dataclass_replacement() -> None:
    """Prove construction, rather than experiment execution, triggers validation."""

    with pytest.raises(ValueError):
        replace(ENDOTYPE_CONFIG, repeats_per_level=1)
    with pytest.raises(ValueError):
        replace(TRANSPORT_CONFIG, repeats=1)


def test_config_classes_are_owned_by_the_visible_config_modules() -> None:
    """Keep parameter definitions where ``run.py list`` tells users to edit them."""

    classes = (
        CounterfactualConfig,
        SimulationConfig,
        FittingConfig,
        EndotypeExperimentConfig,
        ComparisonConfig,
        HospitalSpec,
        TransportSimulationConfig,
        TransportExperimentConfig,
    )
    assert all(config_type.__module__.startswith("configs.") for config_type in classes)

    # Kernels re-export the same class objects for legacy scripts and notebooks.
    assert counterfactual_kernel.ExperimentConfig is CounterfactualConfig
    assert endotype_kernel.ExperimentConfig is EndotypeExperimentConfig
    assert comparison_kernel.ComparisonConfig is ComparisonConfig
    assert transport_kernel.TransportExperimentConfig is TransportExperimentConfig


def test_transport_adapter_exposes_a_precise_config_union() -> None:
    """Keep the transport Model API typed without weakening it to ``Any``."""

    hints = get_type_hints(PooledAssociativeTransportModel.fit)
    assert hints["config"] == TransportModelConfig
    assert hints["config"] is not Any
    assert _fitting_config(TRANSPORT_CONFIG) is TRANSPORT_CONFIG.fitting
    assert _fitting_config(TRANSPORT_CONFIG.fitting) is TRANSPORT_CONFIG.fitting
    with pytest.raises(TypeError):
        _fitting_config(object())  # type: ignore[arg-type]


def test_plotting_surface_names_every_legacy_figure_family() -> None:
    """Expose shared plotting discovery while retaining exact kernel functions."""

    assert panels.plot_counterfactual_primary is counterfactual_kernel.plot_primary_figure
    assert panels.plot_model_comparison is comparison_kernel.plot_comparison
    assert panels.plot_endotype_recovery is endotype_kernel.plot_primary_figure
    assert panels.plot_endotype_controls is endotype_kernel.plot_control_figure
    assert panels.plot_endotype_example_patient is endotype_kernel.plot_example_patient
    assert panels.plot_transportability_primary is transport_kernel.plot_transport_figure
    assert panels.plot_transportability_controls is transport_kernel.plot_transport_controls
    assert panels.plot_transportability_ablation is transport_kernel.plot_ablation_figure

    assert DEFAULT_FONT == "DejaVu Sans"
    assert PALETTE["associative"] == "#D97706"
    assert PALETTE["causal"] == "#1D4ED8"
    assert FIGURE_SIZES_INCHES["endotype_recovery"] == (12.2, 5.1)
    assert FIGURE_SIZES_INCHES["transportability_primary"] == (12.4, 5.2)


def test_two_mechanism_kernels_retain_distinct_dtype_contracts() -> None:
    """Guard an intentional non-unification that could change cited arithmetic."""

    latent_rng = np.random.default_rng(12345)
    latent = endotype_kernel.simulate_two_mechanism_cohort(
        latent_rng,
        12,
        0.75,
        ENDOTYPE_CONFIG.simulation,
    )

    preliminary_config = replace(COUNTERFACTUAL_CONFIG, patients_per_repeat=12)
    preliminary_rng = np.random.default_rng(12345)
    preliminary = counterfactual_kernel.simulate_two_mechanism_study(
        preliminary_config,
        0.75,
        preliminary_rng,
    )

    assert latent.renal_dysfunction.dtype == np.dtype(np.int8)
    assert latent.true_mechanism.dtype == np.dtype(np.int8)
    assert preliminary["renal"].dtype == np.dtype(int)
    assert preliminary["mechanism"].dtype == np.dtype(int)


def _legacy_anchor_reference(
    class_means: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Reproduce the pre-extraction arithmetic as a frozen test oracle."""

    noise_sd = np.sqrt(variance)
    score = (
        class_means[:, endotype_kernel.Biomarker.ATRIAL_ELECTRICAL]
        / noise_sd[endotype_kernel.Biomarker.ATRIAL_ELECTRICAL]
        - class_means[:, endotype_kernel.Biomarker.COMPETING_SPECIFIC]
        / noise_sd[endotype_kernel.Biomarker.COMPETING_SPECIFIC]
    )
    atrial_component = int(np.argmax(score))
    competing_component = 1 - atrial_component
    order = np.asarray((atrial_component, competing_component), dtype=int)
    return order, float(score[atrial_component] - score[competing_component])


def test_shared_anchor_is_bit_identical_to_both_legacy_wrappers() -> None:
    """Lock tie behavior, label order, and the exact floating-point margin bits."""

    rng = np.random.default_rng(89_331)
    cases = [(np.zeros((2, 3)), np.ones(3))]
    cases.extend(
        (rng.normal(size=(2, 3)), np.exp(rng.normal(size=3)))
        for _ in range(128)
    )
    class_probability = np.asarray((0.37, 0.63))
    responsibility = rng.random((17, 2))

    for class_means, variance in cases:
        expected_order, expected_margin = _legacy_anchor_reference(
            class_means,
            variance,
        )
        actual_order, actual_margin = anchor_order(
            class_means,
            variance,
            atrial_electrical_index=endotype_kernel.Biomarker.ATRIAL_ELECTRICAL,
            competing_specific_index=endotype_kernel.Biomarker.COMPETING_SPECIFIC,
        )
        kernel_order, kernel_margin = endotype_kernel._anchor_order(
            class_means,
            variance,
        )
        assert np.array_equal(actual_order, expected_order)
        assert np.array_equal(kernel_order, expected_order)
        assert np.float64(actual_margin).view(np.uint64) == np.float64(
            expected_margin
        ).view(np.uint64)
        assert np.float64(kernel_margin).view(np.uint64) == np.float64(
            expected_margin
        ).view(np.uint64)

        oriented = transport_kernel._anchor_fit(
            class_probability,
            class_means,
            variance,
            responsibility,
        )
        assert np.array_equal(oriented[0], class_probability[expected_order])
        assert np.array_equal(oriented[1], class_means[expected_order])
        assert np.array_equal(oriented[2], responsibility[:, expected_order])
        assert np.float64(oriented[3]).view(np.uint64) == np.float64(
            expected_margin
        ).view(np.uint64)
