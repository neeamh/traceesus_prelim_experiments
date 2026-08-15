"""Non-numerical contracts that prevent the package facade from overclaiming."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pytest

from configs.endotype_discovery import (
    CONFIG as ENDOTYPE_CONFIG,
    ExperimentConfig as EndotypeExperimentConfig,
    FittingConfig,
    SimulationConfig,
)
from configs.transportability import (
    CONFIG as TRANSPORT_CONFIG,
    HospitalSpec,
    TransportExperimentConfig,
    TransportSimulationConfig,
)
from traceesus.experiments.endotype_discovery import kernel as endotype_kernel
from traceesus.experiments.transportability import kernel as transport_kernel
from traceesus.models.modular_causal_scm import (
    PooledAssociativeTransportModel,
    TransportModelConfig,
    _fitting_config,
)
from traceesus.plotting.style import DEFAULT_FONT, FIGURE_SIZES_INCHES, PALETTE
from traceesus.queries.posterior import anchor_order


@pytest.mark.parametrize(
    "config",
    (
        ENDOTYPE_CONFIG,
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
        SimulationConfig,
        FittingConfig,
        EndotypeExperimentConfig,
        HospitalSpec,
        TransportSimulationConfig,
        TransportExperimentConfig,
    )
    assert all(config_type.__module__.startswith("configs.") for config_type in classes)

    # Kernels re-export the same class objects for legacy scripts and notebooks.
    assert endotype_kernel.ExperimentConfig is EndotypeExperimentConfig
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


def test_plotting_package_contains_style_only() -> None:
    """Keep every rendering implementation outside the importable package."""

    plotting = Path(__file__).resolve().parents[1] / "traceesus" / "plotting"
    assert {path.name for path in plotting.glob("*.py")} == {"__init__.py", "style.py"}
    assert not hasattr(endotype_kernel, "plot_primary_figure")
    assert not hasattr(transport_kernel, "plot_transport_figure")
    assert DEFAULT_FONT == "DejaVu Sans"
    assert PALETTE["associative"] == "#D97706"
    assert PALETTE["causal"] == "#1D4ED8"
    assert FIGURE_SIZES_INCHES["endotype_recovery"] == (12.2, 5.1)
    assert FIGURE_SIZES_INCHES["transportability_primary"] == (12.4, 5.2)


def test_only_style_module_imports_matplotlib() -> None:
    """Enforce the computation/presentation import boundary mechanically."""

    package = Path(__file__).resolve().parents[1] / "traceesus"
    importers = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if "import matplotlib" in path.read_text(encoding="utf-8")
        or "from matplotlib" in path.read_text(encoding="utf-8")
    }
    assert importers == {"plotting/style.py"}


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
