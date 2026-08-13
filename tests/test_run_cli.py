"""Focused contracts for the root command-line orchestration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

import run
from configs.counterfactual import CONFIG as COUNTERFACTUAL_CONFIG
from configs.endotype_discovery import CONFIG as ENDOTYPE_CONFIG
from configs.model_comparison import CONFIG as MODEL_COMPARISON_CONFIG
from configs.transportability import CONFIG as TRANSPORT_CONFIG
from traceesus.core.io import sha256_file, write_manifest


@pytest.mark.parametrize(
    "config",
    (
        COUNTERFACTUAL_CONFIG,
        ENDOTYPE_CONFIG,
        MODEL_COMPARISON_CONFIG,
        TRANSPORT_CONFIG,
    ),
)
def test_manifest_config_round_trip_restores_frozen_dataclasses(config: object) -> None:
    """Keep figure regeneration tied to the run's nested immutable config types."""

    serialized = json.loads(json.dumps(asdict(config)))
    restored = run._restore_config_like(config, serialized)
    assert restored == config
    assert type(restored) is type(config)


@dataclass(frozen=True)
class _SmokeConfig:
    """Minimal config used to isolate figure-only manifest behavior."""

    seed: int = 91
    repeats_per_level: int = 500

    def validate(self) -> None:
        """Reject an invalid smoke repeat count before the fake plot runs."""

        if self.repeats_per_level < 2:
            raise ValueError("At least two repeats are required.")


class _FigureOnlyExperiment:
    """Test double that mutates only one figure artifact."""

    def __init__(self, config: _SmokeConfig, output_directory: Path) -> None:
        self.config = config
        self.output_directory = Path(output_directory)

    def configure(self) -> _SmokeConfig:
        """Mirror the real facade's validation and directory creation."""

        self.config.validate()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return self.config

    def figures_only(self) -> None:
        """Represent regenerated figure bytes without invoking Matplotlib."""

        (self.output_directory / "figure.png").write_bytes(b"regenerated figure")


def test_figures_preserves_smoke_config_and_experiment_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent figure-only work from relabeling a smoke run as 500 repeats."""

    smoke = _SmokeConfig(repeats_per_level=2)
    (tmp_path / "figure.png").write_bytes(b"original figure")
    original_runtime = 12.5
    write_manifest(
        tmp_path,
        experiment="model_comparison",
        config=smoke,
        master_seed=smoke.seed,
        wall_clock_runtime_seconds=original_runtime,
    )

    def factory(*, repeats: int | None, workers: int | None, output: Path) -> object:
        assert repeats is None
        assert workers is None
        return _FigureOnlyExperiment(_SmokeConfig(), output)

    monkeypatch.setitem(run.FACTORIES, "model_comparison", factory)
    status = run._figures(
        argparse.Namespace(experiment="model_comparison", out=tmp_path)
    )
    assert status == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"] == asdict(smoke)
    assert manifest["master_seed"] == smoke.seed
    assert manifest["wall_clock_runtime_seconds"] == original_runtime
    assert manifest["last_operation"] == "figures_only"
    assert manifest["last_operation_runtime_seconds"] >= 0.0
    assert manifest["output_file_sha256"] == {
        "figure.png": sha256_file(tmp_path / "figure.png")
    }
