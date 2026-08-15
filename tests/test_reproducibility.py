"""Proposal-lock regression tests for the retained TRACE-ESUS facades.

The expected hashes in ``fixtures/reproducibility_expected.json`` were created
once from package runs that had already been checked against their legacy
scripts.  Tests deliberately have no fixture-regeneration path: changing a
number requires an explicit scientific review and an intentional fixture edit.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# Configure Matplotlib before any experiment kernel imports it.  Both locations
# are writable in sandboxed and CI runs and avoid mutating a researcher's home.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "trace-esus-pytest-cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import numpy as np
import pytest

from configs.endotype_discovery import CONFIG as ENDOTYPE_CONFIG
from configs.transportability import CONFIG as TRANSPORT_CONFIG
from traceesus.core.io import sha256_file
from traceesus.core.model import Model, assert_truth_free_fit_interfaces
from traceesus.core.seeds import ENDOTYPE_RECOVERY_SEED_ROOT
from traceesus.core.runner import (
    latent_null_seed_ledger,
    latent_recovery_seed_ledger,
    ordered_map,
    transport_seed_ledger,
)
from traceesus.core.simulator import Cohort
from traceesus.experiments.endotype_discovery import EndotypeDiscoveryExperiment
from traceesus.experiments.transportability import TransportabilityExperiment
from traceesus.registry import LOCKED_MODEL_SET
from traceesus.models import (
    AdjustedLatentClassModel,
    AssociativeLatentClassModel,
    BiologicallyConstrainedCausalSCM,
)
from traceesus.models.modular_causal_scm import (
    FrozenCausalSCM,
    ModularCausalSCM,
    PooledAssociativeTransportModel,
    TargetAdjustedAssociativeModel,
    TargetTransportOracle,
)


EXPECTED_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "reproducibility_expected.json"
)
EXPECTED = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))


def _ordered_map_worker(task: tuple[int, int]) -> tuple[int, int]:
    """Return a deterministic value while retaining input identity and order."""

    index, value = task
    return index, value * value + 3


def _legacy_hashes(output_directory: Path) -> dict[str, str]:
    """Hash every recursive compatibility CSV/JSON except the new manifest."""

    return {
        path.relative_to(output_directory).as_posix(): sha256_file(path)
        for path in sorted(output_directory.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and path.suffix.lower() in {".csv", ".json"}
    }


def _manifest_inventory(output_directory: Path) -> dict[str, str]:
    """Recompute the complete post-run file inventory recorded by a manifest."""

    return {
        path.relative_to(output_directory).as_posix(): sha256_file(path)
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


@pytest.fixture(scope="session")
def reproducibility_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, Any]]:
    """Execute every retained facade once with two repeats and other defaults intact."""

    root = tmp_path_factory.mktemp("trace-esus-reproducibility")
    specifications = {
        "endotype_discovery": (
            replace(
                ENDOTYPE_CONFIG,
                repeats_per_level=2,
                null_repeats=2,
                workers=1,
            ),
            EndotypeDiscoveryExperiment,
        ),
        "transportability": (
            replace(TRANSPORT_CONFIG, repeats=2, workers=1),
            TransportabilityExperiment,
        ),
    }

    completed: dict[str, dict[str, Any]] = {}
    for name, (config, experiment_type) in specifications.items():
        output_directory = root / EXPECTED["experiments"][name]["output_directory"]
        if name == "endotype_discovery":
            experiment = experiment_type(config, output_directory, LOCKED_MODEL_SET)
        else:
            experiment = experiment_type(config, output_directory)
        result = experiment.execute()
        completed[name] = {
            "config": config,
            "directory": output_directory,
            "result": result,
        }
    return completed


@pytest.mark.parametrize(
    "experiment_name", ("endotype_discovery", "transportability")
)
def test_every_legacy_csv_and_json_has_locked_sha256(
    experiment_name: str,
    reproducibility_runs: dict[str, dict[str, Any]],
) -> None:
    """Reject a byte change in any legacy table or metadata/check JSON."""

    run = reproducibility_runs[experiment_name]
    expected = EXPECTED["experiments"][experiment_name]["legacy_file_sha256"]
    actual = _legacy_hashes(run["directory"])
    assert actual == expected


@pytest.mark.parametrize(
    "experiment_name", ("endotype_discovery", "transportability")
)
def test_manifest_inventory_and_checksums_are_complete(
    experiment_name: str,
    reproducibility_runs: dict[str, dict[str, Any]],
) -> None:
    """Require the manifest to cover every artifact with its current checksum."""

    run = reproducibility_runs[experiment_name]
    manifest_path = run["directory"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = EXPECTED["experiments"][experiment_name]

    assert manifest["experiment"] == experiment_name
    assert manifest["master_seed"] == expected["master_seed"]
    # The manifest is JSON, so immutable dataclass tuples are represented as
    # arrays after round-tripping.  Compare the same serialized data model.
    serialized_config = json.loads(json.dumps(asdict(run["config"])))
    if experiment_name == "endotype_discovery":
        serialized_config["simulation"].pop("heart_failure_prevalence")
        serialized_config["simulation"].pop("heart_failure_effect_levels_sd")
    assert manifest["config"] == serialized_config
    assert np.isfinite(manifest["wall_clock_runtime_seconds"])
    assert manifest["wall_clock_runtime_seconds"] >= 0.0
    assert manifest["output_file_sha256"] == _manifest_inventory(run["directory"])


def test_seed_ledgers_match_locked_sentinels() -> None:
    """Lock the audited uint64 repeat seeds and nested target seed streams."""

    assert latent_recovery_seed_ledger(ENDOTYPE_RECOVERY_SEED_ROOT, 4, 2) == [
        [16_189_921_146_218_001_160, 15_513_242_208_653_705_928],
        [12_682_294_075_236_838_282, 13_107_041_859_235_158_231],
        [6_981_041_078_731_962_955, 13_358_525_377_037_369_752],
        [13_533_547_887_973_413_680, 4_438_523_213_558_081_714],
    ]
    assert latent_null_seed_ledger(ENDOTYPE_RECOVERY_SEED_ROOT, 2) == [
        6_569_377_144_667_487_967,
        5_967_866_994_814_613_138,
    ]

    transport_seeds = transport_seed_ledger(ENDOTYPE_RECOVERY_SEED_ROOT, 2)
    assert transport_seeds == [
        8_829_251_784_368_036_273,
        18_035_949_171_693_999_477,
    ]
    repeat_zero_children = np.random.SeedSequence(transport_seeds[0]).spawn(12)
    target_calibration_and_test = [
        int(repeat_zero_children[index].generate_state(1, dtype=np.uint64)[0])
        for index in (9, 10)
    ]
    assert target_calibration_and_test == [
        9_046_439_147_895_618_138,
        10_373_056_651_876_582_244,
    ]


def test_ordered_map_is_identical_with_one_and_two_workers() -> None:
    """Verify process scheduling cannot reorder paired repeat results."""

    tasks = [(index, value) for index, value in enumerate((9, 2, 7, 4, 1, 8))]
    sequential = list(ordered_map(_ordered_map_worker, tasks, workers=1))
    try:
        parallel = list(ordered_map(_ordered_map_worker, tasks, workers=2))
    except (OSError, PermissionError) as error:
        pytest.skip(f"Process workers are prohibited by this sandbox: {error}")
    assert parallel == sequential


def test_unsupervised_fit_boundary_excludes_truth() -> None:
    """Keep simulator truth absent from Cohort and every unsupervised fit API."""

    assert set(Cohort.__dataclass_fields__) == {
        "biomarkers",
        "covariates",
        "measurement_indicators",
    }
    forbidden = {"truth", "true_mechanism", "mechanism_labels", "labels"}
    models: tuple[Model, ...] = (
        AssociativeLatentClassModel(),
        AdjustedLatentClassModel(),
        BiologicallyConstrainedCausalSCM(),
        PooledAssociativeTransportModel(),
        TargetAdjustedAssociativeModel(),
        FrozenCausalSCM(),
        ModularCausalSCM(),
        TargetTransportOracle(TRANSPORT_CONFIG.simulation),
    )
    assert_truth_free_fit_interfaces(models)
    for model in models:
        assert forbidden.isdisjoint(inspect.signature(model.fit).parameters)
