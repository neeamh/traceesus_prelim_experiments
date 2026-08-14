"""Re-run retained experiments and compare every generated CSV cell exactly."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "trace-esus-golden-cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import numpy as np
import pandas as pd
import pytest

from configs.counterfactual import CONFIG as COUNTERFACTUAL_CONFIG
from configs.endotype_discovery import CONFIG as ENDOTYPE_CONFIG
from configs.transportability import CONFIG as TRANSPORT_CONFIG
from traceesus.experiments.counterfactual import CounterfactualExperiment
from traceesus.experiments.endotype_discovery import EndotypeDiscoveryExperiment
from traceesus.experiments.transportability import TransportabilityExperiment


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCKED_ROOT = REPOSITORY_ROOT / "outputs_locked"


@dataclass(frozen=True)
class ExperimentSpecification:
    """Bind one retained facade to its locked output directory."""

    name: str
    output_directory: str
    config: Any
    experiment_type: type
    additive_csvs: tuple[str, ...]


EXPERIMENTS = (
    ExperimentSpecification(
        "endotype_discovery",
        "outputs_latent_endotyping",
        ENDOTYPE_CONFIG,
        EndotypeDiscoveryExperiment,
        (
            "endotype_discovery_contrasts.csv",
            "endotype_discovery_example_patient.csv",
            "endotype_discovery_raw_long.csv",
            "endotype_discovery_summary.csv",
        ),
    ),
    ExperimentSpecification(
        "transportability",
        "outputs_transportability",
        TRANSPORT_CONFIG,
        TransportabilityExperiment,
        (
            "transportability_contrasts.csv",
            "transportability_raw_long.csv",
            "transportability_summary.csv",
        ),
    ),
    ExperimentSpecification(
        "counterfactual",
        "outputs",
        COUNTERFACTUAL_CONFIG,
        CounterfactualExperiment,
        (
            "counterfactual_contrasts.csv",
            "counterfactual_raw_long.csv",
            "counterfactual_summary.csv",
            "data_dictionary.csv",
        ),
    ),
)


def _relative_csvs(directory: Path) -> tuple[str, ...]:
    """Return the complete recursive CSV inventory in stable path order."""

    return tuple(
        path.relative_to(directory).as_posix()
        for path in sorted(directory.rglob("*.csv"))
    )


def _assert_column_exact(
    expected: pd.Series,
    actual: pd.Series,
    *,
    relative_path: str,
) -> None:
    """Compare one column and identify the first exact value mismatch."""

    try:
        np.testing.assert_array_equal(expected.to_numpy(), actual.to_numpy())
    except AssertionError as error:
        expected_values = expected.to_numpy()
        actual_values = actual.to_numpy()
        equal = (expected_values == actual_values) | (
            pd.isna(expected_values) & pd.isna(actual_values)
        )
        row = int(np.flatnonzero(~np.asarray(equal, dtype=bool))[0])
        pytest.fail(
            f"{relative_path}: column={expected.name!r}, row={row}, "
            f"locked={expected_values[row]!r}, actual={actual_values[row]!r}\n{error}"
        )


def _assert_csv_exact(expected_path: Path, actual_path: Path, relative_path: str) -> None:
    """Require identical CSV shape, schema, dtypes, and cell values."""

    expected = pd.read_csv(expected_path)
    actual = pd.read_csv(actual_path)
    assert actual.shape == expected.shape, (
        f"{relative_path}: locked shape={expected.shape}, actual shape={actual.shape}"
    )
    assert list(actual.columns) == list(expected.columns), (
        f"{relative_path}: locked columns={list(expected.columns)!r}, "
        f"actual columns={list(actual.columns)!r}"
    )
    assert actual.dtypes.equals(expected.dtypes), (
        f"{relative_path}: locked dtypes={expected.dtypes.to_dict()!r}, "
        f"actual dtypes={actual.dtypes.to_dict()!r}"
    )
    for column in expected.columns:
        _assert_column_exact(expected[column], actual[column], relative_path=relative_path)
    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_exact=True,
        check_dtype=True,
        check_like=False,
        obj=relative_path,
    )


@pytest.fixture(scope="session", params=EXPERIMENTS, ids=lambda item: item.name)
def golden_run(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, tuple[str, ...]]:
    """Execute one full proposal configuration in an isolated output directory."""

    specification = request.param
    run_root = tmp_path_factory.mktemp(f"golden-{specification.name}")
    candidate = run_root / specification.output_directory
    specification.experiment_type(specification.config, candidate).execute()
    return (
        LOCKED_ROOT / specification.output_directory,
        candidate,
        specification.additive_csvs,
    )


def test_every_output_csv_matches_locked_exactly(
    golden_run: tuple[Path, Path, tuple[str, ...]],
) -> None:
    """Require every locked CSV exactly and only the named additive tables."""

    locked, candidate, additive_csvs = golden_run
    locked_csvs = _relative_csvs(locked)
    candidate_csvs = _relative_csvs(candidate)
    expected_csvs = tuple(sorted((*locked_csvs, *additive_csvs)))
    assert candidate_csvs == expected_csvs, (
        f"{candidate.name}: locked CSVs={locked_csvs!r}, "
        f"actual CSVs={candidate_csvs!r}"
    )
    for relative_path in locked_csvs:
        _assert_csv_exact(
            locked / relative_path,
            candidate / relative_path,
            relative_path,
        )
