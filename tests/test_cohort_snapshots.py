"""Lock five raw two-mechanism cohorts before simulator code is moved."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from configs.endotype_discovery import CONFIG
from traceesus.simulators.two_mechanism import TwoMechanismSimulator


SNAPSHOT_VERSION = 1
PATIENT_COUNT = 128
RENAL_EFFECT_SD = 1.50
SEEDS = (
    16_189_921_146_218_001_160,
    15_513_242_208_653_705_928,
    12_682_294_075_236_838_282,
    13_107_041_859_235_158_231,
    6_981_041_078_731_962_955,
)
SNAPSHOT_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / f"two_mechanism_cohorts_v{SNAPSHOT_VERSION}.npz"
)


@pytest.fixture(scope="module")
def snapshot() -> np.lib.npyio.NpzFile:
    """Load the immutable versioned generator snapshot without object arrays."""

    with np.load(SNAPSHOT_PATH, allow_pickle=False) as stored:
        yield stored


def test_snapshot_records_its_generation_contract(
    snapshot: np.lib.npyio.NpzFile,
) -> None:
    """Keep seed roots, cohort size, effect level, and version explicit."""

    np.testing.assert_array_equal(snapshot["version"], np.array([SNAPSHOT_VERSION]))
    np.testing.assert_array_equal(snapshot["seeds"], np.asarray(SEEDS, dtype=np.uint64))
    np.testing.assert_array_equal(snapshot["patient_count"], np.array([PATIENT_COUNT]))
    np.testing.assert_array_equal(snapshot["renal_effect_sd"], np.array([RENAL_EFFECT_SD]))


@pytest.mark.parametrize("index,seed", tuple(enumerate(SEEDS)))
def test_raw_cohort_arrays_are_bit_identical(
    index: int,
    seed: int,
    snapshot: np.lib.npyio.NpzFile,
) -> None:
    """Regenerate one cohort and compare all four raw arrays without tolerance."""

    simulator = TwoMechanismSimulator(
        config=CONFIG.simulation,
        renal_effect_sd=RENAL_EFFECT_SD,
        heart_failure_effect_sd=0.0,
    )
    generated = simulator.simulate(np.random.default_rng(seed), PATIENT_COUNT)
    arrays = {
        "biomarkers": generated.observed.biomarkers,
        "renal": generated.observed.covariate("renal_dysfunction"),
        "heart_failure": generated.observed.covariate("heart_failure"),
        "mechanism": generated.truth.mechanism,
    }
    for name, actual in arrays.items():
        np.testing.assert_array_equal(
            snapshot[f"seed_{index}_{name}"],
            actual,
            err_msg=f"seed={seed}, array={name}",
        )
