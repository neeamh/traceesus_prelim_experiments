"""Ordered parallel execution and legacy seed ledgers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np

from traceesus.core.seeds import (
    ENDOTYPE_NULL_SEED_OFFSET,
    REDUNDANCY_SWEEP_SEED_OFFSET,
    TRANSPORT_SEED_OFFSET,
)

T = TypeVar("T")
R = TypeVar("R")


def ordered_map(worker: Callable[[T], R], tasks: Sequence[T], workers: int) -> Iterable[R]:
    """Map tasks in input order so scheduling cannot reorder reductions."""

    if workers == 1:
        return map(worker, tasks)
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        return list(
            executor.map(
                worker,
                tasks,
                chunksize=max(1, len(tasks) // (20 * workers)),
            )
        )
    finally:
        executor.shutdown(wait=True)


def spawned_uint64_seeds(root: np.random.SeedSequence, count: int) -> list[int]:
    """Derive per-repeat seeds with the legacy one-word uint64 conversion."""

    return [
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in root.spawn(count)
    ]


def nested_seed_sequence_ledger(
    master_seed: int,
    level_count: int,
    repeats_per_level: int,
) -> list[list[np.random.SeedSequence]]:
    """Spawn the exact level-then-repeat SeedSequence tree used by two studies.

    These children must remain ``SeedSequence`` objects: the known-SCM
    experiment spawns additional data and tie descendants from each repeat.
    Converting them through integers would define a different tree.
    """

    levels = np.random.SeedSequence(master_seed).spawn(level_count)
    return [level.spawn(repeats_per_level) for level in levels]


def seed_sequence_ledger(
    master_seed: int,
    repeat_count: int,
) -> list[np.random.SeedSequence]:
    """Spawn flat repeat children when descendants must retain SeedSequence state."""

    return np.random.SeedSequence(master_seed).spawn(repeat_count)


def latent_recovery_seed_ledger(
    master_seed: int,
    level_count: int,
    repeats_per_level: int,
) -> list[list[int]]:
    """Return the exact nested level/repeat seed ledger for latent recovery."""

    levels = np.random.SeedSequence(master_seed).spawn(level_count)
    return [spawned_uint64_seeds(level, repeats_per_level) for level in levels]


def latent_null_seed_ledger(master_seed: int, repeats: int) -> list[int]:
    """Return the exact K=1 null ledger using the named null offset."""

    return spawned_uint64_seeds(
        np.random.SeedSequence(master_seed + ENDOTYPE_NULL_SEED_OFFSET), repeats
    )


def redundancy_sweep_seed_ledger(
    master_seed: int,
    level_count: int,
    repeats_per_level: int,
) -> list[list[int]]:
    """Return HF-sweep seeds rooted at the named redundancy offset.

    The distinct salt guarantees the sweep shares no seed with the locked
    recovery ledger or the K=1 null, so redundancy results can never be
    mistaken for — or perturb — a proposal-cited artifact.
    """

    levels = np.random.SeedSequence(
        master_seed + REDUNDANCY_SWEEP_SEED_OFFSET
    ).spawn(level_count)
    return [spawned_uint64_seeds(level, repeats_per_level) for level in levels]


def transport_seed_ledger(master_seed: int, repeats: int) -> list[int]:
    """Return transport repeat seeds rooted at the named transport offset."""

    return spawned_uint64_seeds(
        np.random.SeedSequence(master_seed + TRANSPORT_SEED_OFFSET), repeats
    )
