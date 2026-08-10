"""Validated configuration contracts.

Experiment-specific dataclasses retain their historical field names and values;
this module supplies the common validation lifecycle without normalizing any
numerical control.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidatedConfig(ABC):
    """Base for frozen configs that reject invalid states at construction.

    Early validation matters here because a malformed configuration must fail
    before a seed sequence is spawned or an output directory is partially
    populated.
    """

    def __post_init__(self) -> None:
        self.validate()

    @abstractmethod
    def validate(self) -> None:
        """Reject values that violate the experiment's prespecified design."""


@dataclass(frozen=True)
class ParallelConfig(ValidatedConfig):
    """Minimal shared seed and worker controls for repeat-based experiments."""

    master_seed: int
    workers: int = 1

    def validate(self) -> None:
        """Require a nonnegative seed and at least one ordered worker."""

        if self.master_seed < 0:
            raise ValueError("master_seed must be nonnegative.")
        if self.workers < 1:
            raise ValueError("workers must be at least one.")
