"""Experiment lifecycle shared by all four study designs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Any

from .io import write_json


class Experiment(ABC):
    """Run a reproducible configure-run-summarize-plot-write lifecycle.

    The fixed lifecycle makes side effects auditable: numerical work is complete
    before plotting and output persistence, and elapsed time is available to the
    manifest writer without entering any simulation calculation.
    """

    wall_clock_runtime_seconds: float = 0.0

    def execute(self) -> Any:
        """Execute the required lifecycle in order and return the final artifacts."""

        started = perf_counter()
        self.configure()
        self.run()
        self.summarize()
        self.plot()
        self.wall_clock_runtime_seconds = perf_counter() - started
        result = self.write()
        self.wall_clock_runtime_seconds = perf_counter() - started
        if isinstance(result, dict) and isinstance(result.get("manifest"), dict):
            manifest = result["manifest"]
            manifest["wall_clock_runtime_seconds"] = self.wall_clock_runtime_seconds
            output_directory = getattr(self, "output_directory", None)
            if output_directory is None:
                raise RuntimeError(
                    "An experiment returning a manifest must expose output_directory."
                )
            write_json(Path(output_directory) / "manifest.json", manifest)
        return result

    @abstractmethod
    def configure(self) -> Any:
        """Validate and expose the immutable experiment configuration."""

    @abstractmethod
    def run(self) -> Any:
        """Execute repeat-level simulations and fits without output mutation."""

    @abstractmethod
    def summarize(self) -> Any:
        """Apply the historical reductions in their original order."""

    @abstractmethod
    def plot(self) -> Any:
        """Create figures from completed summaries without rerunning models."""

    @abstractmethod
    def write(self) -> Any:
        """Persist compatibility artifacts and the additive run manifest."""
