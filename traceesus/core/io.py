"""Compatibility writers and additive reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from traceesus import __version__


# Provenance belongs to the code that produced an artifact, not to the chosen
# output directory (which may deliberately live under /tmp or another mount).
CODE_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: Any) -> None:
    """Write legacy JSON formatting exactly: UTF-8 and two-space indentation."""

    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_standard_tables(
    output_directory: Path,
    experiment_name: str,
    raw_long: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    """Write the additive tidy output contract without replacing legacy tables."""

    raw_long.to_csv(
        output_directory / f"{experiment_name}_raw_long.csv", index=False
    )
    summary.to_csv(
        output_directory / f"{experiment_name}_summary.csv", index=False
    )
    contrasts.to_csv(
        output_directory / f"{experiment_name}_contrasts.csv", index=False
    )


def sha256_file(path: Path) -> str:
    """Hash one completed artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(start: Path) -> tuple[str | None, str]:
    """Describe the checkout containing the executing code without failing a run."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, "git_unavailable"
    if completed.returncode == 0:
        return completed.stdout.strip(), "available"
    try:
        probe = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, "git_unavailable"
    return (None, "unborn_repository" if probe.returncode == 0 else "no_repository")


def write_manifest(
    output_directory: Path,
    *,
    experiment: str,
    config: Any,
    master_seed: int,
    wall_clock_runtime_seconds: float,
) -> dict[str, Any]:
    """Write an additive provenance record after every compatibility artifact.

    The manifest excludes itself from its checksum map.  An absent or unborn Git
    repository is represented explicitly rather than causing a completed
    scientific run to fail during provenance collection.
    """

    commit, git_state = _git_state(CODE_CHECKOUT_ROOT)
    checksums = {
        str(path.relative_to(output_directory)): sha256_file(path)
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    serialized_config = asdict(config) if is_dataclass(config) else config
    manifest = {
        "experiment": experiment,
        "config": serialized_config,
        "master_seed": master_seed,
        "package_version": __version__,
        "git_commit": commit,
        "git_state": git_state,
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": version("matplotlib"),
        },
        "wall_clock_runtime_seconds": float(wall_clock_runtime_seconds),
        "output_file_sha256": checksums,
    }
    write_json(output_directory / "manifest.json", manifest)
    return manifest
