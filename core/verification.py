"""Exact, machine-readable comparison of locked and candidate output trees.

The TRACE-ESUS results are cited scientific artifacts.  This module therefore
avoids pandas, approximate floating-point comparisons, row sorting, and JSON
normalization: each of those conveniences can hide a reproducibility failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import scipy

from traceesus import __version__
from traceesus.core.io import CODE_CHECKOUT_ROOT


_MISSING = object()


@dataclass(frozen=True, slots=True)
class Discrepancy:
    """Describe one exact mismatch so failures can be diagnosed automatically.

    A structured location is retained instead of embedding coordinates only in
    prose because a full verification run may need to route thousands of
    mismatches back to the simulator, model, summarizer, or writer responsible.
    """

    relative_path: str
    kind: str
    location: Mapping[str, object] = field(default_factory=dict)
    expected: object = _MISSING
    actual: object = _MISSING
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe record without erasing float bit information."""

        record: dict[str, object] = {
            "path": self.relative_path,
            "kind": self.kind,
            "location": dict(self.location),
            "message": self.message,
        }
        if self.expected is not _MISSING:
            record["expected"] = _json_safe(self.expected)
        else:
            record["expected"] = {"state": "missing"}
        if self.actual is not _MISSING:
            record["actual"] = _json_safe(self.actual)
        else:
            record["actual"] = {"state": "missing"}
        return record


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Collect the complete comparison result rather than stopping at first error.

    Scientific verification needs the full mismatch surface: a single early
    failure does not reveal whether drift starts in raw draws, summaries, JSON
    validation, or rendered outputs.
    """

    locked_root: str
    candidate_root: str
    compared_files: tuple[str, ...]
    allowed_additions: tuple[str, ...]
    ignored_files: tuple[str, ...]
    discrepancies: tuple[Discrepancy, ...]

    @property
    def passed(self) -> bool:
        """Return true only when every required comparison is exact."""

        return not self.discrepancies

    @property
    def exit_code(self) -> int:
        """Expose a process-safe status so CI cannot mistake drift for success."""

        return 0 if self.passed else 1

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete report for CI, audits, and ``VERIFICATION.md``."""

        return {
            "status": "pass" if self.passed else "fail",
            "exit_code": self.exit_code,
            "locked_root": self.locked_root,
            "candidate_root": self.candidate_root,
            "compared_file_count": len(self.compared_files),
            "compared_files": list(self.compared_files),
            "allowed_addition_count": len(self.allowed_additions),
            "allowed_additions": list(self.allowed_additions),
            "ignored_file_count": len(self.ignored_files),
            "ignored_files": list(self.ignored_files),
            "discrepancy_count": len(self.discrepancies),
            "discrepancies": [item.to_dict() for item in self.discrepancies],
        }

    def write_json(self, path: str | Path) -> None:
        """Persist an audit artifact whose contents are strict RFC-compatible JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def compare_output_trees(
    locked_root: str | Path,
    candidate_root: str | Path,
    *,
    compare_png_pixels: bool = True,
    allow_additive_manifests: bool = True,
    legacy_data_only: bool = False,
    include_suffixes: Collection[str] | None = None,
    pdf_mode: str = "ignore",
) -> VerificationReport:
    """Compare a candidate output tree against an immutable locked tree.

    CSV cells and JSON values receive format-aware exact comparisons; PNGs may
    be compared after decoding so harmless encoder metadata does not obscure a
    pixel change.  PDFs are ignored by default because creation dates and
    renderer metadata make their encoded bytes a false numerical gate.  Set
    ``legacy_data_only`` to compare only legacy CSV/JSON artifacts, or pass an
    explicit suffix set.  New per-run ``manifest.json`` files are permitted
    only after their provenance and checksums have been validated.
    """

    locked = Path(locked_root).expanduser().resolve()
    candidate = Path(candidate_root).expanduser().resolve()
    discrepancies: list[Discrepancy] = []

    if legacy_data_only and include_suffixes is not None:
        raise ValueError("legacy_data_only and include_suffixes are mutually exclusive")
    if pdf_mode not in {"ignore", "bytes"}:
        raise ValueError("pdf_mode must be 'ignore' or 'bytes'")
    selected_suffixes = (
        {".csv", ".json"}
        if legacy_data_only
        else _normalize_suffixes(include_suffixes)
    )

    if not locked.exists():
        discrepancies.append(
            Discrepancy(
                relative_path=".",
                kind="locked_root_missing",
                expected="existing directory containing locked outputs",
                actual=_MISSING,
                message="The locked baseline does not exist; verification cannot pass.",
            )
        )
        locked_files: dict[str, Path] = {}
    elif not locked.is_dir():
        discrepancies.append(
            Discrepancy(
                relative_path=".",
                kind="locked_root_not_directory",
                expected="directory",
                actual=_path_kind(locked),
                message="The locked baseline must be a directory tree.",
            )
        )
        locked_files = {}
    else:
        locked_files = _files_by_relative_path(locked)
        if not locked_files:
            discrepancies.append(
                Discrepancy(
                    relative_path=".",
                    kind="locked_tree_empty",
                    expected="at least one locked file",
                    actual=0,
                    message="An empty locked tree is not evidence of reproducibility.",
                )
            )

    if not candidate.exists():
        discrepancies.append(
            Discrepancy(
                relative_path=".",
                kind="candidate_root_missing",
                expected="existing directory containing candidate outputs",
                actual=_MISSING,
                message="The candidate output directory does not exist.",
            )
        )
        candidate_files: dict[str, Path] = {}
    elif not candidate.is_dir():
        discrepancies.append(
            Discrepancy(
                relative_path=".",
                kind="candidate_root_not_directory",
                expected="directory",
                actual=_path_kind(candidate),
                message="The candidate outputs must be a directory tree.",
            )
        )
        candidate_files = {}
    else:
        candidate_files = _files_by_relative_path(candidate)

    compared_files: list[str] = []
    allowed_additions: list[str] = []
    ignored_files: list[str] = []

    for relative_path in sorted(locked_files):
        expected_path = locked_files[relative_path]
        suffix = expected_path.suffix.lower()
        if selected_suffixes is not None and suffix not in selected_suffixes:
            ignored_files.append(relative_path)
            continue
        actual_path = candidate_files.get(relative_path)
        if actual_path is None:
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="required_file_missing",
                    expected="file present",
                    actual=_MISSING,
                    message="Every locked artifact is required in the candidate tree.",
                )
            )
            continue

        if suffix == ".pdf" and pdf_mode == "ignore":
            ignored_files.append(relative_path)
            continue

        compared_files.append(relative_path)
        if suffix == ".csv":
            _compare_csv(expected_path, actual_path, relative_path, discrepancies)
        elif suffix == ".json":
            _compare_json(expected_path, actual_path, relative_path, discrepancies)
        elif suffix == ".png" and compare_png_pixels:
            _compare_png_pixels(expected_path, actual_path, relative_path, discrepancies)
        else:
            _compare_bytes(expected_path, actual_path, relative_path, discrepancies)

    for relative_path in sorted(set(candidate_files) - set(locked_files)):
        if allow_additive_manifests and Path(relative_path).name == "manifest.json":
            allowed_additions.append(relative_path)
            continue
        if (
            selected_suffixes is not None
            and Path(relative_path).suffix.lower() not in selected_suffixes
        ):
            ignored_files.append(relative_path)
            continue
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="unexpected_file",
                expected=_MISSING,
                actual="file present",
                message="Only additive manifest.json files are permitted.",
            )
        )

    if allow_additive_manifests:
        for relative_path in allowed_additions:
            _validate_manifest(
                manifest_path=candidate_files[relative_path],
                relative_path=relative_path,
                locked_root=locked,
                candidate_root=candidate,
                discrepancies=discrepancies,
            )

    return VerificationReport(
        locked_root=str(locked),
        candidate_root=str(candidate),
        compared_files=tuple(compared_files),
        allowed_additions=tuple(allowed_additions),
        ignored_files=tuple(sorted(set(ignored_files))),
        discrepancies=tuple(discrepancies),
    )


def verify_outputs(
    locked_root: str | Path,
    candidate_root: str | Path,
    *,
    compare_png_pixels: bool = True,
    allow_additive_manifests: bool = True,
    legacy_data_only: bool = False,
    include_suffixes: Collection[str] | None = None,
    pdf_mode: str = "ignore",
) -> VerificationReport:
    """Provide a verb-oriented alias for callers implementing ``run.py verify``."""

    return compare_output_trees(
        locked_root,
        candidate_root,
        compare_png_pixels=compare_png_pixels,
        allow_additive_manifests=allow_additive_manifests,
        legacy_data_only=legacy_data_only,
        include_suffixes=include_suffixes,
        pdf_mode=pdf_mode,
    )


def verify_output_trees(
    locked_root: str | Path,
    candidate_root: str | Path,
    *,
    compare_png_pixels: bool = True,
    allow_additive_manifests: bool = True,
    legacy_data_only: bool = False,
    include_suffixes: Collection[str] | None = None,
    pdf_mode: str = "ignore",
) -> VerificationReport:
    """Retain an explicit-name alias for integration code and external tooling."""

    return compare_output_trees(
        locked_root,
        candidate_root,
        compare_png_pixels=compare_png_pixels,
        allow_additive_manifests=allow_additive_manifests,
        legacy_data_only=legacy_data_only,
        include_suffixes=include_suffixes,
        pdf_mode=pdf_mode,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the verifier as a JSON-emitting command and return a strict exit code."""

    parser = argparse.ArgumentParser(
        description="Compare TRACE-ESUS candidate outputs against locked outputs exactly."
    )
    parser.add_argument("locked_root", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument(
        "--png-mode",
        choices=("pixels", "bytes"),
        default="pixels",
        help="Compare decoded RGBA pixels (default) or encoded PNG bytes.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--legacy-data-only",
        action="store_true",
        help="Compare only exact legacy CSV cells and JSON values.",
    )
    selection.add_argument(
        "--include-suffix",
        action="append",
        help="Limit comparison to this suffix; repeat for multiple suffixes.",
    )
    parser.add_argument(
        "--pdf-mode",
        choices=("ignore", "bytes"),
        default="ignore",
        help="Ignore PDF encoding drift (default) or compare encoded bytes.",
    )
    parser.add_argument(
        "--reject-additive-manifests",
        action="store_true",
        help="Treat candidate-only manifest.json files as unexpected.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Also write the complete machine-readable report to this path.",
    )
    args = parser.parse_args(argv)

    report = compare_output_trees(
        args.locked_root,
        args.candidate_root,
        compare_png_pixels=args.png_mode == "pixels",
        allow_additive_manifests=not args.reject_additive_manifests,
        legacy_data_only=args.legacy_data_only,
        include_suffixes=args.include_suffix,
        pdf_mode=args.pdf_mode,
    )
    if args.report is not None:
        report.write_json(args.report)
    json.dump(report.to_dict(), sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return report.exit_code


def _normalize_suffixes(suffixes: Collection[str] | None) -> set[str] | None:
    if suffixes is None:
        return None
    normalized: set[str] = set()
    for suffix in suffixes:
        value = suffix.lower()
        normalized.add(value if value.startswith(".") else f".{value}")
    if not normalized:
        raise ValueError("include_suffixes must contain at least one suffix")
    return normalized


def _validate_manifest(
    *,
    manifest_path: Path,
    relative_path: str,
    locked_root: Path,
    candidate_root: Path,
    discrepancies: list[Discrepancy],
) -> None:
    """Validate provenance against locked configuration and the live runtime.

    A manifest cannot establish its own seed or configuration.  Those values
    are therefore checked against the nearest legacy metadata file in the
    corresponding locked directory (walking upward supports transportability's
    nested ablation directory).  Checksums are scoped to the directory that
    contains the manifest and include every non-manifest descendant.
    """

    manifest = _read_json(manifest_path, relative_path, "candidate", discrepancies)
    if manifest is _MISSING:
        return
    if not isinstance(manifest, dict):
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="manifest_type_invalid",
                location={"json_pointer": ""},
                expected="JSON object",
                actual=_json_type_name(manifest),
                message="A reproducibility manifest must be a JSON object.",
            )
        )
        return

    required_keys = {
        "experiment",
        "config",
        "master_seed",
        "package_version",
        "git_commit",
        "git_state",
        "package_versions",
        "wall_clock_runtime_seconds",
        "output_file_sha256",
    }
    for key in sorted(required_keys - set(manifest)):
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="manifest_key_missing",
                location={"json_pointer": _json_pointer_child("", key)},
                expected="required key",
                actual=_MISSING,
                message="A required reproducibility-manifest field is absent.",
            )
        )

    if "experiment" in manifest and (
        not isinstance(manifest["experiment"], str) or not manifest["experiment"]
    ):
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="manifest_experiment_invalid",
                location={"json_pointer": "/experiment"},
                expected="non-empty string",
                actual=manifest["experiment"],
                message="The experiment identifier must be explicit.",
            )
        )

    if "config" in manifest and not isinstance(manifest["config"], dict):
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="manifest_config_type_invalid",
                location={"json_pointer": "/config"},
                expected="JSON object",
                actual=_json_type_name(manifest["config"]),
                message="The serialized dataclass configuration must be an object.",
            )
        )

    expected_config = _locked_config_for_manifest(
        locked_root=locked_root,
        manifest_relative_path=relative_path,
        discrepancies=discrepancies,
    )
    if expected_config is not _MISSING and "config" in manifest:
        _compare_json_values(
            expected_config,
            manifest["config"],
            relative_path,
            "/config",
            discrepancies,
        )

    expected_seed = _seed_from_config(expected_config)
    if expected_seed is _MISSING:
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="manifest_seed_reference_missing",
                location={"json_pointer": "/master_seed"},
                expected="top-level master_seed or seed in locked configuration",
                actual=_MISSING,
                message="The manifest seed could not be checked independently.",
            )
        )
    elif "master_seed" in manifest:
        _compare_json_values(
            expected_seed,
            manifest["master_seed"],
            relative_path,
            "/master_seed",
            discrepancies,
        )

    if "package_version" in manifest:
        _compare_json_values(
            __version__,
            manifest["package_version"],
            relative_path,
            "/package_version",
            discrepancies,
        )
    if "package_versions" in manifest:
        _compare_json_values(
            _live_package_versions(),
            manifest["package_versions"],
            relative_path,
            "/package_versions",
            discrepancies,
        )

    expected_commit, expected_git_state = _live_git_state(CODE_CHECKOUT_ROOT)
    if "git_commit" in manifest:
        _compare_json_values(
            expected_commit,
            manifest["git_commit"],
            relative_path,
            "/git_commit",
            discrepancies,
        )
    if "git_state" in manifest:
        _compare_json_values(
            expected_git_state,
            manifest["git_state"],
            relative_path,
            "/git_state",
            discrepancies,
        )

    if "wall_clock_runtime_seconds" in manifest:
        runtime = manifest["wall_clock_runtime_seconds"]
        if (
            isinstance(runtime, bool)
            or not isinstance(runtime, (int, float))
            or not math.isfinite(runtime)
            or runtime < 0.0
        ):
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="manifest_runtime_invalid",
                    location={"json_pointer": "/wall_clock_runtime_seconds"},
                    expected="finite nonnegative number",
                    actual=runtime,
                    message="Wall-clock runtime must be finite and nonnegative.",
                )
            )

    if "output_file_sha256" in manifest:
        _validate_manifest_checksums(
            manifest_path=manifest_path,
            relative_path=relative_path,
            checksum_value=manifest["output_file_sha256"],
            candidate_root=candidate_root,
            discrepancies=discrepancies,
        )


def _locked_config_for_manifest(
    *,
    locked_root: Path,
    manifest_relative_path: str,
    discrepancies: list[Discrepancy],
) -> object:
    manifest_directory = locked_root / Path(manifest_relative_path).parent
    try:
        manifest_directory.relative_to(locked_root)
    except ValueError:
        return _MISSING

    directory = manifest_directory
    while True:
        for name in ("metadata.json", "run_metadata.json"):
            metadata_path = directory / name
            if not metadata_path.is_file():
                continue
            metadata_relative = metadata_path.relative_to(locked_root).as_posix()
            metadata = _read_json(
                metadata_path,
                metadata_relative,
                "locked",
                discrepancies,
            )
            if isinstance(metadata, dict):
                if "experiment_config" in metadata:
                    return metadata["experiment_config"]
                if "config" in metadata:
                    return metadata["config"]
        if directory == locked_root:
            break
        if locked_root not in directory.parents:
            break
        directory = directory.parent

    discrepancies.append(
        Discrepancy(
            relative_path=manifest_relative_path,
            kind="manifest_config_reference_missing",
            location={"json_pointer": "/config"},
            expected="config or experiment_config in nearest locked metadata",
            actual=_MISSING,
            message="The manifest configuration could not be checked independently.",
        )
    )
    return _MISSING


def _seed_from_config(config: object) -> object:
    if not isinstance(config, dict):
        return _MISSING
    if "master_seed" in config:
        return config["master_seed"]
    if "seed" in config:
        return config["seed"]
    return _MISSING


def _live_package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _live_git_state(start: Path) -> tuple[str | None, str]:
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


def _validate_manifest_checksums(
    *,
    manifest_path: Path,
    relative_path: str,
    checksum_value: object,
    candidate_root: Path,
    discrepancies: list[Discrepancy],
) -> None:
    if not isinstance(checksum_value, dict):
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="manifest_checksum_map_invalid",
                location={"json_pointer": "/output_file_sha256"},
                expected="JSON object mapping relative paths to SHA-256 hex digests",
                actual=_json_type_name(checksum_value),
                message="The checksum inventory must be an object.",
            )
        )
        return

    manifest_directory = manifest_path.parent
    actual_files = {
        path.relative_to(manifest_directory).as_posix(): path
        for path in manifest_directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    declared_paths = set(checksum_value)
    actual_paths = set(actual_files)

    for path in sorted(actual_paths - declared_paths):
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="manifest_checksum_entry_missing",
                location={
                    "json_pointer": _json_pointer_child("/output_file_sha256", path),
                    "artifact": (manifest_directory / path)
                    .relative_to(candidate_root)
                    .as_posix(),
                },
                expected=_file_fingerprint(actual_files[path])["sha256"],
                actual=_MISSING,
                message="A non-manifest output is absent from the checksum inventory.",
            )
        )
    for path in sorted(declared_paths - actual_paths):
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="manifest_checksum_entry_unexpected",
                location={"json_pointer": _json_pointer_child("/output_file_sha256", path)},
                expected=_MISSING,
                actual=checksum_value[path],
                message="The checksum inventory names no current output file.",
            )
        )
    for path in sorted(actual_paths & declared_paths):
        declared = checksum_value[path]
        actual_digest = _file_fingerprint(actual_files[path])["sha256"]
        if (
            not isinstance(declared, str)
            or len(declared) != 64
            or any(character not in "0123456789abcdef" for character in declared)
        ):
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="manifest_checksum_invalid",
                    location={"json_pointer": _json_pointer_child("/output_file_sha256", path)},
                    expected="64-character lowercase SHA-256 hexadecimal string",
                    actual=declared,
                    message="A declared checksum has an invalid representation.",
                )
            )
        elif declared != actual_digest:
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="manifest_checksum_mismatch",
                    location={"json_pointer": _json_pointer_child("/output_file_sha256", path)},
                    expected=actual_digest,
                    actual=declared,
                    message="The declared SHA-256 digest does not match the output file.",
                )
            )


def _files_by_relative_path(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }


def _path_kind(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    if path.is_symlink():
        return "symlink"
    return "other"


def _compare_csv(
    expected_path: Path,
    actual_path: Path,
    relative_path: str,
    discrepancies: list[Discrepancy],
) -> None:
    expected_rows = _read_csv(expected_path, relative_path, "locked", discrepancies)
    actual_rows = _read_csv(actual_path, relative_path, "candidate", discrepancies)
    if expected_rows is None or actual_rows is None:
        return

    if len(expected_rows) != len(actual_rows):
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="csv_row_count_mismatch",
                location={"scope": "table"},
                expected=len(expected_rows),
                actual=len(actual_rows),
                message="CSV record counts differ; records were not reordered or sorted.",
            )
        )

    expected_header = expected_rows[0] if expected_rows else []
    actual_header = actual_rows[0] if actual_rows else []
    for row_index in range(max(len(expected_rows), len(actual_rows))):
        record_number = row_index + 1
        if row_index >= len(actual_rows):
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="csv_row_missing",
                    location={"row": record_number},
                    expected=expected_rows[row_index],
                    actual=_MISSING,
                    message="A locked CSV record is absent at this ordered position.",
                )
            )
            continue
        if row_index >= len(expected_rows):
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="csv_row_unexpected",
                    location={"row": record_number},
                    expected=_MISSING,
                    actual=actual_rows[row_index],
                    message="The candidate CSV contains an additional ordered record.",
                )
            )
            continue

        expected_row = expected_rows[row_index]
        actual_row = actual_rows[row_index]
        if len(expected_row) != len(actual_row):
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="csv_column_count_mismatch",
                    location={"row": record_number},
                    expected=len(expected_row),
                    actual=len(actual_row),
                    message="CSV field counts differ within this ordered record.",
                )
            )

        for column_index in range(max(len(expected_row), len(actual_row))):
            location: dict[str, object] = {
                "row": record_number,
                "column": column_index + 1,
            }
            if row_index == 0:
                location["record"] = "header"
            elif column_index < len(expected_header):
                location["column_name"] = expected_header[column_index]
            elif column_index < len(actual_header):
                location["candidate_column_name"] = actual_header[column_index]

            if column_index >= len(actual_row):
                discrepancies.append(
                    Discrepancy(
                        relative_path=relative_path,
                        kind="csv_cell_missing",
                        location=location,
                        expected=expected_row[column_index],
                        actual=_MISSING,
                        message="A locked CSV cell is absent at this row and column.",
                    )
                )
            elif column_index >= len(expected_row):
                discrepancies.append(
                    Discrepancy(
                        relative_path=relative_path,
                        kind="csv_cell_unexpected",
                        location=location,
                        expected=_MISSING,
                        actual=actual_row[column_index],
                        message="The candidate CSV has an additional cell at this position.",
                    )
                )
            elif expected_row[column_index] != actual_row[column_index]:
                discrepancies.append(
                    Discrepancy(
                        relative_path=relative_path,
                        kind="csv_cell_mismatch",
                        location=location,
                        expected=expected_row[column_index],
                        actual=actual_row[column_index],
                        message="Raw CSV cell strings differ exactly.",
                    )
                )


def _read_csv(
    path: Path,
    relative_path: str,
    side: str,
    discrepancies: list[Discrepancy],
) -> list[list[str]] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error) as error:
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="csv_read_error",
                location={"side": side},
                expected="valid UTF-8 CSV" if side == "locked" else _MISSING,
                actual={"error_type": type(error).__name__, "message": str(error)},
                message=f"The {side} CSV could not be parsed exactly.",
            )
        )
        return None


def _compare_json(
    expected_path: Path,
    actual_path: Path,
    relative_path: str,
    discrepancies: list[Discrepancy],
) -> None:
    expected = _read_json(expected_path, relative_path, "locked", discrepancies)
    actual = _read_json(actual_path, relative_path, "candidate", discrepancies)
    if expected is _MISSING or actual is _MISSING:
        return
    _compare_json_values(expected, actual, relative_path, "", discrepancies)


def _read_json(
    path: Path,
    relative_path: str,
    side: str,
    discrepancies: list[Discrepancy],
) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="json_read_error",
                location={"side": side},
                expected="valid UTF-8 JSON" if side == "locked" else _MISSING,
                actual={"error_type": type(error).__name__, "message": str(error)},
                message=f"The {side} JSON could not be parsed.",
            )
        )
        return _MISSING


def _compare_json_values(
    expected: object,
    actual: object,
    relative_path: str,
    pointer: str,
    discrepancies: list[Discrepancy],
) -> None:
    location = {"json_pointer": pointer}
    if type(expected) is not type(actual):
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="json_type_mismatch",
                location=location,
                expected={"type": _json_type_name(expected), "value": expected},
                actual={"type": _json_type_name(actual), "value": actual},
                message="JSON types differ; booleans, integers, and floats are distinct.",
            )
        )
        return

    if isinstance(expected, dict):
        # JSON object keys are strings.  Sorting is only for deterministic error
        # order; it does not alter or normalize list/record order.
        expected_keys = set(expected)
        actual_mapping = actual
        assert isinstance(actual_mapping, dict)
        actual_keys = set(actual_mapping)
        for key in sorted(expected_keys - actual_keys):
            child_pointer = _json_pointer_child(pointer, key)
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="json_key_missing",
                    location={"json_pointer": child_pointer},
                    expected=expected[key],
                    actual=_MISSING,
                    message="A locked JSON object key is absent.",
                )
            )
        for key in sorted(actual_keys - expected_keys):
            child_pointer = _json_pointer_child(pointer, key)
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="json_key_unexpected",
                    location={"json_pointer": child_pointer},
                    expected=_MISSING,
                    actual=actual_mapping[key],
                    message="The candidate JSON object has an additional key.",
                )
            )
        for key in sorted(expected_keys & actual_keys):
            _compare_json_values(
                expected[key],
                actual_mapping[key],
                relative_path,
                _json_pointer_child(pointer, key),
                discrepancies,
            )
        return

    if isinstance(expected, list):
        actual_sequence = actual
        assert isinstance(actual_sequence, list)
        if len(expected) != len(actual_sequence):
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="json_list_length_mismatch",
                    location=location,
                    expected=len(expected),
                    actual=len(actual_sequence),
                    message="JSON list lengths differ; list order is never normalized.",
                )
            )
        for index in range(min(len(expected), len(actual_sequence))):
            _compare_json_values(
                expected[index],
                actual_sequence[index],
                relative_path,
                _json_pointer_child(pointer, str(index)),
                discrepancies,
            )
        for index in range(len(actual_sequence), len(expected)):
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="json_list_item_missing",
                    location={"json_pointer": _json_pointer_child(pointer, str(index))},
                    expected=expected[index],
                    actual=_MISSING,
                    message="A locked ordered JSON list item is absent.",
                )
            )
        for index in range(len(expected), len(actual_sequence)):
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="json_list_item_unexpected",
                    location={"json_pointer": _json_pointer_child(pointer, str(index))},
                    expected=_MISSING,
                    actual=actual_sequence[index],
                    message="The candidate JSON list has an additional ordered item.",
                )
            )
        return

    if isinstance(expected, float):
        assert isinstance(actual, float)
        expected_bits = _float_bits(expected)
        actual_bits = _float_bits(actual)
        if expected_bits != actual_bits:
            discrepancies.append(
                Discrepancy(
                    relative_path=relative_path,
                    kind="json_float_bits_mismatch",
                    location=location,
                    expected=expected,
                    actual=actual,
                    message="IEEE-754 binary64 values differ, including the sign of zero.",
                )
            )
        return

    if expected != actual:
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="json_value_mismatch",
                location=location,
                expected=expected,
                actual=actual,
                message="JSON scalar values differ exactly.",
            )
        )


def _json_pointer_child(pointer: str, key: str) -> str:
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{pointer}/{escaped}"


def _json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _float_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _compare_png_pixels(
    expected_path: Path,
    actual_path: Path,
    relative_path: str,
    discrepancies: list[Discrepancy],
) -> None:
    try:
        if _same_bytes(expected_path, actual_path):
            return
    except OSError as error:
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="file_read_error",
                location={"scope": "PNG file pair"},
                expected="two readable PNG files",
                actual={"error_type": type(error).__name__, "message": str(error)},
                message="At least one PNG could not be read before pixel comparison.",
            )
        )
        return

    try:
        from PIL import Image
    except ImportError as error:
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="png_decoder_unavailable",
                expected="Pillow available for decoded pixel comparison",
                actual={"error_type": type(error).__name__, "message": str(error)},
                message="PNG bytes differ and decoded pixels could not be checked.",
            )
        )
        return

    try:
        with Image.open(expected_path) as expected_image, Image.open(actual_path) as actual_image:
            expected_frames = int(getattr(expected_image, "n_frames", 1))
            actual_frames = int(getattr(actual_image, "n_frames", 1))
            if expected_frames != actual_frames:
                discrepancies.append(
                    Discrepancy(
                        relative_path=relative_path,
                        kind="png_frame_count_mismatch",
                        location={"scope": "image"},
                        expected=expected_frames,
                        actual=actual_frames,
                        message="Decoded PNG frame counts differ.",
                    )
                )

            for frame_index in range(min(expected_frames, actual_frames)):
                expected_image.seek(frame_index)
                actual_image.seek(frame_index)
                if expected_image.size != actual_image.size:
                    discrepancies.append(
                        Discrepancy(
                            relative_path=relative_path,
                            kind="png_size_mismatch",
                            location={"frame": frame_index},
                            expected=list(expected_image.size),
                            actual=list(actual_image.size),
                            message="Decoded PNG dimensions differ.",
                        )
                    )
                    continue

                expected_rgba = expected_image.convert("RGBA").tobytes()
                actual_rgba = actual_image.convert("RGBA").tobytes()
                if expected_rgba == actual_rgba:
                    continue
                first_offset: int | None = None
                differing_channel_count = 0
                for offset, (left, right) in enumerate(zip(expected_rgba, actual_rgba)):
                    if left == right:
                        continue
                    if first_offset is None:
                        first_offset = offset
                    differing_channel_count += 1
                assert first_offset is not None
                pixel_index, channel_index = divmod(first_offset, 4)
                width = expected_image.size[0]
                y, x = divmod(pixel_index, width)
                channel = ("R", "G", "B", "A")[channel_index]
                discrepancies.append(
                    Discrepancy(
                        relative_path=relative_path,
                        kind="png_pixel_mismatch",
                        location={
                            "frame": frame_index,
                            "x": x,
                            "y": y,
                            "channel": channel,
                        },
                        expected=expected_rgba[first_offset],
                        actual=actual_rgba[first_offset],
                        message=(
                            "Decoded RGBA pixels differ; location is the first differing "
                            f"channel of {differing_channel_count} differing channels."
                        ),
                    )
                )
    except (OSError, ValueError) as error:
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="png_decode_error",
                location={"scope": "image pair"},
                expected="two decodable PNG images",
                actual={"error_type": type(error).__name__, "message": str(error)},
                message="At least one PNG could not be decoded for pixel comparison.",
            )
        )


def _compare_bytes(
    expected_path: Path,
    actual_path: Path,
    relative_path: str,
    discrepancies: list[Discrepancy],
) -> None:
    try:
        if _same_bytes(expected_path, actual_path):
            return
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="file_bytes_mismatch",
                location={"scope": "file"},
                expected=_file_fingerprint(expected_path),
                actual=_file_fingerprint(actual_path),
                message="Opaque file bytes differ exactly.",
            )
        )
    except OSError as error:
        discrepancies.append(
            Discrepancy(
                relative_path=relative_path,
                kind="file_read_error",
                location={"scope": "file pair"},
                expected="two readable files",
                actual={"error_type": type(error).__name__, "message": str(error)},
                message="At least one file could not be read for byte comparison.",
            )
        )


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _json_safe(value: Any) -> object:
    if value is _MISSING:
        return {"state": "missing"}
    if isinstance(value, float):
        return {
            "type": "float",
            "repr": repr(value),
            "ieee754_hex": _float_bits(value),
            "finite": math.isfinite(value),
        }
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)}


if __name__ == "__main__":
    raise SystemExit(main())
