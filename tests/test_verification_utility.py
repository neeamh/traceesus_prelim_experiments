"""Focused contract tests for exact output verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traceesus.core.io as io_module
import traceesus.core.verification as verification_module
from traceesus.core.io import write_manifest
from traceesus.core.verification import compare_output_trees, main


def _write(path: Path, contents: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    else:
        path.write_text(contents, encoding="utf-8", newline="")


def _manifest(
    output_directory: Path,
    *,
    config: dict[str, object] | None = None,
    seed: int = 17,
    experiment: str = "test_experiment",
) -> dict[str, object]:
    return write_manifest(
        output_directory,
        experiment=experiment,
        config=config or {"master_seed": seed, "repeats": 2},
        master_seed=seed,
        wall_clock_runtime_seconds=0.25,
    )


def test_identical_recursive_tree_allows_only_additive_manifests(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    _write(locked / "experiment" / "raw.csv", "repeat,value\n0,1.00\n")
    _write(candidate / "experiment" / "raw.csv", "repeat,value\n0,1.00\n")
    config = {"master_seed": 17, "repeats": 2}
    _write(
        locked / "experiment" / "metadata.json",
        json.dumps({"config": config, "b": [True, 2], "a": 1}),
    )
    _write(
        candidate / "experiment" / "metadata.json",
        json.dumps({"a": 1, "b": [True, 2], "config": config}),
    )
    _write(locked / "experiment" / "artifact.bin", b"\x00\x01")
    _write(candidate / "experiment" / "artifact.bin", b"\x00\x01")
    _manifest(candidate / "experiment", config=config)

    report = compare_output_trees(locked, candidate)

    assert report.passed
    assert report.exit_code == 0
    assert report.allowed_additions == ("experiment/manifest.json",)
    assert report.to_dict()["discrepancy_count"] == 0


def test_manifest_git_provenance_tracks_code_when_output_is_outside_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arbitrary output location must not replace the producing code checkout."""

    locked = tmp_path / "locked"
    candidate = tmp_path / "external-output"
    assert io_module.CODE_CHECKOUT_ROOT not in candidate.parents
    config = {"master_seed": 17, "repeats": 2}
    metadata = json.dumps({"config": config})
    _write(locked / "metadata.json", metadata)
    _write(candidate / "metadata.json", metadata)
    _write(locked / "raw.csv", "x\n1\n")
    _write(candidate / "raw.csv", "x\n1\n")

    write_probes: list[Path] = []
    verify_probes: list[Path] = []

    def fake_write_git_state(start: Path) -> tuple[str, str]:
        write_probes.append(start)
        return "0123456789abcdef", "available"

    def fake_live_git_state(start: Path) -> tuple[str, str]:
        verify_probes.append(start)
        return "0123456789abcdef", "available"

    monkeypatch.setattr(io_module, "_git_state", fake_write_git_state)
    monkeypatch.setattr(
        verification_module,
        "_live_git_state",
        fake_live_git_state,
    )
    manifest = _manifest(candidate, config=config)
    report = compare_output_trees(locked, candidate, legacy_data_only=True)

    assert manifest["git_commit"] == "0123456789abcdef"
    assert write_probes == [io_module.CODE_CHECKOUT_ROOT]
    assert verify_probes == [verification_module.CODE_CHECKOUT_ROOT]
    assert report.passed


def test_legacy_data_only_is_exact_for_csv_json_and_ignores_figures(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    _write(locked / "raw.csv", "x\n1.00\n")
    _write(candidate / "raw.csv", "x\n1.0\n")
    _write(locked / "metadata.json", '{"config":{"seed":3},"value":1}\n')
    _write(candidate / "metadata.json", '{"value":2,"config":{"seed":3}}\n')
    _write(locked / "figure.pdf", b"locked PDF bytes")
    _write(candidate / "new-render.png", b"candidate-only rendering")

    report = compare_output_trees(locked, candidate, legacy_data_only=True)

    assert {(item.relative_path, item.kind) for item in report.discrepancies} == {
        ("raw.csv", "csv_cell_mismatch"),
        ("metadata.json", "json_value_mismatch"),
    }
    assert set(report.compared_files) == {"raw.csv", "metadata.json"}
    assert set(report.ignored_files) == {"figure.pdf", "new-render.png"}


def test_suffix_selection_normalizes_missing_dot(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    _write(locked / "raw.csv", "x\n1\n")
    _write(candidate / "raw.csv", "x\n1\n")
    _write(locked / "metadata.json", '{"x":1}\n')
    _write(candidate / "metadata.json", '{"x":2}\n')

    report = compare_output_trees(locked, candidate, include_suffixes={"csv"})

    assert report.passed
    assert report.compared_files == ("raw.csv",)
    assert report.ignored_files == ("metadata.json",)


def test_pdf_is_ignored_by_default_but_byte_mode_is_strict(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    _write(locked / "metadata.json", '{"x":1}\n')
    _write(candidate / "metadata.json", '{"x":1}\n')
    _write(locked / "figure.pdf", b"same rendered figure, locked metadata")
    _write(candidate / "figure.pdf", b"same rendered figure, candidate metadata")

    ignored = compare_output_trees(locked, candidate)
    strict = compare_output_trees(locked, candidate, pdf_mode="bytes")

    assert ignored.passed
    assert ignored.ignored_files == ("figure.pdf",)
    assert [item.kind for item in strict.discrepancies] == ["file_bytes_mismatch"]


def test_missing_locked_file_and_nonmanifest_addition_fail(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    _write(locked / "required.csv", "x\n1\n")
    _write(candidate / "extra.json", "{}\n")

    report = compare_output_trees(locked, candidate)
    kinds = {(item.relative_path, item.kind) for item in report.discrepancies}

    assert not report.passed
    assert report.exit_code != 0
    assert ("required.csv", "required_file_missing") in kinds
    assert ("extra.json", "unexpected_file") in kinds


def test_empty_locked_tree_cannot_pass_vacuously(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    locked.mkdir()
    candidate.mkdir()

    report = compare_output_trees(locked, candidate)

    assert {item.kind for item in report.discrepancies} == {"locked_tree_empty"}
    assert report.exit_code == 1


def test_csv_compares_raw_strings_in_order_with_exact_coordinates(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    _write(locked / "raw.csv", "repeat,value\n0,-0.0\n1,1.00\n")
    _write(candidate / "raw.csv", "repeat,value\n0,0.0\n1,1.0\n")

    report = compare_output_trees(locked, candidate)
    cell_differences = [
        item for item in report.discrepancies if item.kind == "csv_cell_mismatch"
    ]

    assert len(cell_differences) == 2
    assert cell_differences[0].location == {
        "row": 2,
        "column": 2,
        "column_name": "value",
    }
    assert cell_differences[0].expected == "-0.0"
    assert cell_differences[0].actual == "0.0"
    assert cell_differences[1].location["row"] == 3
    assert cell_differences[1].expected == "1.00"
    assert cell_differences[1].actual == "1.0"


def test_json_checks_types_order_key_sets_and_signed_zero_bits(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    _write(
        locked / "values.json",
        json.dumps(
            {
                "zero": -0.0,
                "integer": 1,
                "items": [True, {"x": "a"}],
                "ordered": ["first", "second"],
            }
        ),
    )
    _write(
        candidate / "values.json",
        json.dumps(
            {
                "zero": 0.0,
                "integer": 1.0,
                "items": [1, {"y": "a"}],
                "ordered": ["second", "first"],
            }
        ),
    )

    report = compare_output_trees(locked, candidate)
    by_pointer = {
        item.location.get("json_pointer"): item.kind for item in report.discrepancies
    }

    assert by_pointer["/zero"] == "json_float_bits_mismatch"
    assert by_pointer["/integer"] == "json_type_mismatch"
    assert by_pointer["/items/0"] == "json_type_mismatch"
    assert by_pointer["/items/1/x"] == "json_key_missing"
    assert by_pointer["/items/1/y"] == "json_key_unexpected"
    assert by_pointer["/ordered/0"] == "json_value_mismatch"
    assert by_pointer["/ordered/1"] == "json_value_mismatch"
    zero = next(item for item in report.discrepancies if item.location == {"json_pointer": "/zero"})
    serialized = zero.to_dict()
    assert serialized["expected"]["ieee754_hex"] == "8000000000000000"
    assert serialized["actual"]["ieee754_hex"] == "0000000000000000"


def test_png_pixel_mode_ignores_metadata_but_byte_mode_does_not(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    png_module = pytest.importorskip("PIL.PngImagePlugin")
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    locked.mkdir()
    candidate.mkdir()

    pixels = image_module.new("RGBA", (2, 1), (10, 20, 30, 255))
    locked_metadata = png_module.PngInfo()
    locked_metadata.add_text("marker", "locked")
    candidate_metadata = png_module.PngInfo()
    candidate_metadata.add_text("marker", "candidate")
    pixels.save(locked / "figure.png", pnginfo=locked_metadata)
    pixels.save(candidate / "figure.png", pnginfo=candidate_metadata)
    assert (locked / "figure.png").read_bytes() != (candidate / "figure.png").read_bytes()

    pixel_report = compare_output_trees(locked, candidate, compare_png_pixels=True)
    byte_report = compare_output_trees(locked, candidate, compare_png_pixels=False)

    assert pixel_report.passed
    assert [item.kind for item in byte_report.discrepancies] == ["file_bytes_mismatch"]


def test_manifest_rejects_false_provenance_and_stale_checksums(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    config = {"master_seed": 17, "repeats": 2}
    metadata = json.dumps({"experiment_config": config})
    _write(locked / "metadata.json", metadata)
    _write(candidate / "metadata.json", metadata)
    _write(locked / "raw.csv", "x\n1\n")
    _write(candidate / "raw.csv", "x\n1\n")
    manifest = _manifest(candidate, config=config)
    manifest["config"] = {"master_seed": 99, "repeats": 3}
    manifest["master_seed"] = 99
    manifest["package_version"] = "wrong"
    manifest["package_versions"]["numpy"] = "wrong"
    manifest["git_commit"] = "wrong"
    manifest["git_state"] = "wrong"
    manifest["wall_clock_runtime_seconds"] = float("inf")
    manifest["output_file_sha256"]["raw.csv"] = "0" * 64
    _write(candidate / "manifest.json", json.dumps(manifest))

    report = compare_output_trees(locked, candidate, legacy_data_only=True)
    pointers = {
        item.location.get("json_pointer") for item in report.discrepancies
    }
    kinds = {item.kind for item in report.discrepancies}

    assert "/config/master_seed" in pointers
    assert "/config/repeats" in pointers
    assert "/master_seed" in pointers
    assert "/package_version" in pointers
    assert "/package_versions/numpy" in pointers
    assert "/git_commit" in pointers
    assert "/git_state" in pointers
    assert "manifest_runtime_invalid" in kinds
    assert "manifest_checksum_mismatch" in kinds


def test_manifest_requires_complete_schema_and_independent_config(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    _write(locked / "raw.csv", "x\n1\n")
    _write(candidate / "raw.csv", "x\n1\n")
    _write(candidate / "manifest.json", "{}\n")

    report = compare_output_trees(locked, candidate, legacy_data_only=True)
    kinds = [item.kind for item in report.discrepancies]

    assert kinds.count("manifest_key_missing") == 9
    assert "manifest_config_reference_missing" in kinds
    assert "manifest_seed_reference_missing" in kinds


def test_nested_manifests_use_nearest_locked_config_and_scoped_inventory(
    tmp_path: Path,
) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    config = {"master_seed": 41, "repeats": 2}
    metadata = json.dumps({"experiment_config": config})
    _write(locked / "metadata.json", metadata)
    _write(candidate / "metadata.json", metadata)
    _write(locked / "main.csv", "x\n1\n")
    _write(candidate / "main.csv", "x\n1\n")
    _write(locked / "ablations" / "raw.csv", "x\n2\n")
    _write(candidate / "ablations" / "raw.csv", "x\n2\n")

    # The child has no local metadata; the locked parent config is authoritative.
    _manifest(candidate / "ablations", config=config, seed=41, experiment="ablation")
    # Parent checksums cover both main.csv and ablations/raw.csv, but no manifest.
    parent = _manifest(candidate, config=config, seed=41, experiment="transportability")

    report = compare_output_trees(locked, candidate, legacy_data_only=True)

    assert report.passed
    assert report.allowed_additions == ("ablations/manifest.json", "manifest.json")
    assert set(parent["output_file_sha256"]) == {
        "metadata.json",
        "main.csv",
        "ablations/raw.csv",
    }


def test_parent_manifest_detects_output_added_after_it(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    config = {"master_seed": 41}
    metadata = json.dumps({"config": config})
    _write(locked / "metadata.json", metadata)
    _write(candidate / "metadata.json", metadata)
    _manifest(candidate, config=config, seed=41)
    _write(locked / "ablations" / "late.csv", "x\n1\n")
    _write(candidate / "ablations" / "late.csv", "x\n1\n")

    report = compare_output_trees(locked, candidate, legacy_data_only=True)

    missing = [
        item
        for item in report.discrepancies
        if item.kind == "manifest_checksum_entry_missing"
    ]
    assert len(missing) == 1
    assert missing[0].location["artifact"] == "ablations/late.csv"


def test_manifest_checksum_map_must_not_name_missing_files(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    config = {"seed": 5}
    metadata = json.dumps({"config": config})
    _write(locked / "metadata.json", metadata)
    _write(candidate / "metadata.json", metadata)
    manifest = _manifest(candidate, config=config, seed=5)
    manifest["output_file_sha256"]["ghost.csv"] = "0" * 64
    _write(candidate / "manifest.json", json.dumps(manifest))

    report = compare_output_trees(locked, candidate, legacy_data_only=True)

    assert "manifest_checksum_entry_unexpected" in {
        item.kind for item in report.discrepancies
    }


def test_cli_emits_machine_readable_failure_and_nonzero_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    report_path = tmp_path / "verification.json"
    _write(locked / "raw.csv", "x\n1\n")
    _write(candidate / "raw.csv", "x\n2\n")

    exit_code = main([str(locked), str(candidate), "--report", str(report_path)])
    stdout_report = json.loads(capsys.readouterr().out)
    written_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code != 0
    assert stdout_report["status"] == "fail"
    assert stdout_report["discrepancy_count"] == 1
    assert stdout_report["discrepancies"][0]["location"] == {
        "column": 1,
        "column_name": "x",
        "row": 2,
    }
    assert written_report == stdout_report


def test_cli_legacy_data_only_ignores_figure_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    locked = tmp_path / "locked"
    candidate = tmp_path / "candidate"
    _write(locked / "raw.csv", "x\n1\n")
    _write(candidate / "raw.csv", "x\n1\n")
    _write(locked / "figure.pdf", b"locked")

    exit_code = main([str(locked), str(candidate), "--legacy-data-only"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["status"] == "pass"
    assert report["compared_files"] == ["raw.csv"]
    assert report["ignored_files"] == ["figure.pdf"]
