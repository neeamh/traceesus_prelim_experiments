#!/usr/bin/env python3
"""Single entry point for all TRACE-ESUS experiments.

Examples
--------
python run.py list
python run.py run endotype_discovery
python run.py run transportability --repeats 10 --workers 4 --out /tmp/smoke
python run.py run all
python run.py figures endotype_discovery
python run.py verify --against ./outputs_locked
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceesus.core.verification import compare_output_trees


EXPERIMENTS = {
    "model_comparison": {
        "config": "configs/model_comparison.py",
        "output": "outputs_associative_vs_scm",
        "description": "Supervised associative classifiers versus a supervised fitted SCM.",
    },
    "endotype_discovery": {
        "config": "configs/endotype_discovery.py",
        "output": "outputs_latent_endotyping",
        "description": "Unsupervised latent endotype recovery plus the K=1 null control.",
    },
    "transportability": {
        "config": "configs/transportability.py",
        "output": "outputs_transportability",
        "description": "Unlabeled cross-hospital transport, shifts, and ablations.",
    },
    "counterfactual": {
        "config": "configs/counterfactual.py",
        "output": "outputs",
        "description": "Known-SCM kidney-blind, posterior, and counterfactual query comparison.",
    },
    "redundancy_sweep": {
        "config": "configs/endotype_discovery.py",
        "output": "outputs_redundancy_sweep",
        "description": (
            "HF path on PTFV1 swept at fixed strong renal distortion; four "
            "nuisance-profile subgroups. Own seed root; touches no locked output."
        ),
    },
}


def _endotype_factory(
    *, repeats: int | None, workers: int | None, output: Path
) -> Any:
    from configs.endotype_discovery import CONFIG
    from traceesus.experiments.endotype_discovery import EndotypeDiscoveryExperiment

    config = CONFIG
    if repeats is not None:
        config = replace(config, repeats_per_level=repeats, null_repeats=repeats)
    if workers is not None:
        config = replace(config, workers=workers)
    return EndotypeDiscoveryExperiment(config, output)


def _model_comparison_factory(
    *, repeats: int | None, workers: int | None, output: Path
) -> Any:
    from configs.model_comparison import CONFIG
    from traceesus.experiments.model_comparison import ModelComparisonExperiment

    if workers not in (None, 1):
        raise ValueError(
            "model_comparison has no legacy parallel-worker setting; use --workers 1."
        )
    config = replace(CONFIG, repeats_per_level=repeats) if repeats is not None else CONFIG
    return ModelComparisonExperiment(config, output)


def _transportability_factory(
    *, repeats: int | None, workers: int | None, output: Path
) -> Any:
    from configs.transportability import CONFIG
    from traceesus.experiments.transportability import TransportabilityExperiment

    config = CONFIG
    if repeats is not None:
        config = replace(config, repeats=repeats)
    if workers is not None:
        config = replace(config, workers=workers)
    return TransportabilityExperiment(config, output)


def _counterfactual_factory(
    *, repeats: int | None, workers: int | None, output: Path
) -> Any:
    from configs.counterfactual import CONFIG
    from traceesus.experiments.counterfactual import CounterfactualExperiment

    if workers not in (None, 1):
        raise ValueError(
            "counterfactual has no legacy parallel-worker setting; use --workers 1."
        )
    config = CONFIG
    if repeats is not None:
        config = replace(
            config,
            repeats_per_level=repeats,
            null_repeats=repeats,
        )
    return CounterfactualExperiment(config, output)


def _redundancy_sweep_factory(
    *, repeats: int | None, workers: int | None, output: Path
) -> Any:
    from configs.endotype_discovery import CONFIG
    from traceesus.experiments.endotype_discovery import RedundancySweepExperiment

    config = CONFIG
    if repeats is not None:
        config = replace(config, repeats_per_level=repeats, null_repeats=repeats)
    if workers is not None:
        config = replace(config, workers=workers)
    return RedundancySweepExperiment(config, output)


FACTORIES: dict[str, Callable[..., Any]] = {
    "model_comparison": _model_comparison_factory,
    "endotype_discovery": _endotype_factory,
    "transportability": _transportability_factory,
    "counterfactual": _counterfactual_factory,
    "redundancy_sweep": _redundancy_sweep_factory,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List experiments, config files, and outputs.")

    run_parser = commands.add_parser("run", help="Run one experiment or all experiments.")
    run_parser.add_argument("experiment", choices=(*EXPERIMENTS, "all"))
    run_parser.add_argument("--repeats", type=int, help="Smoke-run repeat override.")
    run_parser.add_argument("--workers", type=int, help="Ordered local worker processes.")
    run_parser.add_argument(
        "--out",
        type=Path,
        help="Output directory for one experiment, or parent directory for all.",
    )

    figures_parser = commands.add_parser(
        "figures", help="Regenerate figures from existing CSV/JSON outputs."
    )
    figures_parser.add_argument("experiment", choices=tuple(EXPERIMENTS))
    figures_parser.add_argument("--out", type=Path, help="Existing experiment output directory.")

    verify_parser = commands.add_parser(
        "verify", help="Compare compatibility outputs against a locked baseline."
    )
    verify_parser.add_argument("--against", type=Path, required=True)
    verify_parser.add_argument(
        "--candidate",
        type=Path,
        default=Path.cwd(),
        help="Parent containing candidate output directories (default: current directory).",
    )
    verify_parser.add_argument(
        "--report",
        type=Path,
        default=Path("verification_discrepancies.json"),
    )
    return parser


def _list_experiments() -> int:
    print("TRACE-ESUS experiments\n")
    for name, details in EXPERIMENTS.items():
        print(f"{name}")
        print(f"  tests:  {details['description']}")
        print(f"  config: {details['config']}")
        print(f"  output: {details['output']}")
    return 0


def _output_for(name: str, override: Path | None, *, all_run: bool) -> Path:
    default_name = Path(str(EXPERIMENTS[name]["output"]))
    if override is None:
        return default_name
    return override / default_name if all_run else override


def _restore_config_like(template: Any, serialized: Any) -> Any:
    """Rehydrate a manifest config without hard-coding four dataclass schemas.

    Figure-only regeneration must use the configuration that produced the
    tables, including smoke-run overrides.  The already-constructed default is
    used only as a type template; tuple order and nested dataclass types are
    restored recursively before their construction-time validation runs.
    """

    if is_dataclass(template) and not isinstance(template, type):
        if not isinstance(serialized, dict):
            raise ValueError("A serialized dataclass config must be a JSON object.")
        values = {
            field.name: _restore_config_like(
                getattr(template, field.name),
                serialized[field.name],
            )
            for field in fields(template)
            if field.name in serialized
        }
        return type(template)(**values)
    if isinstance(template, tuple):
        if not isinstance(serialized, list):
            raise ValueError("A serialized tuple config field must be a JSON array.")
        if not template:
            return tuple(serialized)
        return tuple(
            _restore_config_like(template[min(index, len(template) - 1)], value)
            for index, value in enumerate(serialized)
        )
    return serialized


def _run_one(
    name: str,
    *,
    repeats: int | None,
    workers: int | None,
    output: Path,
) -> int:
    factory = FACTORIES.get(name)
    if factory is None:
        raise RuntimeError(f"Experiment {name!r} has not yet been registered.")
    experiment = factory(repeats=repeats, workers=workers, output=output)
    results = experiment.execute()
    checks = results.get("validation_checks") if isinstance(results, dict) else None
    if isinstance(checks, dict) and checks.get("all_required_checks_pass") is False:
        if repeats is None:
            raise RuntimeError(f"{name}: one or more required validation checks failed.")
        print(
            f"warning: {name}: smoke-run validation thresholds did not all pass; "
            "artifacts were retained for runtime and diagnostic inspection.",
            file=sys.stderr,
        )
    print(f"{name}: artifacts written to {output.resolve()}")
    return 0


def _run(arguments: argparse.Namespace) -> int:
    names = tuple(EXPERIMENTS) if arguments.experiment == "all" else (arguments.experiment,)
    for name in names:
        _run_one(
            name,
            repeats=arguments.repeats,
            workers=arguments.workers,
            output=_output_for(name, arguments.out, all_run=arguments.experiment == "all"),
        )
    return 0


def _figures(arguments: argparse.Namespace) -> int:
    from traceesus.core.io import write_json, write_manifest

    name = arguments.experiment
    output = _output_for(name, arguments.out, all_run=False)
    factory = FACTORIES.get(name)
    if factory is None:
        raise RuntimeError(f"Experiment {name!r} has not yet been registered.")
    experiment = factory(repeats=None, workers=None, output=output)
    existing_manifest: dict[str, Any] | None = None
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("config"), dict):
            raise ValueError(f"{manifest_path}: manifest config is invalid.")
        existing_manifest = loaded
        restored_config = _restore_config_like(experiment.config, loaded["config"])
        experiment = type(experiment)(restored_config, output)
    experiment.configure()
    started = perf_counter()
    experiment.figures_only()
    figure_runtime = perf_counter() - started
    config = experiment.config
    if hasattr(config, "master_seed"):
        master_seed = int(config.master_seed)
    elif hasattr(config, "seed"):
        master_seed = int(config.seed)
    else:
        raise RuntimeError(f"{name}: config exposes no reproducibility seed.")
    recorded_runtime = figure_runtime
    if existing_manifest is not None:
        previous_runtime = existing_manifest.get("wall_clock_runtime_seconds")
        if (
            not isinstance(previous_runtime, bool)
            and isinstance(previous_runtime, (int, float))
            and math.isfinite(previous_runtime)
            and previous_runtime >= 0.0
        ):
            recorded_runtime = float(previous_runtime)
    manifest = write_manifest(
        output,
        experiment=name,
        config=config,
        master_seed=master_seed,
        wall_clock_runtime_seconds=recorded_runtime,
    )
    manifest["last_operation"] = "figures_only"
    manifest["last_operation_runtime_seconds"] = figure_runtime
    write_json(manifest_path, manifest)
    print(f"{name}: figures regenerated in {output.resolve()}")
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    all_discrepancies: list[dict[str, object]] = []
    compared = 0
    for name, details in EXPERIMENTS.items():
        relative = Path(str(details["output"]))
        locked = arguments.against / relative
        candidate = arguments.candidate / relative
        if not locked.exists():
            # Experiments added after the freeze (e.g. the redundancy sweep)
            # have no locked baseline; verification covers cited outputs only.
            print(f"note: {name}: no locked baseline at {locked}; skipped.")
            continue
        report = compare_output_trees(
            locked,
            candidate,
            legacy_data_only=True,
        )
        compared += len(report.compared_files)
        all_discrepancies.extend(
            {"experiment": name, **item.to_dict()} for item in report.discrepancies
        )
    payload = {
        "status": "pass" if not all_discrepancies else "fail",
        "compared_file_count": compared,
        "discrepancy_count": len(all_discrepancies),
        "discrepancies": all_discrepancies,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if not all_discrepancies else 1


def main(argv: list[str] | None = None) -> int:
    """Dispatch the root CLI and return a strict process status."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "list":
            return _list_experiments()
        if arguments.command == "run":
            return _run(arguments)
        if arguments.command == "figures":
            return _figures(arguments)
        if arguments.command == "verify":
            return _verify(arguments)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
