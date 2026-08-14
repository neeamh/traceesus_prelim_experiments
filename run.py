#!/usr/bin/env python3
"""Single entry point for all TRACE-ESUS experiments.

Examples
--------
python run.py list
python run.py run endotype_discovery
python run.py run transportability --repeats 10 --workers 4 --out /tmp/smoke
python run.py run all
python run.py figures endotype_discovery  # points to the CSV-only notebook
python run.py verify --against ./outputs_locked
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceesus.core.verification import compare_output_trees


EXPERIMENTS = {
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


FACTORIES: dict[str, Callable[..., Any]] = {
    "endotype_discovery": _endotype_factory,
    "transportability": _transportability_factory,
    "counterfactual": _counterfactual_factory,
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
        "figures", help="Show the path to the CSV-only figure notebook."
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
    """Point presentation requests to the committed CSV-only notebook."""

    notebook = PROJECT_ROOT / "notebooks" / "figures.ipynb"
    print(
        f"{arguments.experiment}: package figure generation was removed; "
        f"run {notebook}."
    )
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    all_discrepancies: list[dict[str, object]] = []
    compared = 0
    for name, details in EXPERIMENTS.items():
        relative = Path(str(details["output"]))
        locked = arguments.against / relative
        candidate = arguments.candidate / relative
        if not locked.exists():
            # Extensions added after the freeze have no locked baseline;
            # verification covers cited experiment outputs only.
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
