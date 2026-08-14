# TRACE-ESUS architecture

TRACE-ESUS separates generated cohorts, fitted models, experiment orchestration, and presentation so the origin of every result is visible without tracing procedural kernels.

## Layers

- `configs/` contains immutable, validated scientific and numerical settings. Defaults are the retained experiment designs.
- `traceesus/simulators/` is the only cohort-generation layer. `TwoMechanismSimulator` is a frozen dataclass; site shifts are configuration, not simulator subclasses. It returns observed `Cohort` data separately from simulation truth.
- `traceesus/models/` contains frozen concrete model and fitted-model dataclasses. A model's `fit(cohort, rng, config)` method owns initialization, model-specific priors, stopping, and assembly of fitted parameters. A fitted model's `posterior(cohort)` method is deterministic and accepts no RNG.
- `traceesus/core/em.py` is the single numerical EM implementation. Its one missingness-aware E-step and one missingness-aware M-step serve complete data through an all-observed mask and partially observed data through the same masked operations. Models share these functions directly; no numerical model base class exists.
- `traceesus/experiments/` orchestrates paired seed streams, simulation, fitting, evaluation, and CSV output. It does not own model arithmetic or plotting.
- `traceesus/core/metrics.py`, `stats.py`, `runner.py`, and `io.py` provide evaluation, Monte Carlo summaries, ordered execution, seed ledgers, manifests, and output writing.
- `notebooks/figures.ipynb` is the presentation layer. It reads CSVs only and imports only the small plotting style module from the package.

## Adding an experiment

Add a validated config, compose existing simulator and model objects in one experiment class, register its command-line facade, and write raw-long, summary, contrast, and metadata outputs. Add a new model only when the statistical model is genuinely new; share emission arithmetic through `core/em.py`, not inheritance or a second EM loop. Add new metrics to the data dictionary when they first appear.

## RNG rules

1. Randomness occurs only inside a method or fitting function that accepts an explicit `rng` argument.
2. Every random draw is documented in occurrence order, and existing draw order and child-stream assignment are preserved unless an explicitly approved migration says otherwise.
3. Constructors, properties, caches, and `FittedModel.posterior(data)` never draw randomness.

## Provenance

`outputs_locked/` is immutable pre-unification v1 provenance. `outputs_locked_v2/` is the exact unified-EM baseline. Tests compare current runs exactly to v2 and retain a non-failing v1-to-v2 audit; neither tree is regenerated inside tests.
