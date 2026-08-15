# TRACE-ESUS architecture

TRACE-ESUS separates generated cohorts, fitted models, experiment orchestration, and presentation so the origin of every result is visible without tracing procedural kernels.

## Layers

- `configs/` contains immutable, validated scientific and numerical settings. Defaults are the retained experiment designs.
- `traceesus/simulators/` is the only cohort-generation layer. `TwoMechanismSimulator` is a frozen dataclass; site shifts are configuration, not simulator subclasses. It returns observed `Cohort` data separately from simulation truth.
- `traceesus/models/` contains frozen concrete model and fitted-model dataclasses. A model's `fit(cohort, rng, config)` method owns initialization, model-specific structure, and assembly of fitted parameters. A fitted model's `posterior(cohort)` method is deterministic and accepts no RNG.
- `traceesus/registry.py` is the single design table. It defines the exact four-row v3 model set, the default R1--R6 ladder, and every locked or exploratory experiment's config, cohort sizes, repeats, seed root, evaluation cohort, status, and output directory. `design_table()` renders those declarations directly.
- `traceesus/core/em.py` is the single numerical EM implementation. Its conditional E-step and M-step accept an `n x q` nuisance design and `q x p` Boolean path mask, so renal-only and two-nuisance models differ only by those arrays. Missingness uses the same masked operations. The unconditional associative LCM retains a separate factorization but calls the same Gaussian primitives; no numerical model base class exists.
- `traceesus/experiments/` orchestrates paired seed streams, simulation, fitting, evaluation, and CSV output for the two active experiments: endotype discovery and transportability. The retired known-parameter counterfactual study is under `archive/counterfactual/`.
- `traceesus/core/metrics.py`, `stats.py`, `seeds.py`, `runner.py`, and `io.py` provide evaluation, the single 0.005 MCSE target, collision-checked seed roots, ordered execution, seed ledgers, manifests, and output writing.
- `notebooks/figures.ipynb` is the presentation layer. It reads CSVs only and imports only the small plotting style module from the package.

## Adding an experiment

Add a validated config and one immutable `ExperimentDesign` row, compose existing simulator and model objects in one experiment class, register its command-line facade, and write raw-long, summary, contrast, and metadata outputs. Add a new model only when the statistical model is genuinely new; append it to `FULL_LADDER`, share emission arithmetic through `core/em.py`, and do not alter `LOCKED_MODEL_SET`. Add new metrics to the data dictionary when they first appear.

## RNG rules

1. Randomness occurs only inside a method or fitting function that accepts an explicit `rng` argument.
2. Every random draw is documented in occurrence order, and existing draw order and child-stream assignment are preserved unless an explicitly approved migration says otherwise.
3. Constructors, properties, caches, and `FittedModel.posterior(data)` never draw randomness.

Every active root is a named constant in `core/seeds.py`; importing that module asserts uniqueness. Appending models may extend a `SeedSequence.spawn()` call only after a test proves that all established children are unchanged.

## Provenance

`outputs_locked/` is immutable pre-unification v1 provenance, `outputs_locked_v2/` records the first shared-EM implementation, and `outputs_locked_v3/` records conditional-EM unification. Exact tests target v3 and retain non-failing audits between older versions.

Any approved change to floating-point reduction order creates the next versioned baseline rather than overwriting an existing one. The exact golden test is repointed to that new tree without a tolerance, and a scientific-keyed delta report must state the number and maximum size of changes, every change at reported precision, and every confidence interval whose zero-inclusion status changed.
