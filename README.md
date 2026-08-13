# TRACE-ESUS experiments

TRACE-ESUS is a simulation package for three scientific experiments on causal endotyping in cryptogenic stroke. Endotype discovery and transportability fit without mechanism labels; the preliminary counterfactual experiment queries a known data-generating SCM rather than fitting one.

The package retains the legacy numerical kernels because the existing tables are cited in an NIH proposal and manuscript. Refactoring is not permission to change a seed, numerical default, iteration order, reduction order, or algorithm. Suspected defects and limitations are recorded in [`NOTES.md`](NOTES.md) and deliberately remain unchanged.

## Experiments

### `endotype_discovery`: unlabeled latent recovery

Scientific question: can a biology-constrained latent model recover two mechanisms without labels when kidney dysfunction distorts one biomarker?

The fitted methods are an associative latent-class model, a renal-adjusted associative latent-class model, and a biology-constrained latent SCM. Their `fit()` methods receive observed biomarkers and covariates only. Simulation truth is held in a separate object and is used after fitting for evaluation. Latent labels are oriented with the prespecified electrical-minus-competing anchor contrast; neither truth nor the renal-distorted NT-proBNP-like marker is used for orientation.

The data-generating oracle is a **simulation ceiling**, not a fitted competitor and not a clinically available method. The experiment also includes a K=1 null control. Figure P2's example patient is deliberately selected using truth and method disagreement criteria; it is an illustration, not a random or representative patient.

Legacy output directory: `outputs_latent_endotyping/`

```text
raw_recovery_metrics.csv
recovery_summary.csv
paired_contrasts.csv
fit_diagnostics.csv
parameter_recovery.csv
k1_null_raw.csv
k1_null_summary.csv
example_patient.json
metadata.json
validation_checks.json
figure_P1_latent_recovery.{png,pdf}
figure_P2_example_patient.{png,pdf}
figure_S1_controls.{png,pdf}
```

### `transportability`: unlabeled cross-hospital transport

Scientific question: how do latent associative and causal models transport from three unlabeled source hospitals to a held-out target hospital under controlled distribution shifts?

The methods are pooled associative, target-adjusted associative, frozen causal SCM, modular causal SCM, and a target oracle. Source and target mechanism labels are excluded from fitting and target recalibration; labels are used only for simulation evaluation. Target recalibration can use assay metadata, observed renal status, observed background inflammation, and unlabeled target biomarkers.

The target oracle knows the target data-generating process and is therefore an **evaluation ceiling**, not a deployable method. Assay offset and scale are also assumed known and are inverted exactly for every method. Consequently the assay-only ablation is a calibration negative control and cannot establish robustness to unknown assay shift. The main curve's “No shift” hospital is not the literal identical-source/target design; `exact_no_shift_control/` supplies that separate control.

Legacy output directory: `outputs_transportability/`

```text
raw_transport_metrics.csv
transport_summary.csv
paired_transport_contrasts.csv
transport_degradation.csv
fit_diagnostics.csv
target_calibration_diagnostics.csv
negative_control.json
metadata.json
validation_checks.json
figure_T1_transportability.{png,pdf}
figure_T2_transport_controls.{png,pdf}

ablations/
  raw_ablation_metrics.csv
  ablation_summary.csv
  paired_ablation_contrasts.csv
  ablation_accuracy_changes.csv
  fit_diagnostics.csv
  target_calibration_diagnostics.csv
  negative_control.json
  validation_checks.json
  figure_T3_shift_ablations.{png,pdf}

exact_no_shift_control/
  raw_metrics.csv
  summary.csv
  paired_contrasts.csv
  fit_diagnostics.csv
  target_calibration_diagnostics.csv
  negative_control.json
  metadata.json
  validation_checks.json
```

### `counterfactual`: known-DGP query comparison

Scientific question: in the preliminary symmetric K=2 simulation, how do a kidney-blind posterior, a kidney-aware posterior, and posterior-integrated sufficiency/disablement scoring behave as renal distortion increases?

This experiment does **not** fit an SCM. The kidney-aware posterior and counterfactual scores use the known data-generating parameters, including the true renal effect at each simulation level. They are oracle/known-model queries, not evidence that a fitted causal model has learned those quantities. Sufficiency and disablement are monotone transformations of the correctly specified posterior in this symmetric design, so they cannot outperform that same-SCM posterior in top-rank classification. The K=1 null control likewise removes the known renal contribution before fitting its Gaussian models.

Legacy output directory: `outputs/`

```text
main_simulation_raw_metrics.csv
main_simulation_summary.csv
paired_method_differences.csv
k1_null_raw_results.csv
k1_null_summary.csv
run_metadata.json
figure_P1.png
figure_P1.pdf
```

Every package run adds `manifest.json` without renaming or rewriting the legacy contract. The manifest records the complete config, master seed, package version, Git state, dependency versions, wall-clock runtime, and SHA-256 checksums for generated artifacts.

The former supervised comparison is not part of the executable package. Its source history is
preserved under `archive/supervised/`, and its immutable cited provenance remains under
`outputs_locked/outputs_associative_vs_scm/`.

## Installation

Use the proposal environment exactly. Do not substitute a newer “compatible” scientific stack for a reproduction run.

```bash
# From the project root: the directory containing run.py
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The pinned runtime is:

| Dependency | Version |
|---|---:|
| Python | 3.12.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| SciPy | 1.14.1 |
| Matplotlib | 3.9.2 |

For a headless machine, set `MPLBACKEND=Agg`. A clean environment should also pass `python -m pip check` before a locked run.

## Root command line

`run.py` is the only entry point a new user needs:

```bash
python run.py list
python run.py run endotype_discovery
python run.py run transportability --repeats 10 --workers 4 --out /tmp/smoke
python run.py run all
python run.py figures endotype_discovery
python run.py verify --against ./outputs_locked
```

`--repeats` overrides each experiment's repeat count for runtime estimation. It does not define a proposal-equivalent run. `--workers` preserves ordered result collection. For one experiment, `--out` is that experiment's output directory. For `run all`, it is a parent directory under which the three retained directory names are created.

`figures` reads the existing compatibility tables and regenerates figures without rerunning simulation. It requires the corresponding CSV/JSON outputs to already exist.
When a manifest is present, the command restores that run's exact nested config
(including smoke-repeat and worker overrides), preserves the experiment
runtime, refreshes artifact checksums, and records a separate
`figures_only` operation runtime.

## Configuration map

Experiment config classes and their `CONFIG` instances are defined in the relevant `configs/` file. They are frozen Python dataclasses implementing the shared `ValidatedConfig` contract, so validation runs at construction and again at `configure()`. Do not replace these files with YAML, and do not consolidate defaults that happen to share a name: several experiments intentionally use different seeds, sample sizes, renal grids, effect sizes, and variance-floor semantics. Kernels re-export the class names only for legacy script and notebook compatibility.

| Scientific or computational parameter | Change it here |
|---|---|
| Latent DGP: cohort sizes, prevalence, mechanism effects, renal distortion grid, noise | `configs/endotype_discovery.py` (`SimulationConfig`) |
| Latent fitting: EM tolerance/cap, variance and probability floors, starts, calibration bins | `configs/endotype_discovery.py` (`FittingConfig`) |
| Latent repeats, master seed, worker count, validation thresholds | `configs/endotype_discovery.py` (`ExperimentConfig`) |
| Hospital definitions, assay transforms, nuisance shifts, missingness, target scenarios | `configs/transportability.py` (`HospitalSpec`, `TransportSimulationConfig`) |
| Transport repeats, master seed, source/target sizes, fitting controls, workers | `configs/transportability.py` (`TransportExperimentConfig`) |
| Known-DGP effects, renal grid, sample sizes, seed, counterfactual and K=1-null controls | `configs/counterfactual.py` (`ExperimentConfig`) |
| Shared visual palette and sizing only | `traceesus/plotting/theme.py` |

A numerical change creates a new scientific result. Preserve the old config and locked outputs, record the change, and write to a new output location rather than overwriting cited artifacts.

## Adding a model variant

The unlabeled model registry is the extension point for the planned source-of-gain ablation. A variant should require model code and one registry row, not a new simulator or repeat loop.

1. Add a `FittedModel` subclass under `traceesus/models/`. Implement `posterior(data) -> np.ndarray` with shape `(n, K)`, optional `counterfactual_scores(data)`, and the exact `n_parameters` used by BIC.
2. Add a `Model` subclass with a stable `name` and `fit(data: Cohort, rng: Generator, config: FittingConfig)`. `Cohort` contains observed fields only; do not add truth or label parameters to this interface.
3. Append the model instance to the registry in `traceesus/experiments/endotype_discovery/experiment.py`. Do not reorder existing rows: registry order is part of the fit-stream and output-row contract.
4. Run the reproducibility tests and a paired smoke experiment. `run_model_registry()` simulates the training/test cohorts once per repeat, then gives every registered model those identical cohorts under the same repeat seed ledger. It also evaluates every posterior through the common metric path.
5. Add the new method to the ablation's prespecified contrast/plot specification only if the requested table needs it. Do not let an oracle enter the fitted-model registry: `DataGeneratingOracle` remains an evaluation-only ceiling.

`FittedModel.fit_diagnostics()` is optional and defaults to `None`. Implement it only when the model has convergence evidence comparable to the existing EM fits; a model needs no kernel-specific `fit_result` field to enter posterior evaluation.

An append-only registry preserves the established cohort pairing and existing initial-fit stream positions. Any intentional change to seed allocation, retry behavior, method order, or the number/order of random draws requires a new locked baseline; it is not a refactor.

This one-row extension contract applies specifically to the executable `endotype_discovery` registry used by the planned source-of-gain ablation. The model tuple exposed by `transportability` is a compatibility/validation registry: it asserts scientific boundaries and historical row order, while its proposal-locked kernel remains the numerical dispatch path. Appending to that tuple alone does not add an evaluated method. Generalizing the transport repeat kernel would be a separately verified feature.

## Compatibility architecture

The package separates public contracts from frozen arithmetic. `traceesus/core/` owns cohort, model, experiment, metric, statistical, runner, and output contracts. `traceesus/models/`, `traceesus/simulators/`, and `traceesus/queries/` expose typed adapters. Each experiment's `kernel.py` retains any sequence of draws or floating-point operations that could affect cited outputs.

This boundary explains two intentional non-unifications. First, the preliminary counterfactual simulator and latent-discovery simulator are both two-mechanism generators, but their priors, integer dtypes, effect defaults, and normal-draw expressions differ; forcing them through one implementation would not be an identity-preserving refactor. Second, the plotting package inventories the shared palette and exact figure sizes and exposes named panel adapters, but the retained kernels preserve artist construction order so encoded figures remain reproducible.

## Exact verification

The verification protocol is intentionally stricter than numerical tolerance testing:

1. Run each untouched legacy script with its cited full configuration and copy the four output directories to an immutable `outputs_locked/` parent.
2. Run each package experiment with the identical config into a separate candidate parent.
3. Compare all legacy CSV cells in stored row/column order and every JSON key/value/type. Do not sort rows, normalize JSON, or use approximate floating-point equality.
4. Inspect every reported discrepancy back to its experiment, repeat, row, column, or JSON path. A single mismatch is a failed verification.
5. Record the completed comparison, commands, environment, and any discrepancies in `VERIFICATION.md`. Do not describe the refactor as verified before that record exists.

For candidate outputs held in `./outputs_refactored` and a locked baseline held in `./outputs_locked`:

```bash
python run.py verify \
  --against ./outputs_locked \
  --candidate ./outputs_refactored \
  --report verification_discrepancies.json
```

The command exits nonzero if any legacy CSV cell or JSON value differs and writes a machine-readable discrepancy report. Additive manifests are allowed only when their structure and checksums validate. PNGs and PDFs are regenerated, but encoded figure bytes are not substitutes for the required CSV/JSON identity gate; PDF creation metadata can differ despite identical plotted values.

For future changes, run:

```bash
python -m pytest
```

`tests/test_reproducibility.py` is the fast repeat-level sentinel. It complements, rather than replaces, the full four-experiment locked-output comparison.
The completed proposal-lock comparison, environment, counts, runtimes, and
zero-discrepancy result are recorded in [`VERIFICATION.md`](VERIFICATION.md).
