# TRACE-ESUS legacy notes

This file records behavior found in the four legacy scripts before modularization. It is not a change request. Numerical identity takes priority: the behaviors below must remain unchanged for the locked-output refactor, even when they are awkward or defective. Any scientific or numerical correction requires a separately versioned experiment and new outputs.

Terminology used below:

- **Scientific boundary**: a limitation or estimand distinction that is intentional and must remain explicit.
- **Verified defect**: source behavior demonstrably contradicts its own label, metadata, validation claim, or public configurability. “Verified” does not imply that the default cited numbers are affected.
- **Preserved quirk**: surprising or inconsistent behavior whose intent cannot be established from the source alone. It is flagged, not corrected.

## Scientific boundaries that must survive the refactor

### Model comparison is supervised classification

`r21_associative_vs_scm.py::run_comparison()` creates `atrial_training_label` from `training["mechanism"]` and passes it to both logistic fits. It also passes `training["mechanism"]` to `fit_structural_causal_model()`. Therefore all three methods use true synthetic labels during training. This experiment is **not** latent endotype discovery. Preserve the statement currently saved as `metadata.json -> training_label_boundary`.

The adjusted logistic control is scientifically necessary: it distinguishes a graph/query comparison from the simpler advantage of giving renal status to one method but not another. Preserve the module-level explanation and the biomarkers-only versus kidney-adjusted feature sets.

### Latent discovery is unlabeled, with a truth-only evaluation boundary

`r21_latent_endotyping_experiment.py::_run_one_repeat()` passes biomarkers and renal status, but not `true_mechanism`, to the associative, adjusted, and causal fit functions. `evaluate_posterior()` receives truth only after fitting. The data-generating oracle is an evaluation ceiling, not a fitted competitor.

`_anchor_order()` is load-bearing. It orients labels by the prespecified standardized contrast “atrial electrical minus competing-specific” and deliberately excludes both truth and the renal-distorted NT-proBNP-like marker. Do not replace it with post-hoc truth matching.

The adjusted associative model estimates renal paths for every biomarker (14 K=2 parameters); the constrained causal model estimates the renal path only for the NT-proBNP-like marker (12 parameters). The primary associative model also has 12 parameters. Those counts and path masks define the comparison.

### The preliminary experiment uses a known data-generating SCM, not a fitted SCM

Despite descriptions elsewhere of a “fitted SCM,” `r21_preliminary_experiment.py` contains no SCM fitting step. `kidney_aware_posterior()` and `posterior_integrated_counterfactual_scores()` read the prespecified `ExperimentConfig.mechanism_effects`, biomarker noise, structural prior, and the true per-level `renal_effect_sd` used by the simulator. It is an oracle/known-model query comparison.

Do not introduce fitting during an identity-preserving refactor. That would define a new experiment and change every cited result.

The module and `posterior_integrated_counterfactual_scores()` docstrings contain a critical limitation: in this symmetric K=2 model, normalized sufficiency and disablement are monotone transformations of the correctly specified posterior. The same-SCM posterior is the Bayes top-1 diagnostic; a causal query cannot outperform it “by magic.” The current saved `main_simulation_raw_metrics.csv` has no counterfactual-versus-same-SCM metric differences.

The same limitation carries into `r21_associative_vs_scm.py::scm_counterfactual_scores()`. `run_comparison()` explicitly records `maximum_counterfactual_posterior_difference`; the current saved diagnostics are at floating-point noise (maximum approximately `7.77e-16`). Any gain over an associative comparator is therefore not evidence that the counterfactual query itself beats the same-SCM posterior.

### Transportability assumes known assay calibration and unlabeled target recalibration

`r21_transportability_experiment.py::assay_calibrate()` exactly inverts each hospital’s known offset and scale before any method is fit or evaluated. Target nuisance-path calibration uses renal status, inflammation status, and unlabeled biomarkers only. Mechanism labels are used only for post-prediction evaluation.

The “No shift” point in the main curve matches the reference source hospital, not the full three-hospital source mixture. `run_exact_no_shift_negative_control()` is the literal source-target identity control. Preserve its docstring and do not collapse these two controls.

The target scenarios deliberately reuse the same calibration and test seeds by reconstructing a fresh generator from `target_seed_calibration` and `target_seed_test` for every shift. This is a common-random-number pairing device, not an accidental seed collision.

### The illustrative latent patient is selected, not representative

`r21_latent_endotyping_experiment.py::build_example_patient()` uses truth and method outcomes to select a renal-impaired, true-competing patient, preferring an associative-atrial/causal-competing disagreement. Within eligible patients it uses a hard-coded score containing posterior terms and biomarker weights (`0.25`, `-0.10`, `0.10`). Figure P2 is a deliberately selected illustration, not a random case or population estimate.

## Verified defects preserved for output identity

1. **The latent truth-interface validation is not a validation.** `r21_latent_endotyping_experiment.py::validation_checks()` sets `truth_not_accepted_by_fit_function_interfaces` to the literal `True`. It neither inspects signatures nor asserts the separation, and `all_required_checks_pass` does not include that field. The current interfaces are truth-free, but this saved check cannot detect a regression.

2. **Latent truth-use metadata is literally too narrow.** `run_full_experiment()` says true mechanisms are “used only by evaluate_posterior,” but `build_example_patient()` also reads `test.true_mechanism` for case selection. This does not leak truth into fitting; it does make the metadata claim false.

3. **Transport truth-use metadata is literally too narrow.** Its main metadata says labels are used only inside `evaluate_posterior`, while `_extended_metrics()` directly reads `cohort.true_mechanism` for complete-case, any-missing, and electrical-missing accuracies. Again, this is evaluation-only, not a fitting leak.

4. **Custom-repeat captions can be wrong.** `r21_associative_vs_scm.py::run_full_comparison()` hard-codes “500 paired train/test simulation repeats” in `metadata.json -> ci_definition`, even when `--repeats` overrides the count. `r21_transportability_experiment.py::plot_ablation_figure()` hard-codes “500 paired repeats,” so smoke-run figures are mislabeled. `run_comparison()` also has a hard-coded “Run 500” docstring.

5. **Transport plotting overstates config generality.** `plot_transport_figure()` hard-codes “Three unlabeled source hospitals,” although validation permits any source tuple of length at least two. More seriously, `_run_one_repeat()` spawns exactly 12 child sequences and reserves indices `3` through `10` for fits/targets while source simulation consumes indices beginning at `0`. A fourth configured source would reuse sequence `3`, and sufficiently many sources would index past the allocation. The default three-source experiment is not affected.

6. **The custom adjusted Rand index is wrong for a degenerate label permutation.** `r21_latent_endotyping_experiment.py::_adjusted_rand_index()` returns `0.0` when both partitions consist of one identical cluster encoded with different numeric labels (for example, all-zero truth and all-one predictions), because its zero-denominator branch uses `np.array_equal`. ARI is label-invariant and should be `1.0` for identical partitions. The default recovery design is not known to exercise this edge case; changing it would still violate locked-output identity unless proven otherwise.

7. **The supervised comparison metadata omits generator parameters.** `r21_associative_vs_scm.py::run_comparison()` silently constructs `r21_preliminary_experiment.ExperimentConfig(seed=config.seed)` and changes only patient counts. Consequently its data-generating renal prevalence, mechanism effects, noise, and renal-to-mechanism prior come from another module, but `outputs_associative_vs_scm/metadata.json` serializes only `ComparisonConfig`. A change to preliminary defaults can change supervised outputs without appearing in that legacy config block.

## Preserved scientific and numerical quirks

### Counterfactual normalization has unguarded degenerate cases

Both counterfactual implementations divide disablement by the squared norm of a candidate effect and divide sufficiency by `1 - mismatch_fit`. Config validation does not require nonzero candidate effects or distinct mechanism-effect vectors. The default effects make both denominators nonzero. Treat this as an unexercised edge risk, not evidence of a defect in the cited run.

### “Variance floor” does not have one meaning

- Latent discovery and transport use `FittingConfig.variance_floor = 0.05**2` as a variance floor.
- Preliminary K=1/K=2 null fitting uses `gmm_variance_floor = 0.05` directly as a variance floor.
- Supervised `ComparisonConfig.variance_floor = 1e-6` is applied to an estimated **standard deviation** in `fit_structural_causal_model()` despite its name.

Do not normalize these values or semantics during extraction.

### The K=1 null experiments answer different questions

The latent-discovery null compares K=1 and K=2 by BIC separately for the associative, fully adjusted, and constrained causal families while retaining renal distortion.

The preliminary null instead subtracts the **known true renal contribution** before fitting ordinary diagonal Gaussian K=1/K=2 models. Its `select_k2` is not BIC alone: K=2 must also converge and have minimum component weight at least `0.10`. The summary label “BIC-selected” is therefore shorthand for a composite decision rule. This null does not test a learned renal adjustment.

### Assay-only transport is a negative control, not a stress test

`TransportSimulationConfig.assay_metadata_known` is never read. Known calibration is always applied to every method. Therefore the assay-only ablation is algebraically neutral; the current saved `ablation_accuracy_changes.csv` reports exactly `0.0` for every method. Preserve this result and describe it as a calibration negative control, not evidence of robustness to unknown assay shift.

### Missingness effects have a perhaps-surprising sign

`renal_missingness_log_odds_nt = -0.60` and `inflammation_missingness_log_odds_competing = -0.60` are added to the log odds of **missingness**. Thus renal dysfunction reduces NT-proBNP-like missingness and inflammation reduces competing-marker missingness relative to their hospital base rates. This may represent targeted measurement, but the source does not say. Do not silently flip the signs.

### Latent example fitting and selection have no retry/check

`build_example_patient()` fits one associative and one causal model without the non-convergence retry used by `_run_one_repeat()`, then selects a display case without checking the two fit objects’ `converged` fields. Defaults appear intended to make this stable; preserve the path for identity.

### Confidence intervals and contrast signs differ across scripts

- Preliminary and supervised summaries clip rate CIs to `[0, 1]`; latent and transport summary tables do not clip them (plots may clip visually).
- Latent and transport paired contrasts use a literal causal-minus-comparator sign for all metrics.
- Preliminary and supervised scripts reverse false-attribution error differences so positive means fewer errors for the causal/counterfactual method, and rename the contrast accordingly.

These are output-contract differences, not interchangeable helper behavior.

### Config validation is incomplete but defaults are valid

The frozen dataclasses do not call `validate()` at construction; validation occurs at run entry points. Several fields are not validated at all. Examples include preliminary null sample size/start/iteration/floor controls, supervised logistic iteration/tolerance/floor controls, transport missingness-log-odds finiteness, and the semantic ordering of target hospitals. Do not add validation that alters a default run or its RNG path during identity verification.

Transport summaries and controls assume target index `0` is the no-shift baseline and the maximum index is the strongest shift. `TransportExperimentConfig.validate()` checks only that at least one target exists, not those semantics.

## Defaults that are intentionally inconsistent across experiments

| Property | Counterfactual preliminary | Supervised model comparison | Latent discovery | Transportability |
|---|---:|---:|---:|---:|
| Base seed | `20_260_728` | `20_260_729` | `20_260_728` | `20_260_728 + 404_404` for repeats |
| Main sample sizes | `1,000` per repeat | `3,000` train / `1,000` test | `800` train / `1,000` test | `3 x 600` source / `150` target calibration / `1,000` target test |
| Renal-distortion grid | `0, 0.75, 1.50, 2.25` | `0, 0.75, 1.50, 2.25` | `0, 0.50, 1.00, 1.50` | hospital-specific |
| Atrial signature | `(1.20, 0.80, 0.00)` | inherited from preliminary | `(1.25, 1.00, 0.00)` | `(1.25, 1.00, 0.00)` |
| Atrial probability | renal-dependent: `logit = 0 - 0.40 R` | inherited from preliminary | `0.50` in both renal strata | `0.50`, independent of nuisances |

These differences change the scientific estimand. Shared config classes must not accidentally make them uniform.

## RNG and order-of-consumption quirks that are load-bearing

- Preliminary main repeats use `SeedSequence(seed) -> level -> repeat -> (data, ties)`, then spawn three separate tie generators in method order. Its null uses `seed + 1_000_003`.
- Supervised comparison uses `SeedSequence(seed) -> level -> repeat -> (train, test, ties)` and three method-specific tie generators.
- Latent recovery derives a uint64 seed per level/repeat, then `_run_one_repeat()` spawns eight streams in the fixed order simulation, test, three primary fits, and three retry fits. Its K=1 null uses `master_seed + 91_337`; its example uses `master_seed + 808_080`.
- Transport uses `master_seed + 404_404`, then 12 per-repeat streams. With the default three sources, streams `0:3` simulate sources, `3:9` fit/retry the three mixtures, and generated states from streams `9` and `10` seed all paired target scenarios. Stream `11` is unused.

Do not consolidate these offsets, replace `SeedSequence`, share generators, lazily spawn retry streams, or reorder model iteration.

## Output-contract inconsistencies to preserve

- Supervised comparison writes `outputs_associative_vs_scm/{raw_metrics.csv, fit_diagnostics.csv, summary.csv, paired_contrasts.csv, metadata.json, figure_P1_associative_vs_scm.*}` and no legacy `validation_checks.json`.
- Latent discovery writes `outputs_latent_endotyping/` with `metadata.json`, `validation_checks.json`, the recovery/diagnostic/null CSV names, `example_patient.json`, and P1/P2/S1 figures.
- Transport main writes `metadata.json` and `validation_checks.json`; `ablations/` writes validation and negative-control JSON but no metadata; `exact_no_shift_control/` uses generic `raw_metrics.csv`, `summary.csv`, and `paired_contrasts.csv` names rather than the main transport prefixes.
- Preliminary writes the generic `outputs/` directory, calls its metadata `run_metadata.json`, and has no legacy `validation_checks.json`.
- “Figure P1” is reused by multiple experiments but remains namespaced by output directory and filename.

Compatibility code must reproduce these legacy names exactly. New `manifest.json` files are additive and should not be used to rename or rewrite legacy artifacts.

## Dead or currently ineffective source elements

- `traceesus/models/clinical_rule.py` is intentionally a reserved module. None
  of the four legacy experiments implements a clinical decision rule, so the
  refactor does not invent or relabel one merely to populate the target tree.

- `r21_preliminary_experiment.py::posterior_integrated_counterfactual_scores()` assigns `factual_gate = effects[branch]` but never reads it.
- `r21_transportability_experiment.py::_run_one_repeat()` constructs a target `true_slopes` array immediately before diagnostics but never reads it.
- `TransportSimulationConfig.assay_metadata_known` has no effect; calibration is unconditional.
- The twelfth per-repeat transport seed stream is unused.
- `Sequence` is imported by the transport script but unused.

Removing dead code can still perturb imports, serialized config shape, or future notebook assembly. Leave it in the legacy compatibility layer until output verification is complete.

## Refactor boundaries that remain intentionally narrow

### Config ownership changed, numerical defaults did not

The canonical frozen dataclass definitions now live in `configs/`, exactly where `run.py list` directs users. Every class inherits `ValidatedConfig` and invokes `validate()` during construction. The experiment kernels import and re-export those same class objects, so legacy scripts and notebook imports retain their historical symbol names. Field order, default values, nested default construction, validation predicates, and serialized manifest/metadata shapes remain unchanged. This lifecycle repair catches an invalid config earlier; it does not add new numerical checks or alter a valid run.

### Anchor orientation is shared only at the exact arithmetic boundary

Latent discovery and transport now delegate to one raw-array `anchor_order()` helper. Their public/internal wrapper signatures remain intact. The helper preserves the original `sqrt`, indexed divisions, subtraction, `argmax` tie behavior, label ordering, and margin subtraction order verbatim. A randomized bit-level test compares the shared helper and both wrappers to the pre-extraction expression. Higher-level fit orientation remains in each kernel because the surrounding fitted objects and missing-data responsibilities differ.

### Only the endotype-discovery registry is numerical dispatch

The `endotype_discovery` facade executes its ordered `Model` registry through `run_model_registry()`. This is the supported append-only extension point for the planned source-of-gain ablation: existing initial-fit and retry streams remain fixed, and new variants receive later streams.

The `model_comparison` and `transportability` facades also expose model tuples, but those are compatibility/validation registries. Their frozen repeat kernels still dispatch the proposal-locked methods directly. Appending a class to either tuple does not add a row to its experiment. Replacing those kernels with generic registry dispatch could be valuable, but it requires a separate locked-output proof because fit retries, target recalibration, and tie streams are interleaved differently. The current package documents this honestly rather than presenting a facade tuple as functionality it does not provide.

### The two-mechanism simulators are not numerically interchangeable

The preliminary/model-comparison generator and the latent-discovery generator share a scientific outline but not an exact implementation contract. The former uses a renal-dependent logistic mechanism prior, ordinary integer arrays, preliminary-specific effects `(1.20, 0.80, 0.00)`, and `rng.normal(size=...) * noise_sd`. The latter uses two configured stratum probabilities, `int8` labels/covariates, effects `(1.25, 1.00, 0.00)`, and `rng.normal(0.0, noise_sd, size=...)`. Even when configured to superficially similar values, unifying these expressions can alter array dtypes or floating-point results. They therefore remain separate exact kernels behind a common `Simulator`/`SimulatedData` contract.

The two K=1 null generators diverge more sharply: latent discovery retains renal distortion while comparing three model families; the preliminary null subtracts the known renal path and applies a convergence/minimum-weight gate in addition to BIC. Treating them as one configurable algorithm would conceal different estimands.

### Shared plotting means a stable surface, not normalized legacy figures

`traceesus.plotting.theme` records the common palette, font, and exact per-figure dimensions, and `traceesus.plotting.panels` exposes named adapters for all figure families. The proposal-locked kernels still construct artists locally. Their figure sizes, marker styles, legend placement, and artist insertion order are not actually uniform; redirecting them through a new generic panel builder could change raster or PDF output without changing a table. New figures should use the shared theme, while compatibility figures remain frozen until separately proven identical.
