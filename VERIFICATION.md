# TRACE-ESUS exact-output verification

## Result

**PASS: no legacy CSV cell or JSON value differs.**

All four untouched legacy experiments were run first and copied into
`outputs_locked/`.  All four experiments were then run through the package
facades with the same full configurations into `outputs_refactored/`.  The
exact verifier compared 46 legacy data files, 82,616 CSV data rows, 933,333 CSV
data cells, and 423 JSON scalar leaves.  It found zero discrepancies.

After the final config-module migration, generic model-registry work, shared
anchor extraction, and CLI provenance fix, all four full experiments were run
again through `run.py` into their canonical output directories.  That final
post-change run produced the same result: 46 files compared, zero
discrepancies.  The required manifests now sit beside the canonical outputs.

This is an identity result, not a tolerance result.  CSV values were compared
as stored strings in their original row and column order.  JSON objects were
compared recursively with exact key sets, value types, list order, and
floating-point bit patterns.

## Locked source and environment

The pre-refactor script SHA-256 values were:

| Legacy script | SHA-256 |
|---|---|
| `r21_associative_vs_scm.py` | `d1a86e8696d9b5b269bb460520230da5b0a663ed8d41fb750dcf30d22e7370ed` |
| `r21_latent_endotyping_experiment.py` | `42e9be1a1085c60595b633193897b8860b9479b8e870a8a51c350efed5c1a953` |
| `r21_transportability_experiment.py` | `1f1e9de71635e86c59ced8dda7318b09103a7d4b5c135dfca041d3112c01998a` |
| `r21_preliminary_experiment.py` | `5eb16dbdfe4ced70453352aedd2265f1dc6f5cb213f8d75b90c206349c6ee005` |

The reproduction environment was the proposal environment pinned by
`pyproject.toml`:

| Component | Version |
|---|---:|
| Python | 3.12.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| SciPy | 1.14.1 |
| Matplotlib | 3.9.2 |

The baseline checkout had no Git commit: it was an unborn repository whose
files were untracked.  The destination used for verification is not a Git
working tree, so each manifest correctly records `git_commit: null` and
`git_state: "no_repository"` rather than inventing provenance.

## Protocol executed

1. Read all 6,008 lines of the four legacy scripts before package design.
2. Ran all four legacy scripts with their full cited defaults.
3. Confirmed the fresh legacy CSV and JSON artifacts were exact matches to the
   pre-existing canonical output trees.
4. Snapshotted the resulting 62-file baseline under `outputs_locked/`.
5. Refactored in the required sequence: latent discovery, supervised model
   comparison, transportability, then the preliminary counterfactual study.
6. Ran all four package experiments at full defaults into
   `outputs_refactored/`.
7. Executed the exact comparison:

   ```bash
   .venv/bin/python run.py verify \
     --against ./outputs_locked \
     --candidate ./outputs_refactored \
     --report ./verification_discrepancies.json
   ```

8. Executed the rendered-output verifier and independently rasterized all PDFs
   at 180 dpi for a fixed-pixel comparison and layout inspection.
9. Executed the fast committed reproducibility suite for every package facade.
10. Repeated the full default package runs after the last code change and ran
    the root verifier against the canonical output directories.

The final package wall-clock runtimes recorded in the canonical manifests were:

| Experiment | Repeats | Workers | Runtime (seconds) |
|---|---:|---:|---:|
| Supervised model comparison | 500 per level | 1 | 206.2843 |
| Latent endotype discovery | 500 per level plus 500 null | 4 | 524.0758 |
| Transportability | 500 | 4 | 419.2662 |
| Known-DGP counterfactual | 500 per level plus 500 null | 1 | 22.4947 |

The managed sandbox prohibits process semaphores in its default profile.  The
combined `run all` command therefore completed the sequential supervised run,
then stopped before latent dispatch.  The latent and transport commands were
resumed with permission to use their recorded four-worker configuration; no
seed, config, repeat count, or output path changed.  Counterfactual then ran
sequentially.  Each experiment was exact-gated immediately after completion,
followed by the four-tree root comparison.

## Exact data comparison

| Experiment output | CSV | JSON | Data rows | Data cells | JSON scalar leaves | Differences |
|---|---:|---:|---:|---:|---:|---:|
| `outputs/` | 5 | 1 | 12,533 | 89,385 | 41 | 0 |
| `outputs_associative_vs_scm/` | 4 | 1 | 14,040 | 106,440 | 29 | 0 |
| `outputs_latent_endotyping/` | 7 | 3 | 17,655 | 169,076 | 71 | 0 |
| `outputs_transportability/` | 17 | 8 | 38,388 | 568,432 | 282 | 0 |
| **Total** | **33** | **13** | **82,616** | **933,333** | **423** | **0** |

The machine-readable result is `verification_discrepancies.json`:

```json
{
  "status": "pass",
  "compared_file_count": 46,
  "discrepancy_count": 0,
  "discrepancies": []
}
```

The verifier also rejects missing or extra legacy files.  The only permitted
candidate-only files were the four required additive `manifest.json` files;
each manifest's config, seed, runtime, software versions, Git state, and
recursive checksum inventory validated.

## Figures

All eight PNGs had identical decoded dimensions and RGBA pixels.  The rendered
comparison therefore passed 54 files in total: the 46 CSV/JSON artifacts plus
8 PNGs.

PDF byte identity is not used as a scientific gate because Matplotlib embeds a
generation timestamp in PDF metadata.  The eight locked and refactored PDFs
were instead rasterized with the same Poppler command at 180 dpi.  Every
corresponding raster was byte-identical, and the contact-sheet inspection found
no clipping, overlap, missing panels, or broken text.  Raw PDF byte differences
were confined to expected metadata and do not reflect plotted-value drift.

## Reproducibility tests

`tests/test_reproducibility.py` runs every OOP experiment facade with two main
repeats (and two null repeats where applicable), while retaining the default
sample sizes and numerical controls.  Its committed fixture locks SHA-256 for
all 46 recursive legacy CSV/JSON artifacts.  It also verifies:

- complete manifest checksum inventories;
- audited latent, null, transport, and target-child seed sentinels;
- absence of truth from `Cohort` and all unsupervised fit signatures;
- the supervised-training boundary in model-comparison metadata; and
- ordered repeat results across one and two workers when the platform permits
  process semaphores.

The focused reproducibility run reported `11 passed, 1 skipped` in 51.62 seconds.  The
single skip is environmental: this managed sandbox prohibits the process
semaphore used by the two-worker test.  The same test remains active in normal
CI and local environments.  There were zero artifact-hash or manifest
discrepancies.

The final complete suite, including verification utility, architecture,
registry-extension, config round-trip, and root-CLI manifest tests, reported
`44 passed, 1 skipped` with no failures.  A static AST audit also found zero
public package classes/functions/methods missing docstrings, parameter
annotations, or return annotations.

## Interpretation boundary

This verification establishes that modular execution reproduces the cited
synthetic outputs exactly.  It does not validate the scientific realism of the
data-generating processes, convert the supervised comparison into endotype
discovery, or turn oracle/known-DGP queries into deployable fitted models.
Preserved defects, surprising behavior, dead controls, and metadata wording
problems are recorded without correction in `NOTES.md`.
