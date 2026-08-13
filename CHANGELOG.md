# Changelog

## 2026-08-13 — Supervised comparison archived

- Reduced the executable package to three experiments: `endotype_discovery`,
  `transportability`, and `counterfactual`.
- Archived the supervised model-comparison package, configuration, models,
  entry points, notebook builder, notebook, plotting adapter, HF-grid path, and
  their prior shared-module surfaces under `archive/supervised/`.
- Removed the supervised HF-grid working table and figures and the working
  `outputs_associative_vs_scm/` directory.
- Retained the unsupervised HF grid, confounding sweep, and redundancy extension.
- Kept extension-only HF controls out of the proposal-locked discovery metadata
  and manifest serialization so the retained legacy output schema remains exact.
- Rationale: mechanism labels cannot exist in a real ESUS cohort, so supervised
  comparisons are excluded from the preliminary evidence to keep it focused.
- `outputs_locked/outputs_associative_vs_scm/` remains unchanged as immutable
  provenance for the previously cited result.
