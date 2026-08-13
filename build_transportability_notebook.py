"""Build the executable reader-facing R21 transportability notebook."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path("R21_transportability_experiment.ipynb")


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}

notebook["cells"] = [
    markdown(
        r"""
# R21 cross-hospital transportability experiment

## tl;dr

This notebook tests a narrow hypothesis: **does a biologically constrained
latent causal model preserve hidden-mechanism recovery when nuisance biomarker
relationships change in a new hospital?**

Across 500 paired repeats, strong target-hospital shift produced:

| Model | True-mechanism accuracy | False atrial classification in renal/competing subgroup |
|---|---:|---:|
| Pooled associative latent class | 73.37% | 44.73% |
| Target-calibrated associative latent model | 77.14% | 23.37% |
| Modular causal latent SCM | 77.60% | 22.52% |

The modular causal model gained **4.23 percentage points** over the pooled
associative model, but only **0.47 points** over the well-specified adjusted
associative control. In a separate exact source-target identity control, the
causal-adjusted difference was 0.28 points (95% Monte Carlo CI, 0.23 to 0.33),
inside the prespecified ±1-point equivalence margin.

**Conclusion:** the experiment supports improved transportability relative to a
transport-naive associative latent class model and a small improvement beyond
flexible target adjustment. It does not show that causal models universally
beat associative learning.
"""
    ),
    markdown(
        r"""
## Context and methods

### Scientific question

Each artificial patient has one hidden categorical mechanism: atrial or
competing. A new hospital changes renal dysfunction, inflammation, assay
calibration, and missingness while the biological mechanism signatures remain
stable. Neither model sees the mechanism labels.

The primary outcome is top-rank true-mechanism accuracy in an independent
target test cohort. The key subgroup outcome is false atrial classification
among renal-impaired patients whose true mechanism is competing.

### Generated world

\[
\mathbf B_i^*
=
\boldsymbol\lambda_{Z_i}
+(\gamma_hR_i,0,0)
+(0,0,\delta_hI_i)
+\boldsymbol\epsilon_i,\qquad
\boldsymbol\epsilon_i\sim N(\mathbf0,I_3)
\]

\[
\boldsymbol\lambda_A=(1.25,1.00,0.00),\qquad
\boldsymbol\lambda_C=(0.00,0.00,1.00).
\]

The stable biomarker order is NT-proBNP-like, atrial electrical, and
competing-specific. Hospital-specific assay offsets and scales are assumed
known and are inverted equally for every method.
"""
    ),
    code(
        """
from dataclasses import asdict
from pathlib import Path
import inspect
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from IPython.display import Image, display

from r21_transportability_experiment import (
    ABLATION_TARGETS,
    MAIN_METHODS,
    MODULAR_CAUSAL,
    POOLED_ASSOCIATIVE,
    SOURCE_HOSPITALS,
    TARGET_ADJUSTED_ASSOCIATIVE,
    TARGET_HOSPITALS,
    TransportExperimentConfig,
    fit_missing_gaussian_mixture,
    fit_nuisance_paths,
)

PROJECT_DIRECTORY = Path.cwd()
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs_transportability"
RUN_FULL_SIMULATION = False  # Set True to regenerate all 500-repeat outputs.

config = TransportExperimentConfig(repeats=500, workers=4)
config.validate()
pd.Series(
    {
        "Monte Carlo repeats": config.repeats,
        "source hospitals": len(config.source_hospitals),
        "patients per source hospital": config.simulation.source_patients_per_hospital,
        "unlabeled target calibration patients": config.simulation.target_calibration_patients,
        "independent target test patients": config.simulation.target_test_patients,
        "master seed": config.master_seed,
        "no-shift equivalence margin": config.equivalence_margin_accuracy,
    },
    name="value",
)
"""
    ),
    markdown(
        """
### Reproduce the complete analysis

The default notebook reads the committed replicate-level outputs so that it
opens quickly. Set `RUN_FULL_SIMULATION = True` above to rerun the main
experiment, one-factor ablations, and the literal identical-distribution
negative control.
"""
    ),
    code(
        """
required_outputs = [
    OUTPUT_DIRECTORY / "raw_transport_metrics.csv",
    OUTPUT_DIRECTORY / "transport_summary.csv",
    OUTPUT_DIRECTORY / "paired_transport_contrasts.csv",
    OUTPUT_DIRECTORY / "transport_degradation.csv",
    OUTPUT_DIRECTORY / "validation_checks.json",
    OUTPUT_DIRECTORY / "figure_T1_transportability.png",
    OUTPUT_DIRECTORY / "figure_T2_transport_controls.png",
    OUTPUT_DIRECTORY / "ablations" / "ablation_accuracy_changes.csv",
    OUTPUT_DIRECTORY / "ablations" / "figure_T3_shift_ablations.png",
    OUTPUT_DIRECTORY / "exact_no_shift_control" / "summary.csv",
    OUTPUT_DIRECTORY / "exact_no_shift_control" / "negative_control.json",
]

if RUN_FULL_SIMULATION or not all(path.exists() for path in required_outputs):
    command = [
        sys.executable,
        "r21_transportability_experiment.py",
        "--repeats",
        "500",
        "--workers",
        "4",
        "--with-ablation",
        "--with-exact-negative-control",
        "--output-dir",
        str(OUTPUT_DIRECTORY),
    ]
    subprocess.run(command, check=True)

assert all(path.exists() for path in required_outputs)
print("All required outputs are present.")
"""
    ),
    markdown(
        """
### Source and target environments

The main curve uses three heterogeneous source hospitals. Its reference target
matches Source B. Because that is not literally identical to the pooled
multi-source distribution, a separate exact control makes every source and
target hospital identical.
"""
    ),
    code(
        """
def hospital_table(hospitals):
    return pd.DataFrame(
        [
            {
                "hospital": h.name,
                "renal prevalence": h.renal_prevalence,
                "renal effect on NT-like marker (SD)": h.renal_effect_nt_sd,
                "inflammation prevalence": h.inflammation_prevalence,
                "inflammation effect (SD)": h.inflammation_effect_competing_sd,
                "assay offset": h.assay_offset,
                "assay scale": h.assay_scale,
                "base missingness": h.missingness_base,
            }
            for h in hospitals
        ]
    )

display(hospital_table(SOURCE_HOSPITALS))
display(hospital_table(TARGET_HOSPITALS))
"""
    ),
    markdown(
        r"""
### Models

| Model | What is learned | Target information | Purpose |
|---|---|---|---|
| Pooled associative latent class | \(p(Z)\prod_jp(B_j\mid Z)\) | assay metadata | transport-naive baseline |
| Target-calibrated associative latent model | every renal/inflammation-to-biomarker association | 150 unlabeled patients | strongest associative control |
| Frozen causal latent SCM | restricted nuisance graph, source-average coefficients | assay metadata | tests whether graph alone transports |
| Modular causal latent SCM | restricted graph plus target-specific permitted coefficients | 150 unlabeled patients | proposed transport model |
| Target oracle | true generating parameters | unavailable in practice | information ceiling |

Every latent model is an unsupervised two-component diagonal Gaussian mixture
with a missing-data likelihood and four EM starts. A prespecified
electrical-minus-competing marker contrast names the atrial component. The
misleading NT-proBNP-like marker and truth labels are excluded from class
orientation.
"""
    ),
    markdown(
        """
### Verify that truth cannot enter either fitting interface

The simulator retains truth separately for evaluation, but the fit functions
have no mechanism-label argument.
"""
    ),
    code(
        """
fit_interfaces = {
    "latent mixture": str(inspect.signature(fit_missing_gaussian_mixture)),
    "nuisance path estimator": str(inspect.signature(fit_nuisance_paths)),
}
assert all("true_mechanism" not in signature for signature in fit_interfaces.values())
fit_interfaces
"""
    ),
    markdown("## Data and primary results"),
    code(
        """
raw = pd.read_csv(OUTPUT_DIRECTORY / "raw_transport_metrics.csv")
summary = pd.read_csv(OUTPUT_DIRECTORY / "transport_summary.csv")
contrasts = pd.read_csv(OUTPUT_DIRECTORY / "paired_transport_contrasts.csv")
degradation = pd.read_csv(OUTPUT_DIRECTORY / "transport_degradation.csv")
ablations = pd.read_csv(
    OUTPUT_DIRECTORY / "ablations" / "ablation_accuracy_changes.csv"
)
exact_summary = pd.read_csv(
    OUTPUT_DIRECTORY / "exact_no_shift_control" / "summary.csv"
)

assert raw.shape[0] == 500 * 4 * 5
assert raw["repeat"].nunique() == 500
assert raw["shift"].nunique() == 4
print(f"Loaded {raw.shape[0]:,} method-by-shift replicate rows.")
"""
    ),
    code(
        """
short_names = {
    POOLED_ASSOCIATIVE: "Pooled associative",
    TARGET_ADJUSTED_ASSOCIATIVE: "Adjusted associative",
    MODULAR_CAUSAL: "Modular causal",
}
accuracy = summary[
    summary["metric"].eq("accuracy") & summary["method"].isin(MAIN_METHODS)
].copy()
accuracy["method"] = accuracy["method"].map(short_names)
main_table = (
    accuracy.pivot(index="shift", columns="method", values="mean")
    .loc[["No shift", "Mild shift", "Moderate shift", "Strong shift"]]
    * 100
)
main_table.round(2)
"""
    ),
    code(
        """
display(
    Image(
        filename=str(OUTPUT_DIRECTORY / "figure_T1_transportability.png"),
        width=1200,
    )
)
"""
    ),
    markdown(
        """
Figure T1 is the primary result. The pooled associative model increasingly
mistakes a nuisance-driven NT-proBNP-like elevation for the atrial mechanism.
Both target-calibrated methods are more stable. The causal constraints add a
small improvement beyond flexible associative adjustment, not a categorical
victory.
"""
    ),
    markdown("## Paired strong-shift contrasts"),
    code(
        """
strong = contrasts[
    contrasts["shift"].eq("Strong shift")
    & contrasts["metric"].isin(
        [
            "accuracy",
            "false_atrial_renal_competing",
            "brier_score",
            "expected_calibration_error",
            "accuracy_any_missing",
        ]
    )
].copy()
strong["estimate"] = 100 * strong["mean_difference"]
strong["CI low"] = 100 * strong["ci95_low"]
strong["CI high"] = 100 * strong["ci95_high"]
strong[
    ["comparator", "metric", "estimate", "CI low", "CI high"]
].round(3)
"""
    ),
    markdown(
        """
For accuracy, positive values favor the modular causal model. For false atrial
classification, Brier score, and calibration error, negative values favor it.
The estimates are paired by simulation repeat.
"""
    ),
    code(
        """
display(
    Image(
        filename=str(OUTPUT_DIRECTORY / "figure_T2_transport_controls.png"),
        width=1200,
    )
)
"""
    ),
    markdown(
        """
The frozen causal model is an important falsification control. It deteriorates
when target nuisance coefficients change, showing that a correct causal graph
does not magically transport. The changing modules still have to be measured
and recalibrated.
"""
    ),
    markdown("## Literal identical-distribution negative control"),
    code(
        """
with open(
    OUTPUT_DIRECTORY / "exact_no_shift_control" / "negative_control.json",
    encoding="utf-8",
) as stream:
    exact_negative_control = json.load(stream)

exact_accuracy = exact_summary[
    exact_summary["metric"].eq("accuracy")
    & exact_summary["method"].isin(
        [TARGET_ADJUSTED_ASSOCIATIVE, MODULAR_CAUSAL]
    )
][["method", "mean", "ci95_low", "ci95_high"]].copy()
exact_accuracy[["mean", "ci95_low", "ci95_high"]] *= 100
display(exact_accuracy.round(3))
exact_negative_control
"""
    ),
    markdown(
        """
The 95% interval for modular causal minus adjusted associative accuracy lies
entirely within ±1 percentage point. This passes the prespecified equivalence
negative control. The main curve's reference target is retained as a useful
Source-B reference, but it is not mislabeled as literal distributional
identity.
"""
    ),
    markdown("## One-factor shift ablations"),
    code(
        """
ablation_table = ablations[
    ablations["method"].isin(
        [POOLED_ASSOCIATIVE, TARGET_ADJUSTED_ASSOCIATIVE, MODULAR_CAUSAL]
    )
].copy()
ablation_table["method"] = ablation_table["method"].map(short_names)
ablation_table["accuracy change (pp)"] = 100 * ablation_table["mean_difference"]
(
    ablation_table.pivot(
        index="shift",
        columns="method",
        values="accuracy change (pp)",
    )
    .loc[
        [
            "Kidney only",
            "Inflammation only",
            "Assay only",
            "Missingness only",
            "Combined strong",
        ]
    ]
    .round(2)
)
"""
    ),
    code(
        """
display(
    Image(
        filename=str(
            OUTPUT_DIRECTORY / "ablations" / "figure_T3_shift_ablations.png"
        ),
        width=1100,
    )
)
"""
    ),
    markdown(
        """
The ablations separate three stories:

1. The renal-path shift is the part the structural models can correct.
2. Increased missingness harms every method, including the oracle, because it
   removes information.
3. The assay-only effect is exactly zero because calibration metadata are known
   and inverted for every model. Unknown assay drift is outside this experiment.
"""
    ),
    markdown("## Validation checks"),
    code(
        """
with open(OUTPUT_DIRECTORY / "validation_checks.json", encoding="utf-8") as stream:
    validation = json.load(stream)
with open(
    OUTPUT_DIRECTORY / "ablations" / "validation_checks.json",
    encoding="utf-8",
) as stream:
    ablation_validation = json.load(stream)
with open(
    OUTPUT_DIRECTORY / "exact_no_shift_control" / "validation_checks.json",
    encoding="utf-8",
) as stream:
    exact_validation = json.load(stream)

assert validation["all_required_checks_pass"]
assert ablation_validation["all_required_checks_pass"]
assert exact_validation["all_required_checks_pass"]
pd.DataFrame(
    {
        "main": pd.Series(validation),
        "ablations": pd.Series(ablation_validation),
        "exact no shift": pd.Series(exact_validation),
    }
)
"""
    ),
    markdown(
        r"""
## What the results mean

The transport-naive latent class model learns that a high NT-proBNP-like marker
often accompanies the atrial class in its training mixture. When renal disease
becomes more common and produces a larger NT-proBNP-like elevation in the
target hospital, that association stops transporting.

The modular SCM represents two different explanations for the same observed
marker: an atrial mechanism and a renal nuisance path. It estimates the renal
path from unlabeled target patients, removes that contribution, and applies the
stable latent mechanism model to what remains. This preserves substantially
more accuracy than the pooled baseline.

The fair associative control also observes renal and inflammation status and
uses the same unlabeled target calibration cohort. It nearly matches the SCM.
The residual causal advantage comes from estimating only biologically permitted
paths instead of every covariate-marker association.

### Defensible claim

> In this prespecified synthetic cross-hospital experiment, the modular latent
> SCM preserved hidden-mechanism recovery better than a pooled associative
> latent class model and modestly better than a flexible target-calibrated
> associative latent model, while satisfying an identical-distribution
> equivalence negative control.

### What is not established

- general superiority of causal over associative learning;
- robustness to a wrong graph;
- robustness to unmeasured or noisy renal dysfunction;
- identification of unknown assay calibration;
- transport without an unlabeled target calibration cohort;
- clinical validity in ARCADIA, MIMIC, or another real cohort.

The next sensitivity experiments should vary graph error, nuisance-variable
measurement error, target calibration size, unknown assay drift, mechanism
prevalence, nonlinearities, and missing-not-at-random mechanisms one at a time.
"""
    ),
]

nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")

