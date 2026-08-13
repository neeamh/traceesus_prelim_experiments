"""Build the reader-facing notebook for the hidden-label R21 experiment."""

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path("R21_latent_endotyping_experiment.ipynb")


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
# Hidden-label R21 experiment: associative latent classes versus a latent SCM

## tl;dr

Neither model receives the simulated endotype labels during fitting. With no
renal biomarker distortion, the standard associative latent class model and
biologically constrained latent SCM recover the true hidden mechanism at nearly
the same rate (82.08% versus 82.03%).

Under a strong 1.50-SD renal effect on the NT-proBNP-like marker:

- associative latent-class accuracy falls to **57.85%**;
- causal latent-SCM accuracy remains **81.94%**;
- false atrial classification in renal-impaired patients with a true competing
  mechanism is **75.99%** versus **18.51%**;
- the renal-adjusted associative control reaches **81.64%** accuracy, which
  sharply limits the generality of the causal-superiority claim.

In a K=1 null with a real renal biomarker pathway but no endotypes, the standard
latent class model selects a spurious K=2 solution in 500/500 repeats; the
adjusted associative and causal models select K=2 in 0/500 repeats.

**Interpretation:** correct biological structure protects hidden-mechanism
recovery from a misleading biomarker pathway. This does not prove that causal
models universally beat properly adjusted associative latent-variable models.
"""
    ),
    markdown(
        r"""
## Context & Methods

### Scientific question

> Can a biologically constrained latent structural causal model recover a
> single categorical mechanism when renal dysfunction makes one biomarker look
> atrial for non-atrial reasons?

### Key assumptions

- Each patient has one true mechanism: atrial or competing.
- Mechanism labels are stored by the simulator but withheld from fitting.
- Renal status is observed and measured without error.
- Renal dysfunction is independent of the true mechanism in this first
  experiment; technically it is an alternative cause of the biomarker rather
  than a classical common-cause confounder.
- The causal graph is correctly specified, with a direct renal path only to the
  NT-proBNP-like biomarker.
- Biomarker residuals are linear, Gaussian, independent, and standardized.
- An electrical-minus-competing biomarker anchor names the fitted atrial class;
  neither NT-proBNP-like values nor truth labels are used for label orientation.
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

from r21_latent_endotyping_experiment import (
    ALL_METHODS,
    ASSOCIATIVE_ADJUSTED,
    ASSOCIATIVE_LCA,
    CAUSAL_SCM,
    ORACLE,
    BIOMARKER_NAMES,
    ExperimentConfig,
    SimulationConfig,
    FittingConfig,
    evaluate_posterior,
    fit_associative_latent_class_model,
    fit_conditional_latent_model,
    simulate_two_mechanism_cohort,
)

PROJECT_DIRECTORY = Path.cwd()
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs_latent_endotyping"
RUN_FULL_SIMULATION = False  # Set True to regenerate all 500-repeat outputs.

config = ExperimentConfig(
    master_seed=20_260_728,
    repeats_per_level=500,
    null_repeats=500,
    null_renal_effect_sd=1.50,
    workers=4,
    simulation=SimulationConfig(
        training_patients=800,
        test_patients=1_000,
        renal_dysfunction_prevalence=0.30,
        atrial_probability_if_renal_normal=0.50,
        atrial_probability_if_renal_impaired=0.50,
        atrial_path_effects_sd=(1.25, 1.00, 0.00),
        competing_path_effects_sd=(0.00, 0.00, 1.00),
        biomarker_noise_sd=(1.00, 1.00, 1.00),
        renal_effect_levels_sd=(0.00, 0.50, 1.00, 1.50),
        renal_effect_labels=("None", "Weak", "Moderate", "Strong"),
    ),
    fitting=FittingConfig(
        random_starts=4,
        maximum_em_iterations=300,
        relative_log_likelihood_tolerance=1e-6,
        variance_floor=0.05**2,
        probability_floor=1e-5,
        beta_prior_pseudocount=0.5,
        minimum_effective_class_fraction=0.02,
        calibration_bins=10,
    ),
)
config.validate()
pd.Series(asdict(config), name="value")
"""
    ),
    markdown(
        r"""
### Generated world

\[
\mathbf B_i =
\boldsymbol\lambda_{Z_i}
+(\gamma R_i,0,0)
+\boldsymbol\epsilon_i,\qquad
\boldsymbol\epsilon_i\sim N(\mathbf 0,I_3)
\]

\[
\boldsymbol\lambda_A=(1.25,1.00,0.00),\qquad
\boldsymbol\lambda_C=(0.00,0.00,1.00).
\]

The effect scale is the within-biomarker residual SD. The strong renal effect
(1.50 SD) slightly exceeds the atrial mechanism's effect on the misleading
marker (1.25 SD), while the other two markers retain mechanism-specific
information.

### Model definitions

| Model | Factorization | Renal handling | K=2 parameters |
|---|---|---|---:|
| Associative latent class | \(p(Z)p(R\mid Z)\prod_jp(B_j\mid Z)\) | renal status is a class indicator | 12 |
| Biologically constrained latent SCM | \(p(Z\mid R)\prod_jp(B_j\mid Z,R)\) | direct path restricted to NT-proBNP-like marker | 12 |
| Adjusted associative control | conditional latent class regression | renal association estimated for all markers | 14 |

The primary models therefore use the same observed inputs and the same number
of free parameters. All are fit by transparent multi-start EM implemented in
`r21_latent_endotyping_experiment.py`.
"""
    ),
    markdown(
        """
### 1. Verify that truth cannot enter either fit interface

The simulator truth is intentionally absent from every fitting signature. It is
accepted only by the evaluation function.
"""
    ),
    code(
        """
fit_interfaces = {
    "associative_fit": str(inspect.signature(fit_associative_latent_class_model)),
    "conditional_fit": str(inspect.signature(fit_conditional_latent_model)),
    "evaluation": str(inspect.signature(evaluate_posterior)),
}
fit_interfaces
"""
    ),
    markdown(
        """
## Data

The displayed preview contains observed inputs only. The hidden mechanism array
exists separately inside the simulator object and is not passed into a fit.
"""
    ),
    code(
        """
preview = simulate_two_mechanism_cohort(
    np.random.default_rng(20260728),
    patient_count=8,
    renal_effect_sd=1.50,
    config=config.simulation,
)
preview_observed = pd.DataFrame(
    preview.biomarkers,
    columns=BIOMARKER_NAMES,
).assign(renal_dysfunction=preview.renal_dysfunction)
preview_observed
"""
    ),
    markdown(
        """
### 2. Run or load the definitive simulation

The notebook loads the verified outputs by default. Set
`RUN_FULL_SIMULATION = True` above to rerun 500 repeats per level plus the
500-repeat null. If outputs are missing, the notebook regenerates them
automatically.
"""
    ),
    code(
        """
required_output = OUTPUT_DIRECTORY / "validation_checks.json"
if RUN_FULL_SIMULATION or not required_output.exists():
    command = [
        sys.executable,
        "r21_latent_endotyping_experiment.py",
        "--repeats", str(config.repeats_per_level),
        "--null-repeats", str(config.null_repeats),
        "--training-patients", str(config.simulation.training_patients),
        "--test-patients", str(config.simulation.test_patients),
        "--workers", str(config.workers),
        "--output-dir", str(OUTPUT_DIRECTORY),
    ]
    subprocess.run(command, check=True)

raw_metrics = pd.read_csv(OUTPUT_DIRECTORY / "raw_recovery_metrics.csv")
summary = pd.read_csv(OUTPUT_DIRECTORY / "recovery_summary.csv")
contrasts = pd.read_csv(OUTPUT_DIRECTORY / "paired_contrasts.csv")
diagnostics = pd.read_csv(OUTPUT_DIRECTORY / "fit_diagnostics.csv")
parameter_recovery = pd.read_csv(OUTPUT_DIRECTORY / "parameter_recovery.csv")
null_raw = pd.read_csv(OUTPUT_DIRECTORY / "k1_null_raw.csv")
null_summary = pd.read_csv(OUTPUT_DIRECTORY / "k1_null_summary.csv")
validation = json.loads(
    (OUTPUT_DIRECTORY / "validation_checks.json").read_text(encoding="utf-8")
)

{
    "recovery_metric_rows": len(raw_metrics),
    "diagnostic_rows": len(diagnostics),
    "null_rows": len(null_raw),
    "all_required_checks_pass": validation["all_required_checks_pass"],
}
"""
    ),
    markdown("## Results\n\n### 3. Primary recovery result"),
    code(
        """
strong = max(config.simulation.renal_effect_levels_sd)
strong_summary = summary[
    (summary["renal_effect_sd"] == strong)
    & summary["metric"].isin(["accuracy", "false_atrial_renal_competing"])
].copy()
strong_table = strong_summary.pivot(
    index="method",
    columns="metric",
    values=["mean", "ci95_low", "ci95_high"],
)
strong_table = strong_table.reindex(ALL_METHODS)
strong_table
"""
    ),
    code(
        """
display(Image(filename=str(OUTPUT_DIRECTORY / "figure_P1_latent_recovery.png")))
"""
    ),
    markdown(
        r"""
The curves answer the primary question. With no renal distortion, both methods
recover the same hidden mechanism at about 82%. As the renal path strengthens,
the standard associative model increasingly uses the renal-driven biomarker
pattern to define its classes. The causal model estimates and separates the
renal path, so its recovery remains near the oracle ceiling.

At strong distortion, the paired causal-minus-associative accuracy difference
is 24.08 percentage points (95% MC CI, 23.55–24.62). In renal-impaired patients
with a true competing mechanism, the causal model reduces false atrial
classification by 57.48 points (54.77–60.19).
"""
    ),
    markdown("### 4. Fairness control and K=1 null"),
    code(
        """
control_accuracy = summary[
    (summary["metric"] == "accuracy")
    & summary["method"].isin([ASSOCIATIVE_ADJUSTED, CAUSAL_SCM, ORACLE])
].pivot(index="renal_effect_sd", columns="method", values="mean")

display(control_accuracy)
display(null_summary)
display(Image(filename=str(OUTPUT_DIRECTORY / "figure_S1_controls.png")))
"""
    ),
    markdown(
        r"""
The adjusted associative model is the critical limitation. Once it explicitly
models renal associations with the biomarkers, its strong-distortion accuracy
is 81.64%, only 0.30 points below the causal model. This means the large primary
gap is caused mainly by the correct representation of the renal pathway, not by
the word “causal” or by counterfactual scoring alone.

The K=1 null is also informative. The standard latent class model cannot
represent a within-class renal-to-biomarker dependency, so it uses a second
class to absorb that dependency. Both models that explicitly adjust the
biomarkers for renal status retain K=1.
"""
    ),
    markdown("### 5. Patient-level interpretation"),
    code(
        """
example = json.loads(
    (OUTPUT_DIRECTORY / "example_patient.json").read_text(encoding="utf-8")
)
display(pd.Series(example))
display(Image(filename=str(OUTPUT_DIRECTORY / "figure_P2_example_patient.png")))
"""
    ),
    markdown(
        """
For this simulated competing-mechanism patient, the observed NT-proBNP-like
value is high even though atrial electrical evidence is nearly absent. The
associative latent class model calls the patient atrial with 0.99 probability.
The causal model attributes roughly 1.59 standardized units of the
NT-proBNP-like value to the fitted renal path; after that contribution is
removed, the competing-specific marker dominates and the atrial posterior is
0.08.
"""
    ),
    markdown("### 6. Numerical and reproducibility checks"),
    code(
        """
display(pd.Series(validation, name="result"))

diagnostic_summary = diagnostics.groupby("method").agg(
    convergence_rate=("converged", "mean"),
    median_iterations=("iterations", "median"),
    maximum_iterations=("iterations", "max"),
    minimum_effective_class_fraction=(
        "minimum_effective_class_fraction", "min"
    ),
)
display(diagnostic_summary)

renal_recovery = parameter_recovery.groupby("renal_effect_sd")[
    [
        "causal_estimated_renal_effect_nt",
        "adjusted_estimated_renal_effect_nt",
    ]
].agg(["mean", "std"])
display(renal_recovery)

assert validation["all_required_checks_pass"]
assert diagnostics["converged"].all()
assert null_raw["k2_converged"].all()
"""
    ),
    markdown(
        """
## Takeaways

1. **The narrow hypothesis is supported in this simulator.** A latent SCM with
   the correct renal path recovers a genuinely hidden categorical mechanism
   without label supervision and remains near the Bayes oracle under severe
   biomarker distortion.
2. **The generic latent class model fails for a specific reason.** It cannot
   express renal dysfunction as a within-class cause of the misleading marker,
   so it reorganizes classes around the renal-driven pattern.
3. **This is not proof that causality generally beats association.** The
   renal-adjusted associative latent model nearly matches the SCM and passes the
   null. The defensible claim is that correct biological structure protects
   endotyping and can improve finite-sample efficiency.
4. **The K=1 result is a meaningful guardrail.** The causal model did not invent
   endotypes when one renal biomarker pathway existed without mechanism
   heterogeneity.
5. **The next hard test is graph error.** Measurement error in renal status,
   renal effects on unmodeled biomarkers, correlated/heavy-tailed residuals,
   missingness, and external-environment shifts should be varied before making
   a clinical or general-method superiority claim.

Full definitions, estimands, results, and caveats are in
`R21_PRELIMINARY_METHODOLOGY.md`. Proposal-ready language is in
`R21_PROPOSAL_READY_TEXT.md`.
"""
    ),
]

nbf.validate(notebook)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
