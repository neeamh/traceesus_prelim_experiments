"""Build the executed-results notebook for the associative-versus-SCM study."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_associative_vs_scm"
NOTEBOOK_PATH = ROOT / "R21_associative_vs_scm.ipynb"


def _lookup(
    summary: pd.DataFrame,
    renal_effect: float,
    method: str,
    metric: str,
) -> pd.Series:
    row = summary[
        (summary["renal_effect_sd"] == renal_effect)
        & (summary["method"] == method)
        & (summary["metric"] == metric)
    ]
    if len(row) != 1:
        raise RuntimeError("Expected exactly one matching result.")
    return row.iloc[0]


def _tldr() -> str:
    summary_path = OUTPUT_DIR / "summary.csv"
    paired_path = OUTPUT_DIR / "paired_contrasts.csv"
    if not summary_path.exists() or not paired_path.exists():
        return "## tl;dr\n\nRun the notebook to generate the results."

    summary = pd.read_csv(summary_path)
    paired = pd.read_csv(paired_path)
    biomarker_only = "Associative logistic regression (biomarkers only)"
    adjusted = "Associative logistic regression (+ kidney status)"
    scm = "Structural causal model (counterfactual)"

    acc_unadjusted = _lookup(
        summary, 2.25, biomarker_only, "true_mechanism_accuracy"
    )
    acc_adjusted = _lookup(
        summary, 2.25, adjusted, "true_mechanism_accuracy"
    )
    acc_scm = _lookup(summary, 2.25, scm, "true_mechanism_accuracy")
    false_unadjusted = _lookup(
        summary,
        2.25,
        biomarker_only,
        "false_atrial_confounded_competing",
    )
    false_adjusted = _lookup(
        summary,
        2.25,
        adjusted,
        "false_atrial_confounded_competing",
    )
    false_scm = _lookup(
        summary,
        2.25,
        scm,
        "false_atrial_confounded_competing",
    )
    adjusted_contrast = paired[
        (paired["renal_effect_sd"] == 2.25)
        & (paired["metric"] == "true_mechanism_accuracy")
        & (paired["contrast"] == "scm_minus_kidney_adjusted")
    ].iloc[0]

    return f"""## tl;dr

This experiment answers the intended question directly.

Under strong renal biomarker distortion:

- **Biomarker-only logistic regression:** {100 * acc_unadjusted['mean']:.2f}%
  accuracy and {100 * false_unadjusted['mean']:.2f}% false atrial
  classification in the renal/competing subgroup.
- **Structural causal model:** {100 * acc_scm['mean']:.2f}% accuracy and
  {100 * false_scm['mean']:.2f}% false atrial classification.
- **Kidney-adjusted logistic regression:** {100 * acc_adjusted['mean']:.2f}%
  accuracy and {100 * false_adjusted['mean']:.2f}% false atrial
  classification.

Therefore, the SCM **does beat an ordinary biomarker-only associative
classifier**, but it **does not beat an associative classifier once kidney
status is included**. The SCM-minus-adjusted-logistic accuracy difference was
only {100 * adjusted_contrast['mean_difference']:.3f} percentage points
(95% CI {100 * adjusted_contrast['ci_low']:.3f} to
{100 * adjusted_contrast['ci_high']:.3f})."""


def build_notebook() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# R21 experiment: ordinary association versus a structural causal model\n\n"
            "A supervised synthetic train/test benchmark with known ground-truth "
            "mechanisms."
        ),
        nbf.v4.new_markdown_cell(_tldr()),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "### The models in plain English\n\n"
            "1. **Biomarker-only logistic regression** learns which biomarker "
            "patterns are associated with the atrial label. It has no causal "
            "diagram.\n"
            "2. **Kidney-adjusted logistic regression** is the same ordinary "
            "classifier with kidney status added. It is the fairness control.\n"
            "3. **Structural causal model** represents kidney → NT-proBNP-like "
            "and mechanism → biomarker paths, then switches candidate mechanisms "
            "off or on to calculate disablement and sufficiency.\n\n"
            "Every repeat uses 3,000 training patients and a new 1,000-patient "
            "test cohort. All models use the same paired data. The full methods "
            "and interpretation boundary are in `R21_PRELIMINARY_METHODOLOGY.md`."
        ),
        nbf.v4.new_code_cell(
            "from dataclasses import asdict\n"
            "from pathlib import Path\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "from r21_associative_vs_scm import (\n"
            "    ASSOCIATIVE_ADJUSTED,\n"
            "    ASSOCIATIVE_BIOMARKERS,\n"
            "    SCM_COUNTERFACTUAL,\n"
            "    ComparisonConfig,\n"
            "    run_full_comparison,\n"
            ")\n\n"
            "CONFIG = ComparisonConfig()\n"
            "OUTPUT_DIR = Path.cwd() / 'outputs_associative_vs_scm'\n"
            "pd.set_option('display.max_columns', 30)\n"
            "pd.set_option('display.width', 160)\n"
            "asdict(CONFIG)"
        ),
        nbf.v4.new_markdown_cell(
            "## Data\n\n"
            "The next cell regenerates all 2,000 paired train/test experiments "
            "and overwrites only files inside `outputs_associative_vs_scm/`."
        ),
        nbf.v4.new_code_cell(
            "artifacts = run_full_comparison(CONFIG, OUTPUT_DIR)\n"
            "print(f\"Saved results to {OUTPUT_DIR.resolve()}\")"
        ),
        nbf.v4.new_markdown_cell("### Validation checks"),
        nbf.v4.new_code_cell(
            "raw = artifacts['raw_metrics']\n"
            "summary = artifacts['summary']\n"
            "fits = artifacts['fit_diagnostics']\n\n"
            "assert len(raw) == 4 * 500 * 3 * 2\n"
            "assert raw.groupby(['strength_index', 'method', 'metric']).size().eq(500).all()\n"
            "assert raw['value'].between(0, 1).all()\n"
            "assert fits['biomarkers_only_converged'].all()\n"
            "assert fits['adjusted_converged'].all()\n"
            "assert fits['maximum_counterfactual_posterior_difference'].max() < 1e-12\n"
            "assert summary[['ci_low', 'mean', 'ci_high']].apply(\n"
            "    lambda row: row['ci_low'] <= row['mean'] <= row['ci_high'], axis=1\n"
            ").all()\n"
            "print('All row-count, range, convergence, CI, and SCM invariance checks passed.')"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### Figure P1"),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(artifacts['figure_png']), width=1150))"
        ),
        nbf.v4.new_markdown_cell("### Exact estimates"),
        nbf.v4.new_code_cell(
            "result_table = summary[[\n"
            "    'renal_effect_sd', 'method', 'metric', 'n_repeats',\n"
            "    'mean', 'ci_low', 'ci_high', 'repeat_q025', 'repeat_q975'\n"
            "]].copy()\n"
            "for column in ['mean', 'ci_low', 'ci_high', 'repeat_q025', 'repeat_q975']:\n"
            "    result_table[column] = (100 * result_table[column]).round(3)\n"
            "display(result_table)"
        ),
        nbf.v4.new_markdown_cell("### Paired model contrasts"),
        nbf.v4.new_code_cell(
            "contrasts = artifacts['paired_contrasts'].copy()\n"
            "for column in ['mean_difference', 'ci_low', 'ci_high']:\n"
            "    contrasts[column] = (100 * contrasts[column]).round(3)\n"
            "display(contrasts)"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. The SCM is more robust than a biomarker-only associative "
            "classifier when kidney dysfunction produces misleading atrial "
            "biomarker elevation.\n"
            "2. Kidney-adjusted logistic regression matches the SCM in this "
            "simple linear, correctly specified, same-distribution benchmark.\n"
            "3. The result therefore supports explicit renal adjustment, not a "
            "general claim that counterfactual classification is more accurate "
            "than association.\n"
            "4. This is supervised classification with known synthetic labels. "
            "It is not yet an unsupervised endotype-discovery experiment."
        ),
    ]
    nbf.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build_notebook())
