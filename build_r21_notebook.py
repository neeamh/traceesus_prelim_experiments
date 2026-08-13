"""Build the reader-facing R21 notebook with nbformat.

Run after the full simulation so the top-of-notebook summary reflects the saved
results. The notebook itself reruns the entire experiment when executed.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
NOTEBOOK_PATH = PROJECT_DIR / "R21_counterfactual_preliminary_experiment.ipynb"


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _build_tldr() -> str:
    summary_path = OUTPUT_DIR / "main_simulation_summary.csv"
    null_path = OUTPUT_DIR / "k1_null_summary.csv"
    if not summary_path.exists() or not null_path.exists():
        return (
            "## tl;dr\n\n"
            "Execute the notebook top-to-bottom to generate the 500-repeat "
            "results, Figure P1, and K=1 null check."
        )

    summary = pd.read_csv(summary_path)
    null_summary = pd.read_csv(null_path).iloc[0]

    def lookup(strength: float, method: str, metric: str) -> pd.Series:
        selected = summary[
            (summary["renal_effect_sd"] == strength)
            & (summary["method"] == method)
            & (summary["metric"] == metric)
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"Expected one summary row for {strength=}, {method=}, {metric=}."
            )
        return selected.iloc[0]

    counterfactual = "Counterfactual scoring (kidney-aware)"
    blind = "Posterior matching (kidney-blind)"
    no_conf_cf = lookup(0.0, counterfactual, "true_mechanism_accuracy")
    no_conf_blind = lookup(0.0, blind, "true_mechanism_accuracy")
    strong_cf = lookup(2.25, counterfactual, "true_mechanism_accuracy")
    strong_blind = lookup(2.25, blind, "true_mechanism_accuracy")
    strong_false_cf = lookup(
        2.25, counterfactual, "false_atrial_confounded_competing"
    )
    strong_false_blind = lookup(
        2.25, blind, "false_atrial_confounded_competing"
    )

    return f"""## tl;dr

- With no renal distortion, true-mechanism accuracy was
  **{_percent(no_conf_blind['mean'])}** for kidney-blind posterior matching and
  **{_percent(no_conf_cf['mean'])}** for kidney-aware counterfactual scoring.
- Under the strong 2.25-SD renal effect, accuracy was
  **{_percent(strong_blind['mean'])}** versus
  **{_percent(strong_cf['mean'])}**, respectively.
- In renal-impaired patients whose true mechanism was competing, strong
  distortion produced false atrial classification of
  **{_percent(strong_false_blind['mean'])}** versus
  **{_percent(strong_false_cf['mean'])}**.
- In the homogeneous K=1 null, the corrected selection pipeline chose a
  spurious K=2 model in **{int(null_summary['false_k2_selections'])} of
  {int(null_summary['repeats'])}** repeats
  (Wilson 95% CI {_percent(null_summary['wilson_ci_low'])} to
  {_percent(null_summary['wilson_ci_high'])}).

The gain over kidney-blind matching is evidence for representing the renal
biomarker path. It is **not** evidence that counterfactual syntax outperforms a
correctly specified posterior: in this symmetric toy SCM, the counterfactual
ranking and same-SCM posterior ranking are intentionally equivalent."""


def build_notebook() -> Path:
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

    cells = [
        nbf.v4.new_markdown_cell(
            "# R21 preliminary experiment: counterfactual scoring under renal "
            "biomarker distortion\n\n"
            "A restart-clean, fully scripted synthetic experiment for the "
            "professor meeting. The numerical result is generated from stored "
            "ground truth; no ARCADIA patient data are used."
        ),
        nbf.v4.new_markdown_cell(_build_tldr()),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "### Key assumptions\n\n"
            "- Each patient has one true categorical latent mechanism: atrial "
            "or competing. Posterior probabilities encode uncertainty, not "
            "biological mixture membership.\n"
            "- Renal dysfunction influences mechanism prevalence and directly "
            "raises the NT-proBNP-like biomarker.\n"
            "- The primary baseline deliberately omits the direct renal path. "
            "The counterfactual scorer and same-SCM posterior include it.\n"
            "- Parameters are known (oracle simulation). This isolates scoring "
            "behavior before parameter learning, missingness, and graph "
            "misspecification are introduced.\n\n"
            "The counterfactual calculation enumerates both latent branches, "
            "abducts a branch-specific exogenous residual, reuses that residual "
            "under intervention, and posterior-averages normalized disablement "
            "and sufficiency. See `R21_PRELIMINARY_METHODOLOGY.md` for the full "
            "estimand and interpretation boundary."
        ),
        nbf.v4.new_code_cell(
            "from dataclasses import asdict\n"
            "from pathlib import Path\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "from r21_preliminary_experiment import (\n"
            "    ExperimentConfig,\n"
            "    METHOD_COUNTERFACTUAL,\n"
            "    METHOD_POSTERIOR_BLIND,\n"
            "    METHOD_POSTERIOR_FULL,\n"
            "    kidney_aware_posterior,\n"
            "    posterior_integrated_counterfactual_scores,\n"
            "    run_full_experiment,\n"
            "    simulate_two_mechanism_study,\n"
            ")\n\n"
            "PROJECT_DIR = Path.cwd()\n"
            "OUTPUT_DIR = PROJECT_DIR / 'outputs'\n"
            "CONFIG = ExperimentConfig()\n"
            "pd.set_option('display.max_columns', 30)\n"
            "pd.set_option('display.width', 140)\n"
            "asdict(CONFIG)"
        ),
        nbf.v4.new_markdown_cell(
            "## Data\n\n"
            "The cell below runs 500 paired simulated studies at each of four "
            "renal-effect levels and 500 independent K=1 null studies. It "
            "overwrites only files inside `outputs/`."
        ),
        nbf.v4.new_code_cell(
            "artifacts = run_full_experiment(CONFIG, OUTPUT_DIR)\n"
            "print(f\"Saved outputs to: {OUTPUT_DIR.resolve()}\")"
        ),
        nbf.v4.new_markdown_cell(
            "### Reasonableness and invariance checks\n\n"
            "These checks catch the original notebook's failure modes: missing "
            "repeats, probability errors, discarded score components, and "
            "unsupported claims that the causal query beats the posterior that "
            "generated it."
        ),
        nbf.v4.new_code_cell(
            "summary = artifacts['summary']\n"
            "raw = artifacts['raw_metrics']\n\n"
            "assert raw.groupby(['strength_index', 'method', 'metric']).size().eq(CONFIG.repeats_per_level).all()\n"
            "assert raw['value'].between(0, 1).all()\n"
            "assert summary[['ci_low', 'mean', 'ci_high']].apply(\n"
            "    lambda row: row['ci_low'] <= row['mean'] <= row['ci_high'], axis=1\n"
            ").all()\n\n"
            "# In the symmetric K=2 toy SCM, the normalized counterfactual rank\n"
            "# is deliberately equivalent to the same-SCM posterior rank.\n"
            "check_rng = np.random.default_rng(CONFIG.seed + 99)\n"
            "check_data = simulate_two_mechanism_study(CONFIG, 1.5, check_rng)\n"
            "check_cf = posterior_integrated_counterfactual_scores(\n"
            "    check_data['biomarkers'], check_data['renal'], 1.5, CONFIG\n"
            ")\n"
            "check_posterior = kidney_aware_posterior(\n"
            "    check_data['biomarkers'], check_data['renal'], 1.5, CONFIG\n"
            ")\n"
            "np.testing.assert_allclose(check_cf['combined'], check_posterior, atol=1e-12)\n"
            "print('All structural, range, CI, and same-SCM invariance checks passed.')"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### Figure P1"),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(artifacts['figure_png']), width=1150))"
        ),
        nbf.v4.new_markdown_cell(
            "### Exact main-effect estimates\n\n"
            "The shaded figure intervals are 95% Monte Carlo confidence "
            "intervals for the expected rate. `repeat_q025` and `repeat_q975` "
            "in the saved CSV describe the wider distribution of study-level "
            "estimates."
        ),
        nbf.v4.new_code_cell(
            "main_table = summary[\n"
            "    summary['method'].isin([METHOD_POSTERIOR_BLIND, METHOD_COUNTERFACTUAL])\n"
            "][[\n"
            "    'renal_effect_sd', 'method', 'metric', 'n_repeats',\n"
            "    'mean', 'ci_low', 'ci_high', 'repeat_q025', 'repeat_q975',\n"
            "    'mean_denominator'\n"
            "]].copy()\n"
            "for column in ['mean', 'ci_low', 'ci_high', 'repeat_q025', 'repeat_q975']:\n"
            "    main_table[column] = (100 * main_table[column]).round(2)\n"
            "display(main_table)"
        ),
        nbf.v4.new_markdown_cell(
            "### Fairness diagnostic\n\n"
            "Top-1 accuracy from the same renal-aware posterior must be checked "
            "against the counterfactual ranking. If the counterfactual method "
            "appeared systematically superior here, the implementation or "
            "comparison would require investigation."
        ),
        nbf.v4.new_code_cell(
            "diagnostic = summary[\n"
            "    (summary['metric'] == 'true_mechanism_accuracy')\n"
            "    & summary['method'].isin([METHOD_COUNTERFACTUAL, METHOD_POSTERIOR_FULL])\n"
            "][['renal_effect_sd', 'method', 'mean', 'ci_low', 'ci_high']].copy()\n"
            "for column in ['mean', 'ci_low', 'ci_high']:\n"
            "    diagnostic[column] = (100 * diagnostic[column]).round(3)\n"
            "display(diagnostic)"
        ),
        nbf.v4.new_markdown_cell("### K = 1 null"),
        nbf.v4.new_code_cell(
            "null_table = artifacts['null_summary'].copy()\n"
            "for column in ['false_k2_rate', 'wilson_ci_low', 'wilson_ci_high', 'k2_convergence_rate']:\n"
            "    null_table[column] = (100 * null_table[column]).round(2)\n"
            "display(null_table)"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. The clean preliminary result is a stress test of **renal-aware "
            "causal specification**, not a proof that counterfactual querying "
            "beats an exact posterior.\n"
            "2. Figure P1 directly reports both requested outcomes: overall "
            "true-mechanism ranking and false atrial classification in the "
            "confounded competing-mechanism subgroup.\n"
            "3. The K=1 result is protected by explicit model selection. A "
            "fixed-K run would be circular.\n"
            "4. The next experiment should introduce parameter learning and "
            "prespecified graph misspecification. Missingness and the full "
            "ARCADIA biology should remain out of this first figure."
        ),
    ]
    notebook["cells"] = cells
    nbf.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build_notebook())
