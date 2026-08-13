# TRACE-ESUS — Aim 1 Metrics Table

**For:** Amrit & Bukhari (Approach section)
**Prepared:** 10 Aug 2026
**Source of record:** retained output directories `outputs/`, `outputs_latent_endotyping/`, and `outputs_transportability/`. The archived supervised result remains provenance-only at `outputs_locked/outputs_associative_vs_scm/`; it is not preliminary evidence. Historical four-experiment identity verification is recorded in `VERIFICATION.md`.

**Every number below is already computed.** The "Result" column reports observed values, not projections. All estimates are means over **500 paired simulation repeats** with 95% CIs. Environment: Python 3.12.0 / NumPy 1.26.4 / pandas 2.2.2 / SciPy 1.14.1.

---

## Read this before using the table

**Naming.** The draft table used "Pooled" for two different models. They are not the same and must not be merged:

| Label to use | Experiment | What it is |
|---|---|---|
| **Associative latent class model** | E3 latent discovery | Unadjusted K=2 latent class model, 12 params |
| **Renal-adjusted associative LCM** | E3 latent discovery | Renal path for *every* biomarker, 14 params |
| **Biologically constrained latent SCM** | E3 latent discovery | Renal path for NT-proBNP-like marker only, 12 params — **our method** |
| **Data-generating oracle** | E3 / E4 | Evaluation ceiling, not a fitted competitor |
| **Pooled associative LCM** | E4 transport | Fit on pooled source hospitals, no target recalibration |
| **Frozen / Modular causal latent SCM** | E4 transport | Frozen = no target refit; **Modular = our method** (nuisance paths refit on unlabeled target) |

**The three retained experiments.** Historical IDs E1, E3, and E4 are preserved so citations do not silently renumber. E2 is archived.

| ID | Experiment | Directory | Estimand | Runtime |
|---|---|---|---|---|
| E1 | Known-DGP counterfactual (preliminary) | `outputs/` | Oracle/known-model query comparison — **no SCM is fitted** | 22.5 s |
| E3 | **Latent endotype discovery** | `outputs_latent_endotyping/` | Unsupervised; truth touched only after fitting — **primary Aim 1 evidence** | 524.1 s |
| E4 | Transportability | `outputs_transportability/` | Source→target generalization under nuisance + assay + missingness shift | 419.3 s |

**Renal-distortion grids differ across retained experiments** — E1 uses 0, 0.75, 1.50, 2.25 SD; E3 uses 0, 0.50, 1.00, 1.50 SD. "Strong" means 2.25 SD in E1 and 1.50 SD in E3. Do not quote a single "strong" level across experiments.

---

## Table 1 — Aim 1 primary metrics (E3, latent endotype discovery)

Reported at the strongest renal distortion (1.50 SD) unless noted. "Causal" = biologically constrained latent SCM.

| Metric | What it measures | How to measure | Threshold | Why it's needed | Result (observed) |
|---|---|---|---|---|---|
| **False endotype discovery (K=1 null)** | Whether the model invents endotypes that do not exist | Generate 500 cohorts with one homogeneous regime under renal distortion (1.50 SD); select K by BIC per model family; count K=2 selections | False K=2 rate ≤5% (Wilson upper bound <0.05) | The signature failure of subtyping papers, and the check the field routinely skips | **Causal 0/500 (0%)**, Wilson [0, 0.76%] ✓<br>Renal-adjusted 0/500 (0%) ✓<br>**Associative 500/500 (100%)** ✗<br>Median ΔBIC (K2−K1): causal **+27.29**, renal-adj +27.67, associative **−277.84**<br>K=2 convergence 100% |
| **Mechanism recovery (accuracy)** | Top-1 accuracy against planted mechanism truth | Fit unsupervised on training cohort (n=800), predict held-out test cohort (n=1000), score against simulated truth after fitting | Within 1.0 accuracy point of the data-generating oracle | The model must recover real structure, not just avoid inventing fake structure | **Causal 81.94%** [81.81, 82.06]<br>Oracle 82.70% [82.59, 82.81]<br>**Gap 0.76 pp** ✓<br>Renal-adjusted 81.64% · Associative **57.85%** |
| **Partition agreement (ARI)** | Label-invariant agreement of the recovered partition with truth | Adjusted Rand index, predicted vs true mechanism, held-out test set | Within 0.03 of oracle | Accuracy alone can be inflated by prevalence; ARI is the clustering-native check | **Causal 0.4082** [0.4050, 0.4115]<br>Oracle 0.4279 [0.4250, 0.4307]<br>**Gap 0.0197** ✓<br>Renal-adj 0.4008 · Associative **0.0377** (near chance) |
| **False atrial attribution** | Proportion of renal-impaired, true-competing patients called atrial | Restrict to renal-impaired patients whose true mechanism is competing; compute proportion assigned atrial | Within 2 pp of oracle | The high-stakes clinical error: anticoagulating a patient whose stroke was not cardioembolic | **Causal 18.51%** [17.92, 19.11]<br>Oracle 17.35% [17.07, 17.62]<br>**Gap 1.16 pp** ✓<br>Renal-adj 19.75% · Associative **75.99%** |
| **Probabilistic accuracy (Brier)** | Quality of the posterior, not just the argmax | Brier score of posterior atrial probability vs truth, held-out test set | Within 0.01 of oracle | A subtype call without a trustworthy probability cannot support a treatment decision | **Causal 0.1267** [0.1260, 0.1274]<br>Oracle 0.1216 · **Gap 0.0051** ✓<br>Renal-adj 0.1289 · Associative **0.3481** |
| **Calibration (ECE)** | Whether stated confidence matches observed frequency | 10-bin expected calibration error on posterior atrial probability | ECE ≤0.05 absolute | Miscalibration is how a "good AUC" model still harms patients | **Causal 0.0450** [0.0431, 0.0470] ✓<br>Oracle 0.0279<br>Renal-adj 0.0467 ✓ · Associative **0.3431** ✗ |
| **Prevalence bias** | Whether the model distorts the population mix of endotypes | Mean predicted atrial prevalence vs true 0.50 (equal by construction) | Within 2 pp of 0.50 | A model that shifts prevalence miscounts the trial-eligible population | **Causal 50.06%** (bias +0.06 pp) ✓<br>Oracle 50.10% · Renal-adj 50.13% ✓<br>Associative **41.25%** (bias −8.75 pp) ✗ |
| **Posterior sharpness (entropy)** | Whether the model is confidently wrong | Mean Shannon entropy of the posterior over K=2 | Report alongside ECE; entropy below oracle + poor ECE = confidently wrong | Distinguishes honest uncertainty from overconfidence | Causal **0.3733** vs Oracle 0.3824<br>Associative **0.2177** — *sharpest and worst calibrated*, i.e. confidently wrong |
| **Parameter recovery (renal path)** | Whether the estimated renal→NT-proBNP path matches the planted one | Mean bias of the fitted renal coefficient at 1.50 SD, in SD units | \|bias\| ≤0.10 SD | Recovering the mechanism parameter, not just the labels, is what makes the model interpretable | **Bias −0.0060 SD** ✓ (`validation_checks.json`) |
| **Fit integrity** | Whether results rest on degenerate or non-converged fits | EM convergence rate; minimum effective class fraction across all repeats | Convergence 100%; min class fraction ≥0.02 | A reviewer's first question about any latent-class result | Convergence **100%**; min effective class fraction **0.2519** ✓ |

### Head-to-head contrasts (E3, paired, causal minus comparator, 1.50 SD)

| Contrast | Accuracy | ARI | False atrial | Brier | ECE |
|---|---|---|---|---|---|
| vs **Associative LCM** | **+24.08 pp** [23.55, 24.62] | **+0.3706** | **−57.48 pp** | −0.2214 | −0.2981 |
| vs **Renal-adjusted LCM** | +0.30 pp [0.22, 0.37] | +0.0074 | −1.23 pp | −0.0022 | −0.0017 |

---

## Table 2 — Supporting metrics (E1, E4)

| Experiment | Metric | What it measures | How to measure | Threshold | Why it's needed | Result (observed) |
|---|---|---|---|---|---|---|
| **E4** | **Transport degradation** | Accuracy lost when the deployment environment changes | Accuracy at no-shift minus accuracy at strong shift, held-out target hospital (n=1000) | Within 1.0 accuracy point of the target oracle's own degradation | The model must survive a hospital it was not trained in | **Modular −3.92 pp** [−4.01, −3.82]<br>Oracle −3.83 pp · **Gap 0.09 pp** ✓<br>Target-calibrated −4.08 · Frozen −5.42 · Pooled −6.86 |
| **E4** | **Transport accuracy at strong shift** | Absolute performance in the shifted target | Top-1 accuracy, strong-shift target (renal prev. 0.60, 26.7% biomarker missingness) | Within 1.0 pp of target oracle | Degradation alone hides a bad starting point | **Modular 77.60%** vs Oracle 78.07% (**0.47 pp**) ✓<br>Target-calib 77.14 · Frozen 76.18 · Pooled 73.37 |
| **E4** | **Transport calibration** | Whether the posterior stays honest under shift | 10-bin ECE at each shift level | ECE ≤0.05 at strong shift | Shift breaks calibration long before it breaks accuracy | **Modular 0.0416** ✓ (0.0366→0.0416 across shift)<br>Oracle 0.0294 · Target-calib 0.0494 ✓<br>**Frozen 0.0662** ✗ · **Pooled 0.0713** ✗ |
| **E4** | **False atrial under shift** | High-stakes error in an unfamiliar hospital | Same subgroup definition as Aim 1, strong-shift target | Within 2 pp of target oracle | Clinical harm is the endpoint that transfers, not accuracy | **Modular 22.52%** vs Oracle 22.15% (**0.37 pp**) ✓<br>Target-calib 23.37 · **Frozen 36.60** · **Pooled 44.73** |
| **E4** | **Prevalence stability under shift** | Whether shift silently inflates the atrial endotype | Predicted atrial prevalence vs true 0.50, strong shift | Within 2 pp of 0.50 | Determines whether trial enrollment estimates transfer | **Modular 50.56%** ✓ · Oracle 51.01%<br>**Frozen 57.48%** ✗ · **Pooled 56.95%** ✗ |
| **E4** | **Missingness robustness** | Performance among patients with incomplete panels | Accuracy stratified: complete-case / any-missing / electrical-marker-missing, strong shift (26.7% marker missingness, 1.84% all-missing) | Complete-case within 1 pp of oracle; any-missing within 1 pp | Real ESUS workups are incomplete; this is where deployment actually fails | Complete-case **Modular 82.29%** vs Oracle 82.80% ✓ (Frozen 80.60, Pooled 77.73)<br>Any-missing **Modular 74.56%** vs Oracle 75.00% ✓ (Pooled 70.55) |
| **E4** | **Negative control — equivalence at no shift** | That the causal machinery buys nothing when there is nothing to buy | Modular causal minus target-calibrated associative accuracy at no shift; TOST-style equivalence against a ±1.0 pp margin | 95% CI entirely inside ±1.0 pp | Guards against the reviewer's "your gain is just extra flexibility" objection | **+0.299 pp** [0.247, 0.351] — **CI entirely within margin** ✓ |
| **E4** | **Negative control — exact source/target identity** | Literal identity control, distinct from the "no shift" curve point | Rerun with target = source distribution exactly | 95% CI entirely inside ±1.0 pp | The main curve's "No shift" point matches only the *reference* hospital, not the 3-hospital mixture — this is the true null | **+0.281 pp** [0.229, 0.333] ✓ |
| **E4** | **Shift ablations** | Which shift component drives the gain | Isolate kidney-only / inflammation-only / assay-only / missingness-only / combined; accuracy change vs no shift | Directional, prespecified | Locates the mechanism of the advantage instead of asserting it | Kidney-only: **Modular −0.03 pp** vs Frozen −1.72, Pooled −3.90 → **the gain is renal-path modularity**<br>Missingness-only: −4.35 pp for *every* method incl. oracle → **not a causal-model advantage**<br>Inflammation-only: ≤0.35 pp all methods<br>**Assay-only: exactly 0.000 for all five methods** |
| **E4** | Path recovery in target | Whether refit nuisance paths match the target's true paths | Bias of refit renal and inflammation paths, strong target, SD units | \|bias\| ≤0.15 SD | Modularity claim requires the refit paths to actually be right | Renal **−0.0203 SD**, inflammation **+0.0100 SD** ✓ |
| **E1** | Known-DGP kidney-aware vs kidney-blind | Ceiling-case separation with the true generative model in hand | Known-model posterior queries, 500 repeats × 1000 patients, 2.25 SD | Directional | Establishes the size of the confounding problem before any fitting | Accuracy **+7.60 pp** [7.49, 7.71] (81.22% vs 73.62%)<br>False atrial **−58.96 pp** (13.40% vs 72.36%) |
| **E1** | K=1 null (known-DGP) | False K=2 rate after subtracting the *known true* renal contribution | 500 null cohorts, diagonal Gaussian K=1/K=2, composite rule (BIC + convergence + min weight 0.10) | ≤5% | Distinct question from the E3 null — do not merge the two | **0/500 (0%)**, Wilson [0, 0.76%]; median ΔBIC **+40.93** |
| **E1** | **Counterfactual increment** | Whether the counterfactual query adds anything over the same-SCM posterior | Max \|counterfactual score − same-SCM posterior\| across all repeats | ≤ numerical tolerance (structural prediction) | Pre-empts the claim that the causal *query* is doing the work | **≈7.77×10⁻¹⁶** (floating-point noise) — **exactly zero increment**. In a symmetric K=2 model normalized disablement and sufficiency are monotone transforms of the correctly specified posterior; this is a mathematical identity, not a null result |

### Archived E2 provenance — do not use as preliminary evidence

The former supervised ceiling comparison used synthetic mechanism labels during training. It was removed because such labels cannot exist in a real ESUS cohort. Its cited values remain unchanged under `outputs_locked/outputs_associative_vs_scm/`: SCM versus biomarkers-only **+4.44 pp** [4.33, 4.55]; SCM versus kidney-adjusted **+0.005 pp** [−0.013, +0.023].

---

## Metrics in the draft skeleton that have NO data

State these as planned, not as results. Writing them as blanks in a submitted Approach invites the reviewer to ask.

| Draft row | Status | What to do |
|---|---|---|
| **Clinical-rule benchmark** | **Not implemented.** `traceesus/models/clinical_rule.py` is a reserved empty module; none of the three retained experiments implements a clinical decision rule. | Either drop it from Aim 1, or specify the rule now (e.g. AF-risk score / left-atrial size threshold) and run it. This is the most reviewer-visible gap — every clinician reviewer will ask "versus what a cardiologist already does." |
| **Uncertainty coverage** | **Not computed.** Only mean posterior entropy exists. No interval-coverage metric was implemented. | Either reframe the row as "posterior sharpness (entropy)" — data exists, Table 1 — or specify nominal-vs-empirical coverage and run it. |
| **Selective risk** | **Not computed.** No abstention/coverage curve exists in any output. | Cheap to add from saved posteriors: risk at 90/80/70% coverage after abstaining on lowest-confidence cases. Strong reviewer-facing metric for a diagnostic model. |

---

## Three things a reviewer will hit, and how to pre-empt them

**1. The honest headline is narrower than "causal beats associative."** Against the *naive* associative model the win is enormous (+24.08 accuracy points, ARI 0.038→0.408). Against the *renal-adjusted* associative model it is +0.30 pp — real and CI-excluding-zero, but small. The defensible claim is: **the biological constraint buys you, without being told, what an associative model only gets if someone already knew to adjust for kidney function — and it does so with 12 parameters instead of 14, and without inventing endotypes in the K=1 null, which the adjusted model also passes but the naive one fails 100% of the time.** The *unique* wins are the K=1 null and transport modularity, not raw accuracy.

**2. The counterfactual query provably adds nothing here.** Say it first, in the Approach, framed as a design finding: in a symmetric K=2 model the counterfactual score is a monotone transform of the correct posterior, so the structure — not the query — is what pays. Aim 2/3 with asymmetric or multi-mechanism structure is where a counterfactual increment can exist.

**3. The assay ablation is a calibration negative control, not robustness evidence.** `assay_calibrate()` exactly inverts each hospital's known offset and scale before any fitting, so the result is algebraically forced to 0.000. Describing it as "robust to assay shift" is wrong and checkable from the code. Correct wording: *unknown* assay calibration is not identified by this experiment.

---

## Provenance

| Item | Value |
|---|---|
| Verification | 46 files, 82,616 CSV rows, 933,333 cells, 423 JSON leaves, **0 discrepancies**; 8 PNGs pixel-identical; PDFs raster-identical at 180 dpi |
| Test suite | 44 passed, 1 skipped (sandbox process-semaphore limitation only) |
| Seeds | E1 `20260728`; E3 `20260728`; E4 `20260728 + 404404` |
| Repeats | 500 per level throughout; 500 additional null repeats in E1 and E3 |
| CI definition | Two-sided 95% t interval for the mean across paired repeats; empirical 2.5/97.5 repeat quantiles stored separately |
| Truth boundary | E3/E4: truth never reaches a fit function; used only post-fit in `evaluate_posterior`. E1: **no SCM is fitted** (known-DGP oracle queries) |
