# PROJECT CONTEXT — Causal-ESUS R01 (TRACE-ESUS)

**Owner:** Neeam Hayder (undergraduate advanced researcher, technical lead)
**PI:** Dr. Bukhari · **Clinical Co-I:** Dr. Farhan Khan (vascular neurology, ARCADIA/StrokeNet access)
**Last updated:** 2026-08-02
**Read this before answering anything about this project. Every number below traces to a named source file.**

> **CONFLICT RULE: the most recent statement wins.** Order of precedence, newest first:
> (1) the PI's Paper 1 completion checklist + the bioRxiv draft, both 2 Aug 2026;
> (2) the "19 - Neem R01 and arXiv Action Plan" Google Doc, 1 Aug;
> (3) the late-July call transcript. Where these disagree, follow the newer one and note the override.
>
> Consequences already applied: the venue is **bioRxiv**, not arXiv. Paper 1 experiments (P1–P4 below)
> are the active work — and they double as the R01's Aim 1 preliminary evidence, so there is no real
> conflict with the "diagrams first" instruction. See `PAPER1_EXECUTION_PLAN.md`.

---

## 0. One-paragraph summary

ESUS (embolic stroke of undetermined source) is a syndromic label that probably contains several
distinct biological mechanisms. Four RCTs including ARCADIA found no overall benefit of
anticoagulation. Hypothesis: *mechanistic dilution* — atrial electrical disease, structural
remodeling, prothrombotic states, renal/HF-driven biomarker elevation, and competing vascular
sources produce overlapping marker patterns but differ in causal relevance. The project builds a
**modular causal-AI framework** that separates *stable* biological mechanism paths from *adaptable*
nuisance / assay / missingness / prevalence modules, and produces a continuous, outcome-blind
atrial-thromboembolic mechanism score with uncertainty and abstention rules.

Working title: **TRACE-ESUS** — Transparent and Reproducible AI for Causal Endotyping of Atrial
Cardiopathy in ESUS.

## 1. Strategic decision already made

The project **escalated from R21 to R01** (call between Neeam and PI, late July 2026). Reasoning:
the preliminary synthetic experiments are real (hand-built, not AI-generated), ARCADIA
individual-participant data is accessible via Dr. Khan, and submitting the whole idea as an R21
would burn the concept — an R01 cannot be a minor extension of a funded R21, it must be
substantially bigger. So: go straight to R01, use the R21 material as a base, and publish the
synthetic-data work as an arXiv preprint to serve as preliminary evidence.

**The R21 is dead as a submission target. Do not plan work against it.**

## 2. Aims and milestone mapping (PI's framing)

| Milestone | Aim | Content |
|---|---|---|
| **M1** | Aim 1 | **Develop** the novel Causal AI methodology + initial validation on synthetic data. Proof-of-concept is demonstrated; next step is to finalize and *freeze* the method before broader evaluation. |
| **M2** | Aim 2 | Refine, optimize, expand, and comprehensively evaluate; derive and validate the outcome-blind ARCADIA mechanism score. |
| **M3** | Aim 3 | External validation on independent retrospective datasets (Farhan-provided; StrokeNet is the likely route). |

Current written Specific Aims (Drive doc "04 - Specific Aims"):

- **Aim 1.** Establish the incremental value and operating boundaries of modular causal endotyping.
  M0–M5 source-of-gain ablation on known-ground-truth synthetic + ARCADIA-calibrated semi-synthetic data.
- **Aim 2.** Derive and rigorously validate an outcome-blind atrial-thromboembolic mechanism score in
  ARCADIA. Staged Models 0–3, bootstrap/split-sample stability, biological controls, graph and MNAR
  sensitivity, prespecified score freeze before treatment/outcomes are opened.
- **Aim 3.** Establish transportability and independent biological validity in `[EXTERNAL COHORT —
  BROWN/STROKENET TBD]`. Zero-shot vs limited unlabeled target recalibration; abstention rules.

> **Known gap (raise with PI):** Aim 1 as currently written is an *evaluation/benchmarking* aim, but
> the PI's M1 framing calls for *methodology development*. For an R01 this matters — reviewers reward
> a named algorithmic contribution. See §7.

## 3. Deadlines

| Date | What | Confidence |
|---|---|---|
| **Mon 2026-08-03** | PI + Farhan meeting. First PI update: scripts reproduced, ARCADIA variable map started, ablation runtime estimate. | Confirmed |
| **~2026-08-03** | PI submits IRB application | Stated by PI |
| **2026-08-09** | Gate: one-command rerun + M0–M5 ablation + comparator sign-off | Internal |
| **2026-08-26** | Evidence freeze (all figures regenerated from code) | Internal |
| **2026-09-01** | arXiv methods preprint submitted; evidence inserted into R01 Research Strategy | Internal |
| **2026-09-09 / 09-10** | ML4H Proceedings operational / visible deadline | External |
| **2026-10-05** | **NIH R01 receipt date** (PA-25-301, Parent R01 Clinical Trial Not Allowed, standard new-application date). PI said "October 7th" verbally — **2-day discrepancy, confirm with sponsored programs.** | Needs confirmation |

## 4. VERIFIED PRELIMINARY RESULTS

All local, all reproducible, all with 500 paired repeats and saved CSV/JSON. Master seed 20260728
(20260729 for the supervised experiment). Python 3.12.0, numpy 1.26.4, pandas 2.2.2, scipy 1.14.1.
**Never cite a number that is not in this section or in the named CSV.**

### 4.1 Renal biomarker distortion — latent endotyping
`outputs_latent_endotyping/recovery_summary.csv` · 800 train / 1000 test patients, renal prevalence 30%,
K=2, 500 repeats per level. At **Strong distortion (renal_effect_sd = 1.5):**

| Method | Accuracy | False atrial calls (renal + true-competing patients) |
|---|---|---|
| Pooled associative latent-class model | **57.9%** | **76.0%** |
| Renal-adjusted associative LCM | **81.6%** | **19.7%** |
| Biology-constrained latent SCM | **81.9%** | **18.5%** |
| Data-generating oracle (ceiling) | 82.7% | 17.3% |

Paired contrasts (`paired_contrasts.csv`): causal SCM − pooled = **+24.08 pp** accuracy [23.55, 24.62];
causal SCM − renal-adjusted = **+0.30 pp** [0.22, 0.37]. False atrial: causal − pooled = −57.5 pp;
causal − renal-adjusted = −1.23 pp [−1.94, −0.53].

**Honest reading: the gain is from explicit nuisance-path modeling, not from causal structure per se.
A correctly renal-adjusted associative model does nearly as well. Say this out loud in the proposal.**

### 4.2 K=1 null — the strongest single result
`outputs_latent_endotyping/k1_null_summary.csv` · 500 repeats, renal_effect_sd = 1.5, truth = one
homogeneous regime.

| Method | False K=2 selections | Rate |
|---|---|---|
| Pooled associative LCM | **500 / 500** | **100%** (Wilson CI 99.2–100%) |
| Renal-adjusted associative LCM | 0 / 500 | 0% |
| Biology-constrained latent SCM | 0 / 500 | 0% |

The pooled associative model **invents endotypes 100% of the time when none exist**. Median ΔBIC
(K2−K1) = −277.8 for the associative model vs +27.3 for the causal SCM. This is a false-discovery
argument, not an accuracy argument, and it is the most reviewer-proof result in the package.

### 4.3 Cross-hospital transportability
`outputs_transportability/transport_summary.csv` · 3 source hospitals × 600 patients, 150 unlabeled
target calibration patients, 1000 target test, 500 repeats. At **Strong shift** (renal prevalence
0.60, renal effect 1.8 SD, inflammation 0.55, assay offsets up to 0.45, missingness base 0.28–0.35):

| Method | Accuracy | False atrial | ECE |
|---|---|---|---|
| Pooled associative LCM | **73.4%** | **44.7%** | 0.0713 |
| Frozen causal SCM (no recalibration) | 76.2% | **36.6%** | 0.0662 |
| Target-calibrated associative | 77.1% | 23.4% | 0.0494 |
| **Modular causal SCM** | **77.6%** | **22.5%** | **0.0416** |
| Target oracle (ceiling) | 78.1% | 22.2% | 0.0294 |

Modular causal − pooled = **+4.23 pp** [4.11, 4.36]. Modular causal − target-calibrated =
**+0.47 pp** [0.38, 0.55]. Modular − frozen causal = +1.42 pp.

Degradation strong-vs-no-shift (`transport_degradation.csv`): pooled −6.86 pp · frozen causal −5.42 pp ·
target-calibrated −4.08 pp · **modular causal −3.92 pp** · oracle −3.83 pp. The modular causal model is
the only method that lands essentially on the oracle's degradation floor.

**Negative control passes** (`negative_control.json`): with no shift, modular causal − target-calibrated
= +0.30 pp, CI entirely inside the prespecified ±1 pp equivalence margin. The model does not claim a
win where none should exist.

### 4.4 Shift-component ablation
`outputs_transportability/ablations/ablation_accuracy_changes.csv` · accuracy loss vs no-shift:

| Shift component | Pooled | Frozen causal | Target-calibrated | **Modular causal** | Oracle |
|---|---|---|---|---|---|
| Kidney only | −3.90 pp | −1.71 pp | −0.13 pp | **−0.03 pp** | +0.05 pp |
| Inflammation only | −0.35 pp | +0.03 pp | −0.06 pp | +0.01 pp | +0.03 pp |
| Assay only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Missingness only | −4.26 pp | −4.32 pp | −4.34 pp | −4.35 pp | −4.37 pp |
| Combined strong | −6.86 pp | −5.42 pp | −4.08 pp | −3.92 pp | −3.83 pp |

Two things to know:
1. **Kidney-only is the clean win** — the modular causal model is effectively immune (−0.03 pp) where
   pooled association loses 3.90 pp. This is the sharpest single-component evidence for modularization.
2. **Assay-only is exactly 0.00 for every method by construction** — `metadata.json` sets
   `assay_metadata_known: true`, so calibration metadata is handed to all methods equally. A reviewer
   will spot this. Either drop the assay panel or re-run with unknown assay calibration.
3. **Missingness hurts everyone equally, including the oracle** — that is irreducible information loss,
   not a modeling failure. Frame it that way; do not present it as a limitation of the method.

### 4.5 Supervised SCM vs logistic regression — the honest null
`outputs_associative_vs_scm/` · seed 20260729, 3000 train / 1000 test, 500 repeats.
**`metadata.json` explicitly states all three models use true labels in training — this is supervised
classification, NOT endotype discovery.** At strong confounding (2.25 SD):

| Method | Accuracy | False atrial |
|---|---|---|
| Logistic, biomarkers only | 76.7% | 37.0% |
| Logistic + kidney status | 81.2% | 13.6% |
| SCM counterfactual | 81.2% | 13.5% |

SCM − kidney-adjusted logistic = **+0.005 pp accuracy, CI [−0.013, +0.023] — includes zero.**
Once you adjust correctly, the counterfactual query adds nothing here.

### 4.6 Counterfactual == posterior — and the collapse is ANALYTIC, not empirical
`outputs/main_simulation_summary.csv` · "Counterfactual scoring (kidney-aware)" and "Posterior (same
kidney-aware SCM)" produce **byte-identical means at every confounding level** (0.8114 accuracy /
0.1364 false atrial at 0.75 SD; identical at 0, 1.5, 2.25 SD). Kidney-blind posterior collapses to
73.6% / 72.4% at 2.25 SD.

**Do not report this as an empirical null.** `r21_preliminary_experiment.py`'s own docstring states:
*"In this deliberately symmetric K=2 toy model, normalized sufficiency and disablement are monotone
transformations of the correctly specified posterior."* Monotone transforms preserve ordering, so
identical top-1 accuracy is guaranteed by construction. Re-running this design answers nothing.

Worse, the framework's diagnostic payoff is **redundancy** (high sufficiency + low disablement), which
requires an alternative path that can preserve the evidence after the mechanism is disabled. With
mutually exclusive symmetric mechanisms, no such path exists — **the current generator structurally
cannot produce the phenomenon the framework claims to detect.**

The fix (see `PAPER1_EXECUTION_PLAN.md` §0): state the collapse as a **proposition** — the
continuous-biomarker analogue of Richens' binary noisy-OR result — then violate its premises one at a
time and measure divergence. Highest-value violation is **redundant nuisance paths** (renal / HF /
inflammation acting on the same markers as the atrial gate), because that is literally the NT-proBNP
problem. Others: co-active mechanisms, asymmetric signatures, mechanism-dependent missingness,
heteroscedastic noise. This converts an unsupported claim into a methodological contribution and
simultaneously gives the R01's Aim 1 the "development" object the PI's M1 asks for.

### 4.7 ARCADIA feasibility simulation — the information-limits result
Decision-grade notebook, executed 19 Jul 2026; 2,000 replications per K (1,000 for the sensitivity grid).

- ARCADIA: **1,015 randomized** (507 / 508). **~80 recurrent strokes total** (40 per arm).
  **~71 events by the 2-year analysis horizon** (median across simulated trials).
- **Calibration passes:** type-I error 5.0% / 5.3% / 4.4% / 4.8% for K=2–5; 95% CI coverage
  95.1% / 94.7% / 95.6% / 95.2%.
- **Power fails:** detecting a true treatment × endotype interaction (30% responder prevalence,
  HR 0.65, biomarker separation 1.25) = **9.3% (K=2) · 6.4% (K=3) · 8.1% (K=4) · 7.1% (K=5)**.
- **Best sensitivity corner** (K=2, HR 0.50, strong biomarker information 2.00) = **18.0%**. Nothing
  tested comes near the 80% confirmatory gate.

**Be wary of these numbers in exactly these ways:**
- The simulation is **optimistic by construction** — the posterior uses the *correct* simulated
  biomarker model. Real endotype learning is harder, so 9.3% is a ceiling, not an estimate.
- It audits **only rung 4** (held-out RCT treatment heterogeneity). It does not learn the DAG, recover
  endotypes, or validate sufficiency/disablement. Do not let it be cited as validation of the method.
- Responder prevalence is held fixed as K grows, which deliberately avoids an artificial penalty — good
  design, but it means the K=3–5 numbers are not "what happens if you split the cohort five ways."
- The sensitivity grid used 1,000 reps, the primary 2,000. Report both.

**Consequence, already accepted by the team: ARCADIA is reclassified from confirmatory to exploratory.**
A negative ARCADIA subgroup result would tell us almost nothing. This is why Aim 2 is outcome-blind and
why the post-freeze interaction is a secondary, exploratory product — not a success criterion.

### 4.8 MIMIC-IV coverage — why MIMIC was demoted
From ~2,938 stroke admissions: any NT-proBNP **267** · NT-proBNP + CRP **40** · CRP + D-dimer **58** ·
NT-proBNP + D-dimer **17** · all three complete-case **3**. Missing-not-at-random by construction —
each marker is ordered for a different clinical reason on a different patient. MIMIC is a
supporting/replication source only; it cannot anchor discovery. (Source: project_spec §2.1.)

## 4.9 Paper 1 — bioRxiv draft status (as of 2026-08-02)

`Causal_ESUS_bioRxiv_Paper1_Overleaf.pdf` — 20 pp, `\workdrafttrue`, NOT ready for submission.
Authors: Hayder, Khan, Bukhari. Sections 1–5, 7–9 are written; §6.3–6.6 are `[RESULT REQUIRED]`.

**PI's four priorities** (full sequencing, effort estimates, and dates in `PAPER1_EXECUTION_PLAN.md`):

| | Priority | Coverage today |
|---|---|---|
| P1 | Lock both existing experiments, clean-env rerun, tidy result files, verify every manuscript value, vector figures | ~85% — missing venv/lockfile/one-command runner and tidy consolidation |
| P2 | Decisive same-fitted-SCM posterior vs counterfactual, renal/HF/inflammatory families, uncomplicated vs confounded subgroups | 0% usable — current design is analytically degenerate (§4.6) |
| P3 | Source-of-gain ablation, all 8 rows of `tables/ablation_placeholder.tex`, paired cohorts and seeds | ~55% — rows 1–5 exist but not on common seeds; rows 6–8 are new code |
| P4 | K=1 null, K=2–4, overlap, MCAR/MAR/MNAR, omitted renal path, alternative DAGs, coverage, abstention | ~25% — K=1 null is done; the rest is unimplemented |

**Manuscript errors found in the audit:**
- Renal-adjusted false atrial reported as **19.8%** in the abstract, §6.1, and the Figure 2 caption.
  Locked output is **19.75% → 19.7%**. Fix all three.
- §7.1 still says "the R21's methodological and real-data objectives." R21 is dead — change to R01.
- §6.5 says `[RESULT REQUIRED]` for the K=1 null, but that result already exists (§4.2). Fill it in.
- Everything else — 81.9 / 57.9 / 76.0 / 18.5 / 81.6 / 77.6 / 73.4 / 4.23 pp / 0.47 pp / 44.7 / 36.6 /
  23.4 / 22.5 / 22.2 and all sample sizes — verified correct against the locked CSVs.

**Not implemented anywhere in the codebase, despite being claimed in the paper and the R01:**
abstention rules, overlapping/co-active mechanisms, K≠2, MCAR/MAR/MNAR comparison, alternative-DAG
sensitivity, interval coverage. Abstention appears in the paper's §4.4, in Figure 1, and in R01 Aims 1
and 3 — it is the largest silent gap in the package.

**Effort:** the full checklist is ~106–134 h. At the 8 h/week currently budgeted that is 3–4 months.
Sep 1 is not achievable for Tier A; the plan document proposes three scope tiers and recommends Tier B.

## 5. Hard rules — non-negotiable

1. **No invented numbers.** Every figure and number must trace to a script, config, seed, and saved
   output. If it isn't in a CSV or JSON in this repo, it does not go in the paper or the proposal.
2. **ARCADIA treatment assignment and recurrent-stroke outcomes stay sealed** until the PI signs a
   written score-freeze memo.
3. **No row-level ARCADIA data on GitHub.** Code, synthetic examples, schemas, and DUA-permitted
   aggregates only.
4. **Dr. Khan approves** the clinical DAG, biomarker definitions, positive/negative controls, and the
   interpretation of ARCADIA profiles — before those are used.
5. **Write directly into the Google Docs**, not into side notes. Analysis text goes into
   "05 - Research Strategy - Causal-ESUS R01" at C.2.a–C.2.c, C.3 (Aim 1), C.4 (Aim 2).
6. **Preliminary results are preliminary.** The proposal must say the methodology will be further
   refined, enhanced, and rigorously reassessed. Never imply the software or final method is done.

## 6. Submission gates that are still OPEN

- **Aim 3 external cohort is unnamed.** The Research Strategy says explicitly: if this cannot be
  replaced with a confirmed dataset + counts + access timeline, *the three-aim R01 should not be
  submitted in its current form.* This is the single biggest submission risk.
- ARCADIA DUA + institutional access pathway documented in writing.
- IRB determination for the PI institution and any Brown/StrokeNet performance site.
- Real ARCADIA marker-overlap, Model 0/1, and bootstrap-stability results (currently placeholders).
- M0–M5 component ablation completed (currently only partially covered by the transport ablation).
- Biostatistics/causal-inference investigator named as Senior/Key Personnel with effort and role.
- 2–3 neurologists recruited for independent clinical evaluation; Farhan selects case types and
  cohorts so case selection is not done by the modelling team.
- Program officer confirms NINDS fit and clinical-trial classification.
- Letter of support / DUA from an ARCADIA or major stroke-trial investigator.

## 7. Open critiques to raise with the PI (do not silently paper over these)

1. **Aim 1 is written as evaluation, not development.** The PI's M1 says "develop the novel Causal AI
   methodology." The current Aim 1 text benchmarks existing components. An R01 needs a named
   methodological object — e.g. *modular transportable latent SCM with prespecified stable/adaptable
   partition, identifiability conditions, and an abstention rule*. Rewrite Aim 1 around that.
2. **The counterfactual claim is currently unsupported** (§4.6). Either find the divergence regime or
   demote it from headline to exploratory.
3. **Causal structure ≠ the source of gain.** In two of three experiments a correctly adjusted or
   target-calibrated associative model matches the causal model. The defensible claim is *modularization
   + explicit nuisance paths + limited target recalibration*, plus the K=1 false-discovery result.
4. **The assay ablation is vacuous** (§4.4) because assay metadata is assumed known.
5. **IRB contradiction.** The PI said on the call that keeping the project AI/tool-centric avoids the
   IRB route, then later said he would submit the IRB application. Clarify which is true — it changes
   the Human Subjects attachment.
6. **Priority conflict.** The PI's most recent message says diagrams + technique refinement outrank the
   paper ("we may not have another major proposal submission opportunity"), but the written action plan
   makes the Sep 1 preprint the goal. Get an explicit ranking.
7. **Deadline discrepancy:** Oct 5 (written package, standard NIH new-R01 date) vs Oct 7 (PI verbal).

## 8. Where everything lives

**Google Drive — "Causal-ESUS R01 Working Package"** (`12jtVjZ3oISjIHJ7T6U1-Xv4R_lOYDW0t`)
- `00 - README FIRST` — package orientation, NIH assumptions, submission gates
- `01 - Master Working Draft` — abstract, narrative, aims, submission dashboard
- `04 - Specific Aims`
- `05 - Research Strategy` — the 12-page attachment; **this is where analysis text goes**
- `19 - Neem R01 and arXiv Action Plan` — the Aug 1 – Sep 1 week-by-week plan
- `R21-old proposal and experiments/` — `Causal_ESUS_R21_Revised.docx/.pdf`, `Sample figure`, `time.png`,
  and `Expermients/` (the two Colab notebooks, same as local)

**Google Drive — "Causal Discovery of Latent Atrial Cardiopathy Endotypes in Cryptogenic Stroke"**
(`11LuP2PxrL0-k3IZI2UGnWzwXkdk2nc4s`)
- `experiments/arcadia_pipeline_decision_grade.ipynb` — the ARCADIA feasibility simulation (§4.7)
- `experiments/ARCADIA Simulation Team Brief` (slides, 19 Jul 2026)
- `project_spec (1).pdf` — full technical design, MIMIC finding, five-tier architecture, Richens framing
- `ESUS_R01_Readiness_Analysis.md.pdf` — the three-gates analysis (access / coverage / events)
- `arcadia (1).dta` — ARCADIA data file

**Local repo** (this folder) — experiment scripts, notebooks, and all `outputs*/` result directories.

**Notion** — Research Command Center → Research Projects → "R01 Grant — Causal Endotyping"
(Track = R01, Portfolio = Grant), plus "Causal ESUS — ML4H Methods Paper".

## 9. Figures — what exists and what is missing

**Exists (code-generated, reproducible):**
- `outputs_latent_endotyping/figure_P1_latent_recovery`, `figure_P2_example_patient`, `figure_S1_controls`
- `outputs_transportability/figure_T1_transportability`, `figure_T2_transport_controls`
- `outputs_transportability/ablations/figure_T3_shift_ablations`
- `outputs_associative_vs_scm/figure_P1_associative_vs_scm`, `outputs/figure_P1`
- `figure1_causal_esus_academic.html`, `figure1_causal_esus_symbolic_draft2.html` (hand-built framework
  figure, BioRender-style — **not AI-generated**, and the PI knows this and values it)

**Missing / placeholder — these are the gaps:**
- **M0–M5 source-of-gain ablation figure** — this *is* Aim 1's core experiment and does not exist yet
- **Figure 3, ARCADIA composite** — marker-overlap UpSet + missingness heat map + Model 0/1 profiles
- **Figure 4, five-year work plan with decision gates** (`time.png` may be a draft)
- **M1 / M2 / M3 milestone diagram** mapped to Aims 1/2/3 — the PI framed the project this way but no
  diagram exists
- **Stable-vs-adaptable module diagram** — the single clearest way to show what "modular" means

## 10. Tone and framing rules for anything written for this project

- Lead with the boundary, not the win. "Modular causal structure + limited recalibration reduces false
  mechanism attribution under confounding and shift" — not "causal AI beats machine learning."
- Always report the adjusted/calibrated comparator alongside the pooled one.
- Preliminary means preliminary. Evolving methodology, further validation with additional datasets.
- No treatment-guidance claims. Ever. Not without independent randomized evidence and adequate events.
- Null and boundary results are deliverables, not failures — the proposal is explicitly designed so
  that Pattern 2 (posterior == counterfactual) and Pattern 3 (adjusted association == causal) are
  publishable outcomes.
