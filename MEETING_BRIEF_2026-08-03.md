# Monday meeting brief — 3 August 2026

**Attending:** PI (Dr. Bukhari), Dr. Farhan Khan, Neeam
**Agenda per PI:** review the outline doc + one-month plan → walk the proposal doc → align on aims →
focus on supplementary material for Aims 1 and 2, especially M1.

---

## 1. Bring these — the PI asked for four things

| PI's ask | Status | What to bring |
|---|---|---|
| Current experimental results | ✅ Ready | The four verified result families in §2 below |
| All figures/diagrams that could be Aim 1 preliminary evidence | ⚠️ Partial | 7 code-generated figures exist; the one Aim 1 actually needs (M0–M5) does not |
| Additional diagrams that strengthen the proposal | ❌ Missing | M1/M2/M3 milestone map, stable-vs-adaptable module diagram — see §5 |
| Suggestions for improving methodology + remaining gaps | ✅ Ready | §4 |

Plus, from the written action plan: *"First update to PI: Monday, August 3 — scripts reproduced,
ARCADIA variable map started, and ablation runtime estimate provided."*

---

## 2. The results, stated the way they should be stated

All 500 paired repeats, saved CSVs, validation checks pass, master seed 20260728.

### Headline 1 — Nuisance-path modeling prevents false atrial attribution
`outputs_latent_endotyping/recovery_summary.csv`, strong renal distortion (1.5 SD):

| Model | Accuracy | False atrial calls |
|---|---|---|
| Pooled associative latent-class | 57.9% | 76.0% |
| Renal-adjusted associative | 81.6% | 19.7% |
| Biology-constrained latent SCM | 81.9% | 18.5% |
| Oracle ceiling | 82.7% | 17.3% |

**Say the caveat first, before anyone asks:** the causal model beats the *pooled* model by 24.1
percentage points but beats the *correctly adjusted* model by only 0.30 pp [0.22, 0.37]. This
establishes the value of explicit nuisance-path representation. It does not establish counterfactual
superiority. The proposal already says this — keep it.

### Headline 2 — The K=1 null (the strongest thing we have)
`outputs_latent_endotyping/k1_null_summary.csv`, truth = one homogeneous regime:

- Pooled associative LCM invents a second endotype in **500 / 500** runs (100%, Wilson CI 99.2–100%)
- Renal-adjusted associative: **0 / 500**
- Biology-constrained SCM: **0 / 500**
- Median ΔBIC (K2 − K1): −277.8 associative vs +27.3 causal

This is a false-discovery argument, not an accuracy argument, and it is the hardest result to argue
with. Aim 1's success criterion already sets K=1 false discovery ≤ 5%; we're at 0%. **Lead the Aim 1
preliminary-data paragraph with this, not with the accuracy table.**

### Headline 3 — Transport needs modularization, not just causal structure
`outputs_transportability/transport_summary.csv`, strong shift:

| Model | Accuracy | False atrial | ECE |
|---|---|---|---|
| Pooled associative | 73.4% | 44.7% | 0.071 |
| Frozen causal SCM | 76.2% | 36.6% | 0.066 |
| Target-calibrated associative | 77.1% | 23.4% | 0.049 |
| **Modular causal SCM** | **77.6%** | **22.5%** | **0.042** |
| Target oracle | 78.1% | 22.2% | 0.029 |

Degradation from no-shift to strong shift: pooled −6.86 pp · frozen causal −5.42 pp · target-calibrated
−4.08 pp · **modular causal −3.92 pp** · oracle −3.83 pp. The modular causal model is the only one
sitting on the oracle's floor.

The scientifically interesting sentence: **a frozen causal model is not automatically invariant.** It
loses 36.6% of renal/competing patients to false atrial calls under shift. That's the boundary finding
and it is what motivates the whole modular architecture.

Two controls that make this credible:
- **Identical-distribution negative control passes** — with no shift, modular causal − target-calibrated
  = +0.30 pp, CI entirely inside the ±1 pp equivalence margin. The method doesn't claim wins it hasn't earned.
- **Kidney-only ablation** — modular causal loses **0.03 pp** where pooled association loses 3.90 pp
  and even the frozen causal model loses 1.71 pp. Cleanest single-component evidence in the package.

### Headline 4 — ARCADIA is exploratory, and we can prove why
Decision-grade simulation, 2,000 replications per K:

- 1,015 randomized (507/508) · ~80 recurrent strokes · **~71 events by the 2-year horizon**
- Calibration passes: type-I error 4.4–5.3%, 95% CI coverage 94.7–95.6%
- **Power to detect a real treatment × endotype interaction: 6.4–9.3%.** Best optimistic corner
  (K=2, HR 0.50, strong biomarker separation): **18.0%**. The 80% gate is unreachable.

This is not a weakness in the proposal — it is the justification for the entire outcome-blind design of
Aim 2. Frame it that way: *we simulated the confirmatory question, found the trial cannot answer it,
and redesigned the aim around what the data can support.* Reviewers reward that.

---

## 3. Where the numbers in the current draft need fixing

| Location | Draft says | Output says | Action |
|---|---|---|---|
| Research Strategy C.2.a | renal-adjusted comparator "19.8% false atrial" | 19.75% → **19.7%** | One-character fix, but the rule is every number traces to a CSV |
| Notion R01 deadline | 2026-09-01, "external deadline unknown" | Package says **2026-10-05**; PI said "Oct 7" | Updated in Notion; confirm the true date |

Everything else I checked in C.2.a and C.2.b — 57.9 / 76.0 / 81.9 / 18.5 / 81.6, 77.6 / 73.4 / 22.5 /
44.7 / 0.47 pp / 36.6 — matches the saved outputs exactly.

---

## 4. The four things worth arguing about (bring these as questions, not verdicts)

### 4.1 Aim 1 is written as an evaluation study, but M1 says "develop"
The PI's milestone language is *"M1 should focus on the development of a novel Causal AI methodology.
We need to carefully think through the technical approach and define how this method will be developed."*

The current Aim 1 reads: *"Establish the incremental value and operating boundaries of modular causal
endotyping"* — and then lists comparisons. That is a benchmarking program. Under the simplified review
framework, Factor 1 reviewers will ask what the algorithmic advance is.

**Proposed fix:** make Aim 1 develop a named object —
> a **modular transportable latent SCM**: a prespecified partition of the model into stable mechanism
> gates and adaptable nuisance / assay / missingness / prevalence modules, with (a) identifiability
> conditions for recovering the stable partition, (b) a recalibration protocol that provably touches
> only the adaptable modules using 50–150 unlabeled target patients, and (c) an abstention rule keyed
> to marker coverage and posterior-predictive failure.

Then M0–M5 becomes the *validation* of that object rather than the aim itself. This costs nothing —
we already have the machinery — but it converts a comparison paper into a methods contribution.

### 4.2 The counterfactual claim is currently unsupported by our own data
In `outputs/main_simulation_summary.csv`, "Counterfactual scoring (kidney-aware)" and "Posterior (same
kidney-aware SCM)" produce **identical means at all four confounding levels** — 0.811388, 0.811440,
0.812222, 0.812198 accuracy; 0.135219, 0.136360, 0.136241, 0.133976 false atrial. Bit-for-bit.

And in the supervised comparison, SCM − kidney-adjusted logistic = **+0.005 pp accuracy, CI
[−0.013, +0.023]** — includes zero.

Richens found sufficiency ≈ disablement for binary noisy-OR; we appear to have the continuous analogue.
Either we find the regime where they diverge — mixed/overlapping mechanisms, high informative
missingness, profiles where posterior resemblance stays high after adjustment — or the counterfactual
claim comes out of Aim 1's headline. The proposal already anticipates this as "Pattern 2" and treats it
as publishable. **Better we say it Monday than a reviewer says it in October.**

### 4.3 The assay ablation is vacuous by construction
`ablation_accuracy_changes.csv` shows exactly **0.00 pp** for all five methods under "Assay only,"
because `metadata.json` sets `assay_metadata_known: true` — calibration metadata is handed to every
method equally. The metadata file concedes it: *"Unknown calibration is not identified by this
experiment."* Either make assay calibration a genuinely estimated adaptable module or drop the panel.

### 4.4 Two contradictions in the instructions
- **IRB.** On the call: keeping the project AI/tool-centric means *"it's gonna be easier not even go
  with IRB route."* In the written message: *"Tomorrow, I will also submit the IRB application."*
  Which is it? It changes the Human Subjects attachment and the Aim 2 data-use language.
- **Priority.** The latest message says diagrams and technique refinement outrank the paper. The written
  action plan makes the Sep 1 preprint the goal with a week-by-week schedule built around it. Ask for
  an explicit ranking so the August calendar can be rebuilt against it.

---

## 5. Figures — what to show and what to admit is missing

**Show these (all code-generated, one command, seeds saved):**

| File | Shows |
|---|---|
| `outputs_latent_endotyping/figure_P1_latent_recovery.png` | Renal distortion → Figure 2A |
| `outputs_latent_endotyping/figure_S1_controls.png` | K=1 null + controls |
| `outputs_latent_endotyping/figure_P2_example_patient.png` | Per-patient posterior with uncertainty |
| `outputs_transportability/figure_T1_transportability.png` | Cross-hospital shift → Figure 2B |
| `outputs_transportability/figure_T2_transport_controls.png` | Negative control, calibration |
| `outputs_transportability/ablations/figure_T3_shift_ablations.png` | Component-wise shift ablation |
| `figure1_causal_esus_academic.html` | Framework figure — hand-built, BioRender-style |

**Admit these are missing:**
1. **M0–M5 source-of-gain figure.** This *is* Aim 1's core experiment. The transport ablation covers
   part of the ladder but not the full M0→M5 sequence on identical data and seeds. This is the single
   most important missing deliverable and it's due at the Aug 9 gate.
2. **Figure 3, ARCADIA composite** — marker-overlap UpSet, missingness heat map, Model 0/1 profiles.
   Placeholder in the Research Strategy. Cannot be built until the coverage audit runs.
3. **Figure 4, five-year work plan with hard decision gates.**
4. **M1 / M2 / M3 milestone diagram** mapped to Aims 1–3. The PI framed the whole project this way on
   Monday's agenda and no such diagram exists.
5. **Stable-vs-adaptable module diagram** — the shared DAG with mechanism gates visually separated from
   nuisance/assay/missingness/prevalence modules. This is the clearest single image for explaining what
   "modular" means and why frozen causal structure isn't enough.

Items 4 and 5 are the ones the PI is actually asking for when he says "additional diagrams that may
strengthen the proposal." They're a day's work and they carry the argument.

---

## 6. Ask Farhan these, directly

1. **What is the Aim 3 cohort, by name?** The Research Strategy contains a hard stop: *"If this cannot
   be done, the three-aim R01 should not be submitted in its current form."* Needed: custodian, design,
   sites, N now and N/year, which markers (electrical / structural / NT-proBNP / renal), imaging or
   later-AF or recurrence counts, DUA/IRB status and date, data location.
2. **StrokeNet route** — the PI mentioned it on the call as a multi-university network Farhan belongs to.
   Is it letters of cooperation or a formal DUA? Timeline?
3. **ARCADIA DUA and publication rules** — can aggregate ARCADIA outputs appear in a September preprint,
   or does that need clearance? The action plan says the preprint should not be delayed waiting for it.
4. **Which case types and patient cohorts** for the 2–3 neurologist evaluation panel? The PI was explicit
   that Farhan chooses, not us, so the selection stays unbiased.
5. **Who is the biostatistics / causal-inference investigator?** Still unnamed as Senior/Key Personnel,
   and it's a listed submission gate with long lead time.
6. **An ARCADIA or major-trial investigator** for a letter of support / cooperation on data access.

---

## 7. Your job, in one paragraph

You are the technical lead. Between now and Sep 1 you own: reproducing every existing result from one
command with seeds and an environment manifest; running the M0–M5 source-of-gain ablation and stating
plainly which component creates the gain; auditing ARCADIA marker coverage and building the frozen data
dictionary; fitting outcome-blind Models 0/1 and the bootstrap stability analysis; producing every
figure from code with a one-sentence result and one-sentence limitation caption; and writing the results
**directly into the Research Strategy Google Doc** at C.2.a–C.2.c, C.3, and C.4 — not into side notes.
Treatment assignment and recurrent-stroke outcomes stay sealed until the PI signs a score-freeze memo.
Nothing goes in the paper or the proposal unless it traces to a script, a config, a seed, and a saved
output.

## 8. Backward schedule to Oct 5

| By | Deliverable |
|---|---|
| Aug 9 | One-command rerun · M0–M5 ablation · ARCADIA coverage audit · comparator sign-off · Aim 1 reframed as development |
| Aug 16 | ARCADIA Model 0/1 outcome-blind profiles · Aim 3 cohort named · clinical panel identified |
| Aug 23 | Bootstrap and split-sample stability · multi-seed sensitivity · assay ablation re-run |
| Aug 30 | Null/MNAR/graph sensitivity · posterior-vs-counterfactual disagreement · **evidence freeze Aug 26** |
| Sep 1 | arXiv methods preprint submitted · all evidence inserted into the Research Strategy |
| Sep 9 | ML4H operational deadline |
| Sep–Oct | Biosketches, budget, letters, DMS plan, Human Subjects, mock review |
| **Oct 5** | **R01 receipt date — confirm Oct 5 vs Oct 7 this week** |
