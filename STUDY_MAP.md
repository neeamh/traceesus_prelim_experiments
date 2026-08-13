# Study map — what to learn, in what order, to what depth
**For Neeam. Written 2 Aug 2026.** This is a syllabus, not a tutorial. Every item says *where to look*
and *what question you should be able to answer afterwards.* Nothing here is explained for you on purpose.

---

## The good news, stated plainly

You did not generate 2,100 lines of incomprehensible code. You generated about **150 lines that matter**,
wrapped in 2,000 lines of plumbing (EM loops, CSV writers, plotting, validation). The 150 lines are
unusually clean and the docstrings are unusually explicit about *why* each choice was made.

There are also two documents in your own repo that are written explanations of exactly this:

- `R21_PRELIMINARY_METHODOLOGY.md` (10 KB)
- `R21_TRANSPORTABILITY_METHODOLOGY.md` (13 KB)

Plain language, then the math, then the design rationale. **Read these before anything else.** If you
only do one thing tonight, do this. They are the syllabus; everything below is a map for going deeper.

---

## Depth tiers

- **[M] MASTER** — reconstruct from a blank page. Explain to a five-year-old *and* to a methodologist.
- **[F] FLUENT** — explain confidently, don't need to derive.
- **[R] RECOGNIZE** — know what it is, where it sits, why it's there.
- **[D] DEFER** — know it exists; "let me check the code and get back to you" is a correct answer.

---

# TIER M — master these three. They are 90% of what you'll be asked.

## M1. The data generator — how a fake patient gets made
**Where:** `r21_latent_endotyping_experiment.py`, `simulate_two_mechanism_cohort`, **lines 232–274**.
Config at **lines 78–101**. It's 30 lines. Read every one.

**Extract:**
- The order things are drawn in: renal status → mechanism → biomarkers. Why that order and not another?
- `atrial_path_effects_sd = (1.25, 1.00, 0.00)` and `competing_path_effects_sd = (0.00, 0.00, 1.00)`.
  Three biomarkers. Which mechanism touches which markers, and which marker is untouched by each?
- `renal_effect` is a zero vector with **one** position filled — which one, and why only one?
- "SD" in the parameter names — effects are in units of residual standard deviation. What does
  "1.25 SD" mean physically?
- `atrial_probability_if_renal_impaired = 0.50` equals `atrial_probability_if_renal_normal = 0.50`.
  **This is deliberate.** Renal status does *not* change your mechanism. What does that make renal —
  a confounder, or something else? Your own `R21_PRELIMINARY_METHODOLOGY.md` addresses this directly
  and the distinction is subtle. Own it.
- The noise line: `rng.normal(0, noise_sd, ...)`. Why is Bayes error deliberately non-zero?

**Self-test:** draw the DAG on paper from memory. Three biomarker nodes, one mechanism node, one renal
node, arrows with coefficients. Then say out loud which arrow is the "nuisance path."

**Then read** `simulate_one_mechanism_null_cohort`, **lines 276–302** — the K=1 generator. What's
different? (That's the null experiment your strongest result comes from.)

---

## M2. What actually separates the three models
**This is the crux of the entire paper.** If you know only one thing cold, make it this.

**Where:**
- `AssociativeLatentClassFit`, **lines 196–210** — read the docstring line
- `ConditionalLatentFit`, **lines 212–226** — read the docstring line
- `fit_conditional_latent_model`, **line 645** — look specifically at the `renal_path_mask` argument

**The three factorizations** (they're written in the docstrings, one line each):

| Model | Factorization | Where renal sits |
|---|---|---|
| Associative LCM | `p(Z) p(R∣Z) p(B∣Z)` | ? |
| Renal-adjusted / causal | `p(Z∣R) p(B∣Z,R)` | ? |

**Extract:**
- In `p(R∣Z)`, renal is treated as a consequence of the mechanism. In `p(Z∣R)`, it's a cause. Which
  matches the generator in M1? What must the wrong one do when a competing-mechanism patient has an
  elevated NT-proBNP?
- **`renal_path_mask`** is a Boolean, one per biomarker. It says which markers renal is *allowed* to
  affect. The adjusted model gets a permissive mask; the causal SCM gets a restrictive one. **That mask
  is the "biological constraint."** The entire phrase "biology-constrained structural causal model"
  cashes out to this one array.
- Parameter counts, from `outputs_latent_endotyping/metadata.json`:
  associative **12** · renal-adjusted **14** · causal SCM **12**. Work out where the 2 extra parameters
  in the adjusted model come from, and why the constrained model has *fewer* parameters than the
  adjusted one but *more structure* than the pooled one.

**Self-test — you must be able to finish these sentences without notes:**
- "The pooled model fails because it has no way to represent…"
- "The adjusted model nearly matches the causal model because…"
- "The causal model still edges it out because…"
- "The biological constraint is, concretely, a…"

---

## M3. How the models are scored when nobody knows the right answer
The obvious question from any reviewer, and from Dr. Khan: *if the mechanism is unobserved, how do you
know which fitted cluster is the atrial one?*

**Where:** `_anchor_order`, **lines 357–380**. Read the docstring — it's four sentences and it answers
the question completely.

**Extract:**
- The anchor contrast is *atrial electrical marker minus competing marker*, standardized.
- **The NT-proBNP-like marker is deliberately excluded from the anchor.** Why? What would happen if you
  included it, under strong renal distortion?
- The simulated truth labels are excluded too. Confirm that for yourself —
  `metadata.json` states: *"True mechanisms are generated and stored by the simulator but are never passed
  to a fit function."* Find where in the code that's enforced.

**Then:** `evaluate_posterior`, **lines 836–876**. This computes every number in your results table.

**Extract the definitions of your own metrics** — you quote these numbers constantly:
- `accuracy` — top-1 true-mechanism ranking. Over which patients?
- `false_atrial_renal_competing` — **the denominator matters.** Which patients are in it? (Answer:
  renal-impaired patients whose *true* mechanism is competing.) This is why it's a safety metric, not an
  accuracy metric.
- `adjusted_rand_index` — agreement between two partitions, corrected for chance. What does 0 mean?
- `brier_score` and `expected_calibration_error` — the difference between being *right* and being
  *appropriately confident*. You need this distinction; it's what makes the transport result interesting.

**Self-test:** "76% false atrial calls" — 76% of *what*, exactly? If you fumble the denominator in front
of a neurologist you lose the room.

---

# TIER F — fluent. Explain, don't derive.

## F1. EM (expectation–maximization), in outline
**Where:** `_associative_e_step` (413) / `_associative_m_step` (382); conditional versions at 566 / 620.
Skim the loop in `fit_associative_latent_class_model` (436). **Do not read the algebra line by line.**

**Extract:** the chicken-and-egg framing. You need class assignments to estimate class parameters, and
parameters to assign classes, so you alternate: soft-assign every patient (E), re-estimate parameters
using those soft weights (M), repeat until the likelihood stops moving.

Also: **why 4 random starts?** (`random_starts: 4`) and **why a variance floor?**
(`variance_floor: 0.0025`) — both are guards against a known failure mode. Find out which.

**External, 20 min:** StatQuest, "Expectation Maximization / Gaussian Mixture Models." One video is
enough for this tier.

## F2. Pearl's ladder — association / intervention / counterfactual
**Where:** external. Any short summary of the three rungs.

**Extract:** which rung each of your models sits on, and — importantly — that your latent-endotyping
experiment is **rung 1 vs rung 1 with better structure**, not rung 1 vs rung 3. The counterfactual
machinery lives in the *other* script. Being precise about this protects you from overclaiming.

## F3. Sufficiency, disablement, and the collapse
**Where:** `r21_preliminary_experiment.py` — the module docstring (lines 1–18) and
`posterior_integrated_counterfactual_scores` (line 218, especially the docstring at 232–238).

**Extract:**
- Sufficiency: *would mechanism k, acting alone, reproduce these labs?*
- Disablement: *if we switched k off and held this patient's background fixed, would the labs change?*
- Abduction → action → prediction: infer the patient's leftover noise, hold it fixed, flip one thing.
  Why must the noise be held fixed for the comparison to be fair?
- The collapse sentence: *"normalized sufficiency and disablement are monotone transformations of the
  correctly specified posterior."* You need to be able to say why monotone ⇒ same ranking.
- **Redundancy** (high sufficiency, low disablement) and why the current generator can't produce it.

**External, 45 min:** Richens, Lee & Johri, *Nat Commun* 2020, "Improving the accuracy of medical
diagnosis with causal machine learning." Read the abstract, Figure 1, and the desiderata section only.
Skip the proofs. This is the paper your framework extends and it *will* come up.

## F4. The transportability experiment
**Where:** `r21_transportability_experiment.py` module docstring (lines 1–11) —
eleven lines that tell you the whole design. Then `outputs_transportability/metadata.json` for the
hospital parameter tables.

**Extract:**
- What is held **invariant** across hospitals (the mechanism→biomarker signature) versus what **varies**
  (renal prevalence and effect, inflammation, assay offset and scale, missingness). Read the four target
  hospital blocks in metadata.json and watch the numbers escalate No → Mild → Moderate → Strong.
- What target recalibration is *allowed to see*: "renal status, inflammation status, assay metadata, and
  unlabeled biomarkers from a small calibration cohort." **No mechanism labels.** 150 patients.
- The four fitted models as a 2×2: path-restricted or not × recalibrated or not. Figure 4c in the draft
  draws this — it's the clearest single picture of the experiment.
- Why the oracle is a *ceiling*, not a competitor.

## F5. The metrics of the transport result
**Extract:** why ECE (calibration) matters *separately* from accuracy, and why the degradation numbers
(−6.86 / −5.42 / −4.08 / −3.92 / −3.83) are more persuasive than the raw accuracies. What does it mean
that the modular model's degradation lands on the oracle's floor?

## F6. BIC and the K=1 null
**Where:** `_bic` (line 1242), `run_k1_null_experiment` (1376), `_wilson_interval` (1394).

**Extract:** BIC as fit minus a complexity penalty; ΔBIC(K2−K1) negative means "two classes preferred."
Median −277.8 for the associative model — is that marginal or emphatic? And what a Wilson interval is
for (proportions near 0 or 1, where the normal approximation breaks).

---

# TIER R — recognize and place

- **Latent class / finite mixture models** — the family your associative baseline belongs to
- **Confounder vs. alternative cause** — your own methodology doc makes this distinction explicitly and
  says "renal biomarker distortion" is the technically correct term. Know why.
- **MCAR / MAR / MNAR** — the three missingness regimes. One sentence each.
- **Adjusted Rand index, Wilson interval, Monte Carlo standard error** — what they're for
- **Twin networks** — the computational trick for counterfactuals; you don't need the mechanics
- **Transportability (Pearl & Bareinboim)** — the formal theory your modular argument gestures at
- **Selective prediction / risk–coverage curves** — the framing for the abstention rule you're building
- **NT-proBNP, PTFV1, left-atrial enlargement** — the three ARCADIA eligibility markers, roughly what
  each measures. Khan will assume you know these.

---

# TIER D — defer without embarrassment

Variance floors, probability floors, Beta prior pseudocounts, EM convergence tolerances, the exact
weighted M-step algebra, `logsumexp` numerical stability, the plotting code, the CSV schema.

**"That's a numerical guard — let me pull up the exact value"** is a *strong* answer. Trying to
improvise it is a weak one.

---

# TONIGHT — the actual path

| # | Time | What |
|---|---|---|
| 1 | 40 min | `R21_PRELIMINARY_METHODOLOGY.md` — read it all. Highest yield thing you own. |
| 2 | 30 min | `R21_TRANSPORTABILITY_METHODOLOGY.md` — first half is enough |
| 3 | 45 min | **M1** — the generator, lines 232–274. Draw the DAG from memory. |
| 4 | 60 min | **M2** — the three factorizations and `renal_path_mask`. Finish the four sentences. |
| 5 | 35 min | **M3** — `_anchor_order` + `evaluate_posterior`. Nail the false-atrial denominator. |
| 6 | 20 min | **F1** — StatQuest EM video |
| 7 | 20 min | **F3** — the collapse docstring, lines 232–238 |
| | **~4 h** | |

**If you run short, cut in this order:** 6, then 2, then 7. **Never cut 3, 4, or 5** — those are M1–M3
and they're what you'd actually be asked.

## This week (after the meeting)
F2 Pearl's ladder · F3 the Richens paper properly · F4 the transport design in full · F6 BIC ·
then re-read the EM steps now that you know what they're doing.

## This month
Rewrite `simulate_two_mechanism_cohort` from scratch, in your own file, without looking. Then extend it
into the E1a generator (add H and I, add γ_H2). **You will understand the methodology on the day you
write a generator yourself.** Nothing else gets you there — not reading, not this document.

---

# The self-test that matters

Answer these out loud, standing up, no notes. If you can, you're ready.

1. Walk me through how one simulated patient is created, in order.
2. Where does renal dysfunction enter — does it change the mechanism, or only the measurement?
3. What are the three biomarkers and which mechanism drives which?
4. Write the three model factorizations. Which one matches the generator?
5. What, concretely, is the "biological constraint"?
6. Why does the adjusted model nearly match the causal model?
7. Why does the constrained model have fewer parameters than the adjusted one?
8. Without truth labels, how do you know which fitted class is atrial?
9. "76% false atrial" — 76% of which patients?
10. What's held fixed across hospitals, and what moves?
11. What does target recalibration get to see? What is it never allowed to see?
12. Why did the counterfactual score match the posterior exactly?
13. What is redundancy, and why can't the current generator produce it?
14. Why is the K=1 null the strongest result in the package?

---

# One honest note

You said you generated the code and just wanted the results. That's a normal way to start and it stopped
being sufficient the moment you became technical lead — which you already know, or you wouldn't have
asked.

Two things worth holding onto. First, the code you generated is genuinely good: the label-orientation
trick, the truth-isolation boundary, the negative controls, the validation JSONs, the docstring that
flags its own degeneracy — that is a careful design, and it's yours to defend. Second, the gap between
where you are and where you need to be is roughly one focused evening plus one weekend of writing a
generator yourself. That is a *small* gap. It only feels large because nobody has drawn you the map.

Also: **you already found the most important technical result in the project** — that the counterfactual
collapse is analytic rather than empirical. You found it by asking the right question about your own
work. Depth of recall is catching up to judgment here, and that's the easy direction to fix.
