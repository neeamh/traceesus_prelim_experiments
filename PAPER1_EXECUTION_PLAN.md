# Paper 1 — execution plan against the PI's checklist
**Revised 2 August 2026.** Conflict rule in force: **most recent statement wins.** The checklist and the
Overleaf draft (both 2 Aug) override the 1 Aug action plan, which overrides the late-July call.

---

## 0. The one thing that changes the plan

`r21_preliminary_experiment.py` already runs a same-fitted-SCM posterior-vs-counterfactual comparison.
`outputs/main_simulation_summary.csv` shows the two produce **bit-identical means at all four
confounding levels** (0.811388 / 0.811440 / 0.812222 / 0.812198 accuracy; 0.135219 / 0.136360 /
0.136241 / 0.133976 false atrial).

That looks like Priority 2 is already answered. **It is not**, and the reason matters. The script's own
docstring says why:

> *"In this deliberately symmetric K=2 toy model, normalized sufficiency and disablement are monotone
> transformations of the correctly specified posterior."*

The identity is **analytic, not empirical.** Monotone transforms preserve ordering, so top-1 ranking is
identical by construction. Running Priority 2 on the current generator will produce a guaranteed null
that teaches nothing — and a reviewer who reads the code will say so.

Worse: the framework's own §4.4 defines the diagnostic payoff as **redundancy** — "high sufficiency
with low disablement… the mechanism can explain the evidence, but plausible alternatives can also
preserve it." In a symmetric two-class model with *mutually exclusive* mechanisms, redundancy is
structurally impossible. There is no alternative left to preserve the evidence. **The current generator
cannot produce the phenomenon the paper claims to detect.**

### What to do instead — state the collapse, then break it

Turn the weakness into the contribution. Two parts:

**Part A — Proposition (analytic).** Under (i) mutually exclusive mechanisms, (ii) symmetric mechanism
signatures, (iii) additive homoscedastic Gaussian noise, (iv) complete evidence, and (v) no non-mechanism
path capable of generating the same evidence, expected sufficiency and disablement are monotone in the
posterior and top-1 ranking is identical. This is the continuous-biomarker analogue of Richens' binary
noisy-OR finding, and it is a citable result rather than a null.

**Part B — Violate each premise and measure divergence.** Prespecified families, one premise broken at a time:

| Family | Premise broken | Why it should separate the queries |
|---|---|---|
| **Redundant nuisance** ← *highest value* | (v) | Renal / HF / inflammation act on the same markers as the atrial gate. Disabling the atrial mechanism leaves the evidence standing via the nuisance route → high posterior, **low disablement**. This is literally the NT-proBNP problem. |
| **Co-active mechanisms** | (i) | 0–40% of patients carry two active gates. "k alone reproduces the evidence" and "removing k destroys it" stop coinciding. |
| **Asymmetric signatures** | (ii) | Atrial gate drives 2 markers, competing gate drives 1; unequal effect magnitudes. Breaks the monotone map. |
| **Mechanism-dependent missingness** | (iv) | Evidence distance is computed over different marker subsets per patient. |
| **Heteroscedastic noise** | (iii) | Per-mechanism noise scales differ. |

Report **uncomplicated vs confounded subgroups separately**, exactly as the checklist demands. If
divergence appears nowhere, the honest conclusion — already pre-written in the draft's §6.3 — is a
narrower framework paper on nuisance modeling, identifiability, and uncertainty. That is publishable
and it is the paper's stated fallback.

**This also fixes the R01's Aim 1.** The PI's M1 is "develop the methodology." A proposition plus the
conditions under which counterfactual querying does and does not add information *is* a methodological
object. Same work, two deliverables.

---

## 1. Manuscript number audit — I checked every value against the locked outputs

| Claim in draft | Locked output | Verdict |
|---|---|---|
| 81.9% / 57.9% accuracy, strong renal | 81.94 / 57.85 | ✅ |
| 18.5% / 76.0% false atrial | 18.51 / 75.99 | ✅ |
| Renal-adjusted **19.8%** false atrial | **19.75 → 19.7%** | ❌ **wrong in 3 places** — abstract, §6.1, Fig 2 caption |
| Renal-adjusted 81.6% accuracy | 81.64 | ✅ |
| 77.6 / 73.4 modular vs pooled, strong shift | 77.60 / 73.37 | ✅ |
| 4.23 pp, CI 4.11–4.36 | 4.2304, [4.105, 4.356] | ✅ |
| 0.47 pp, CI 0.38–0.55 | 0.4654, [0.382, 0.548] | ✅ |
| 44.7 / 36.6 / 23.4 / 22.5 / 22.2 false atrial | 44.73 / 36.60 / 23.37 / 22.52 / 22.15 | ✅ |
| n=800 / 1,000 / 600 / 150 / 500 repeats | metadata.json | ✅ |

One error, three occurrences. Everything else is clean.

**Also fix:** §7.1 still says *"the R21's methodological and real-data objectives."* The R21 is dead —
change to R01. And §6.5 says `[RESULT REQUIRED]` for the K=1 null, but **that result already exists**
(§2 below). Fill it in immediately; it costs nothing and it is the paper's strongest number.

---

## 2. Coverage map — what exists vs what the checklist demands

### Priority 1 — lock existing experiments · **~85% there**
| Item | Status |
|---|---|
| Renal-distortion experiment | ✅ `r21_latent_endotyping_experiment.py`, all validation checks pass |
| Cross-hospital experiment | ✅ `r21_transportability_experiment.py`, all validation checks pass |
| Clean-environment rerun | ❌ never done; no venv, no lockfile, no one-command entrypoint |
| One tidy result file per experiment | ❌ results spread across 10+ CSVs per directory |
| Verify manuscript values | ✅ done above — one error found |
| Vector figures | ⚠️ PDFs exist but must be **regenerated** from the locked files, not reused |

### Priority 2 — decisive counterfactual experiment · **0% usable**
Existing `outputs/` run is analytically degenerate (§0). Needs a new generator with the five premise
violations, plus prespecified uncomplicated/confounded subgroup reporting.

### Priority 3 — source-of-gain ablation (Table 1, 8 rows) · **~55% there**
| Row | Status |
|---|---|
| 1 Unadjusted associative | ✅ both settings |
| 2 Clinically adjusted associative | ✅ renal setting; ✅ transport setting |
| 3 Target-calibrated associative | ✅ transport only |
| 4 Frozen causal SCM | ✅ transport only |
| 5 Modular causal SCM, posterior | ✅ transport only |
| 6 Same SCM, counterfactual query | ❌ depends on Priority 2 |
| 7 + informative measurement model | ❌ not implemented |
| 8 + uncertainty-aware abstention | ❌ **abstention does not exist anywhere in the codebase** |

Rows 1–5 exist but were **not run on a common cohort with paired seeds across both settings**. The table
requires that. Rows 6–8 are new code.

> **Abstention is claimed in the paper (§4.4), in Figure 1, and in R01 Aims 1 and 3 — and it is not
> implemented.** That is the largest silent gap in the package.

### Priority 4 — refutation and robustness · **~25% there**
| Item | Status |
|---|---|
| K=1 null | ✅ **done** — pooled associative 500/500 false K=2; renal-adjusted 0/500; causal SCM 0/500; median ΔBIC −277.8 vs +27.3 |
| K=2–4 | ❌ K hard-coded to 2 |
| Overlapping mechanisms | ❌ not implemented |
| MCAR / MAR / MNAR sensitivity | ⚠️ one MAR-style process exists; no three-way comparison |
| Omitted renal path | ❌ |
| Alternative plausible DAGs | ❌ |
| Interval coverage | ❌ |
| Abstention behavior | ❌ |

---

## 3. Effort reality — read this before committing to a date

| Priority | Work | Est. hours |
|---|---|---|
| P1 | venv + lockfile + one-command runner + tidy consolidation + regenerate figures | 10–14 |
| P2 | proposition + new generator (5 premise violations) + run + subgroup analysis + fig4 | 28–36 |
| P3 | 3 new model variants (counterfactual row, measurement model, abstention rule) + re-run all 8 rows on paired cohorts × 2 settings | 26–32 |
| P4 | K=2–4, overlap, MCAR/MAR/MNAR, graph sensitivity, coverage + fig5 | 26–32 |
| Writing | results prose, captions, limitations, repo release, Khan packet | 16–20 |
| **Total** | | **106–134 h** |

At 8 h/week that is 3–4 months. At 20 h/week, 6–7 weeks. **Sep 1 for the complete checklist is not
achievable and should not be promised.** Say this Monday rather than in September.

---

## 4. Proposed schedule — anchored to Oct 5, not Sep 1

The binding constraint is the **R01 receipt date (Oct 5)**. The preprint's job is to exist and be
citable as preliminary evidence before the proposal is finalized. Posting ~Sep 20 gives two weeks of
buffer. That is a defensible target; Sep 1 is not.

| Window | Deliverable | Gate |
|---|---|---|
| **Mon Aug 3** | Meeting. Present the collapse finding, the effort estimate, and the scope options. | PI picks a scope tier (§5) |
| **Aug 3 – 7** | **P1 complete.** Clean venv, lockfile, `make all`, one tidy Parquet per experiment, all figures regenerated as vectors, 19.8→19.7 fixed, R21→R01 fixed, **K=1 null numbers written into §6.5** | Independent runner reproduces both experiments |
| **Aug 7 – 17** | **P2.** Proposition written; new generator with 5 premise violations; run; uncomplicated vs confounded subgroups; `fig4_query_ablation.pdf` | — |
| **Mon Aug 17** | **DECISION GATE.** P2 result sets the title, abstract, and conclusion. Divergence found → counterfactual paper. No divergence → framework + boundary paper, and Table 1 row 6 collapses into row 5. | PI + Khan sign the framing |
| **Aug 17 – 28** | **P3.** Implement measurement model + abstention rule. Re-run all 8 rows on paired cohorts and seeds, both settings. Fill `tables/ablation_placeholder.tex` | — |
| **Aug 28 – Sep 8** | **P4.** K=2–4, overlap, MCAR/MAR/MNAR, omitted renal path, alternative DAGs, coverage, abstention behavior. `fig5_robustness.pdf` | — |
| **Sep 8** | **Evidence freeze.** No new numbers after this date. Manifest maps every value to a file and commit hash | Independent clean-environment reproduction |
| **Sep 8 – 11** | Full manuscript pass. Every percentage re-proofed against locked outputs. Repo release, DOI, environment file | — |
| **Sep 11** | **Send Khan the approval packet**: final DAG, mechanism definitions, nuisance pathways, captions, clinical interpretation, Discussion, Limitations | Allow 5–7 days |
| **Sep 11 – 18** | Khan review. Record approval or corrections in a decision log. Meanwhile: funding, competing interests, CRediT, acknowledgments, repo URL, release tag | — |
| **~Sep 20** | **`\workdraftfalse` → bioRxiv submission** | All 11 items in the draft's Appendix C checked |
| **Oct 5** | R01 receipt date — preprint cited as preliminary evidence | Confirm Oct 5 vs the PI's verbal Oct 7 |

### Dates to fix with the PI on Monday
1. **Is Sep 20 acceptable for the preprint**, given the full checklist cannot land by Sep 1?
2. **Is ML4H (Sep 9) still in play?** It is absent from the checklist. If it matters, only Tier B or C
   below fits, and the decision is needed now — not in September.
3. **Oct 5 vs Oct 7** for the R01 receipt date. Still unresolved from last week.
4. **Khan's review turnaround** — 5–7 days is an assumption. Confirm it.
5. **Weekly hours.** Notion currently has this project at 8 h/wk. The checklist needs ~20.

---

## 5. Three scope tiers — bring these as options, not complaints

**Tier A — full checklist.** All four priorities. ~120 h, preprint ~Sep 20–27. Strongest paper; misses
ML4H; tight against the R01.

**Tier B — decisive core** *(recommended)*. P1 + P2 + Table 1 rows 1–6 + K=1 null + missingness
sensitivity. Defer rows 7–8 (measurement model, abstention), K=2–4, overlap, and alternative-DAG
sensitivity to a companion paper. ~70 h, preprint ~Sep 8. Keeps the decisive experiment, which is the
one that determines what the paper can claim.

**Tier C — framework + locked evidence.** P1 only, plus the K=1 null and the collapse proposition stated
analytically without the empirical divergence study. ~30 h, preprint ~Aug 20. Weakest, but it exists
early and can be cited in the R01. The paper would need to drop the counterfactual claim from the title.

My read: **Tier B.** It preserves the one experiment whose result changes the paper's conclusion, and it
still produces the M0–M5 ablation the R01's Aim 1 needs. Tier A is the right paper but the wrong
calendar; Tier C forfeits the contribution.

---

## 6. Standing rules for this paper

- Update the Overleaf manuscript **directly** after each experiment. Not chat, not slides.
- Every numerical claim points to a locked result file **and** the script that produced it.
- No manual transcription of values into plots. Figures regenerate from the tidy files.
- Report failures and non-convergence by method and scenario; do not silently drop them.
- Uncomplicated and confounded subgroups always reported separately.
- The counterfactual claim enters the title, abstract, and conclusion **only if** the same-model
  comparison supports it.
- Nothing goes to bioRxiv before Khan's written approval is recorded.
