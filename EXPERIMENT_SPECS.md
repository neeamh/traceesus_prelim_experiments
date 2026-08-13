# Causal-ESUS — Experiment Specifications
**Owner: Neeam Hayder (technical lead). Written 2 Aug 2026.**
Everything here is buildable from the existing codebase. Notation matches
`r21_latent_endotyping_experiment.py` and `r21_transportability_experiment.py`.

---

## 0. Where the ground truth actually is

Two things are load-bearing and already real:

- **`outputs_latent_endotyping/`** and **`outputs_transportability/`** — 500 paired repeats each, all
  `validation_checks.json` pass, seed 20260728. These are yours, they run, they're reproducible.
- **`outputs/`** — the posterior-vs-counterfactual comparison. Real, but the design is degenerate (§1).

Everything else in the package is scaffolding. The manuscript prose is scaffolding. But note: **the
numbers in the draft's Results section are not invented** — I checked all fourteen against your CSVs and
thirteen matched to the decimal. Whoever wrote the prose was reading your real output. That means the
evidence base is yours and it's solid; only the framing around it is provisional.

The job now is to generate the *missing* truth, not to polish the wrapper.

---

## 1. Why the current counterfactual experiment cannot answer the question

### 1.1 The current generator

`r21_preliminary_experiment.py`, from `outputs/run_metadata.json`:

```
Z_i ∈ {A, C}                      exactly one mechanism, mutually exclusive
λ_A = (1.2, 0.8, 0.0)             atrial signature
λ_C = (0.0, 0.0, 1.0)             competing signature
B_i = λ_{Z_i} + (γ·R_i, 0, 0) + ε_i,    ε_i ~ N(0, I₃)
R_i ~ Bernoulli(0.30),  γ ∈ {0, 0.75, 1.5, 2.25} SD
```

### 1.2 The algebra

With Gaussian noise and a squared-distance evidence metric, all three queries reduce to the same scalar.
Write the kidney-aware log-odds

```
ℓ_i = [ ‖B_i − μ_C(R_i)‖² − ‖B_i − μ_A(R_i)‖² ] / 2σ²
```

Then:

- posterior: `p_A = σ(ℓ_i)` — sigmoid, strictly increasing in ℓ
- sufficiency: `S_iA = E[exp(−d(B_obs, B^{do(A=1)}))]` — strictly increasing in ℓ
- disablement: `D_iA = E[1 − exp(−d(B_obs, B^{do(A=0)}))]` — strictly increasing in ℓ

All three are **strictly monotone functions of the same scalar ℓ**. Monotone maps preserve ordering, so
`argmax` is identical for all three. Top-1 accuracy is identical **by construction, not by measurement.**

This is exactly what the script's docstring says:
> *"In this deliberately symmetric K=2 toy model, normalized sufficiency and disablement are monotone
> transformations of the correctly specified posterior."*

### 1.3 The deeper problem

The framework's diagnostic payoff is **redundancy** — high sufficiency with low disablement, meaning
"this mechanism can explain the evidence, but so can something else." Redundancy requires an
**alternative generating path that survives disabling the mechanism.**

In the current generator: mechanisms are mutually exclusive, and the only non-mechanism path (renal → B₁)
touches one marker while the atrial signature spans two. Disable the atrial gate and B₂'s evidence
collapses with nothing to replace it. **Redundancy is structurally impossible.** The generator cannot
produce the phenomenon the framework exists to detect.

### 1.4 Which premises actually bind

Be precise here — it matters for the paper and it's the sharp version of the argument.

| Premise | Binding? | Reason |
|---|---|---|
| (i) Mutual exclusivity | **YES** | With K=2 exclusive states the whole problem collapses to one scalar ℓ |
| (ii) Symmetric signatures | No | λ_A already spans 2 markers, λ_C spans 1 — asymmetric, and it still collapsed. Asymmetry shifts constants, not ordering |
| (iii) Homoscedastic Gaussian noise | Weak | Changes the metric, but any monotone-in-ℓ score still collapses at K=2 |
| (iv) Complete evidence | **YES, conditionally** | Different observed marker subsets change *which* mechanism is necessary vs merely consistent |
| (v) No redundant generating path | **YES — strongest** | Without an alternative path, disablement carries no information the posterior lacks |

**Conclusion: the collapse breaks when K ≥ 3, when mechanisms co-activate, or when a nuisance path can
regenerate the mechanism's full signature.** Not from asymmetry or noise shape alone. Say it this way and
it's a proposition; say it loosely and it's hand-waving.

---

## 2. E1 — Collapse proposition and divergence study
**The decisive experiment. Everything downstream depends on its result.**

### 2.1 Deliverable A — the proposition (analytic, ~4 h, no compute)

> **Proposition.** Let mechanisms be mutually exclusive with K = 2, evidence complete, noise additive
> Gaussian and homoscedastic, and let no non-mechanism path be capable of generating a mechanism's full
> signature. Then expected sufficiency S_ik, expected disablement D_ik, and the posterior P(Z_i = k | e_i)
> are strictly monotone transformations of a common scalar statistic. Consequently
> `argmax_k S_ik = argmax_k D_ik = argmax_k P_ik`, and top-1 accuracy, false-attribution rate, and every
> ranking-based metric coincide exactly.
>
> **Corollary.** Any observed advantage of counterfactual querying over an equally specified posterior
> must originate from a violation of at least one premise.

This is the continuous-biomarker analogue of Richens' binary noisy-OR result. It is a genuine, citable
contribution, and it converts "our null result" into "we characterized when the query matters."

### 2.2 Deliverable B — five violation families

One premise broken at a time, ordered by expected yield. **Same fitted SCM for both queries in every
family** — that's the whole point of the design.

---

#### E1a — Redundant nuisance paths ← run this first, highest value

**What changes.** Add heart failure as a second nuisance that loads on the *second* atrial marker, so
that renal + HF jointly can reproduce the entire atrial signature.

```
R_i ~ Bern(0.30)          renal dysfunction
H_i ~ Bern(0.20)          heart failure
I_i ~ Bern(0.25)          inflammation

B_i1 = λ_{Z,1} + γ_R·R_i + γ_H1·H_i + ε_1        NT-proBNP-like
B_i2 = λ_{Z,2} + γ_H2·H_i               + ε_2     atrial electrical
B_i3 = λ_{Z,3} + δ_I·I_i                + ε_3     competing-specific

λ_A = (1.25, 1.00, 0.00),  λ_C = (0.00, 0.00, 1.00),  ε ~ N(0, I₃)
```

**The critical parameter is γ_H2.** When `γ_H2 ≈ λ_{A,2} = 1.0`, a patient with H=1 has the atrial
electrical marker elevated *without* the atrial mechanism. Now disable the atrial gate: B₁ is still high
(renal + HF), B₂ is still high (HF). **Evidence survives → low disablement despite high posterior.** That
is redundancy, and it is the NT-proBNP problem stated in code.

**Grid.** γ_R ∈ {0, 0.5, 1.0, 1.5, 2.0}; γ_H1 ∈ {0, 0.75, 1.5}; γ_H2 ∈ {0, 0.5, 1.0, 1.5}; δ_I ∈ {0, 0.8}.
Primary cell: γ_R = 1.5, γ_H1 = 1.0, γ_H2 = 1.0.

**Prespecified subgroups — report separately, this is an explicit PI requirement:**
- *Uncomplicated*: R = 0, H = 0, I = 0
- *Single nuisance*: exactly one of R, H, I = 1
- *Redundant*: R = 1 **and** H = 1 — the cell where divergence should appear

**Primary contrast.** Paired difference in top-1 accuracy, counterfactual − posterior, within the
redundant subgroup, same fitted SCM, 500 paired replicates.

**Prediction.** ≈ 0 in uncomplicated (the proposition holds there). Non-zero in redundant, if the
framework's premise is right. **If it's zero everywhere, that is the headline result** and the paper
becomes a framework + boundary paper — which the draft already pre-authorizes.

**Secondary output — the redundancy diagnostic.** For each patient compute (S_iA, D_iA) and plot the
joint distribution by subgroup. The claim in §4.4 is that the high-S/low-D quadrant flags redundancy.
Show the quadrant populating as γ_H2 rises. **This is the money figure for `fig4_query_ablation.pdf`** —
and it's the one Dr. Khan will immediately understand.

**Cost.** ~14 h build, ~4 h compute.

---

#### E1b — Co-active mechanisms · breaks premise (i)

State space becomes `(A_atrial, A_competing) ∈ {0,1}²` with P(both active) ∈ {0, 0.10, 0.20, 0.40}.

```
B_i = A_atrial·λ_A + A_comp·λ_C + nuisance + ε_i
```

Posterior is now over four states, so "sufficiency of atrial alone" and "posterior mass on atrial" stop
being the same object. **Metrics change:** top-2 coverage, multi-label F1, and per-gate calibration —
not top-1 accuracy, which is ill-defined when two gates are active.

Cost: ~10 h build, ~3 h compute.

---

#### E1c — K = 3 and K = 4 · breaks the two-class scalar collapse

Add a third mechanism gate (inflammatory/prothrombotic) with signature λ_I = (0.4, 0.0, 0.6) — note the
deliberate partial overlap with both existing signatures. With K ≥ 3 the ranking is no longer determined
by a single scalar, and sufficiency and disablement weight the *competing set* differently: sufficiency
sets all others to reference, disablement preserves them. Their argmaxes can genuinely diverge.

Prevalence grid 10–50% per regime. Also gives you the K = 2/3/4 model-selection row Priority 4 needs.

Cost: ~12 h build, ~4 h compute.

---

#### E1d — Mechanism-dependent missingness · breaks premise (iv)

```
P(R_ij = 1 | ·) = logit⁻¹(δ_0j + δ_Aj·A_i + δ_Nj·N_i + δ_Bj·B_ij)
```

`δ_Bj ≠ 0` makes it MNAR — observation depends on the unobserved value. Sweep overall missingness
10–60%. When markers are missing, sufficiency and disablement integrate over different evidence subsets,
so their rankings can separate.

This doubles as the MCAR/MAR/MNAR arm of Priority 4: MCAR sets δ_A = δ_N = δ_B = 0; MAR sets δ_B = 0;
MNAR sets δ_B ≠ 0.

Cost: ~8 h build (reuses transport missingness machinery), ~3 h compute.

---

#### E1e — Heteroscedastic / non-Gaussian noise · breaks premise (iii) · LOWEST priority

Per-mechanism noise scales σ_A ≠ σ_C, and a heavy-tailed variant (t₃). Expected yield is low —
per §1.4 this probably doesn't break the collapse at K = 2. **Run it to close the premise list, not
because you expect a result.** Cut this first if time is short.

Cost: ~5 h build, ~2 h compute.

---

## 3. E2 — Abstention rule
**Build this before the ablation. It's the largest silent gap in the whole package.**

Abstention is claimed in the paper's §4.4, drawn in Figure 1, and promised in R01 Aims 1 and 3.
`grep -rn "abstain" *.py` returns nothing. It does not exist.

### 3.1 Specification

Abstain on patient *i* if **any** trigger fires:

| # | Trigger | Rule |
|---|---|---|
| T1 | Posterior margin | `p_(1) − p_(2) < τ_margin` |
| T2 | Interval overlap | Monte Carlo CI of top score overlaps CI of runner-up |
| T3 | Evidence coverage | fewer than *m* of the top mechanism's signature markers observed |
| T4 | Posterior-predictive failure | Mahalanobis distance of B_i under the fitted predictive exceeds χ²₃,₀.₉₉ |
| T5 | Query disagreement | posterior top-1 ≠ counterfactual top-1 |

τ_margin calibrated on a synthetic development split to hit target coverage levels. **Never tuned on the
evaluation set** — state that explicitly.

### 3.2 Metrics — use the selective-prediction framing

- **Risk–coverage curve**: accuracy among retained patients vs fraction retained. Sweep coverage 100% → 50%.
- **AURC** (area under risk–coverage) — single scalar for the ablation table.
- **Selective accuracy at 90% and 80% coverage.**
- **Abstention precision**: among abstained patients, what fraction would the full model have gotten wrong?
  If abstention is working, this is well above the base error rate.
- Abstention rate stratified by: nuisance burden, missingness, mechanism overlap, shift severity.

**Success criterion, prespecify it:** abstention rate rises monotonically with confounding, missingness,
and shift severity, and selective accuracy at 90% coverage exceeds full-coverage accuracy.

Cost: ~12 h build, ~3 h compute.

---

## 4. E3 — M0–M7 source-of-gain ablation
**Identical cohorts. Paired seeds. Both settings. 500 replicates. Each row adds exactly one component.**

| Row | Model | Path restriction | Nuisance adj. | Target recal. | Query | Meas. model | Abstain |
|---|---|---|---|---|---|---|---|
| M0 | Pooled associative LCM | – | – | – | posterior | – | – |
| M1 | + clinical adjustment | – | ✓ | – | posterior | – | – |
| M2 | + target recalibration | – | ✓ | ✓ | posterior | – | – |
| M3 | Frozen causal SCM | ✓ | ✓ | – | posterior | – | – |
| M4 | Modular causal SCM | ✓ | ✓ | ✓ | posterior | – | – |
| M5 | Same SCM, counterfactual | ✓ | ✓ | ✓ | **suff/disable** | – | – |
| M6 | + informative measurement | ✓ | ✓ | ✓ | suff/disable | ✓ | – |
| M7 | + abstention | ✓ | ✓ | ✓ | suff/disable | ✓ | ✓ |

**The contrasts are the result — report these, not just the best row:**

```
M1 − M0  =  value of clinical/nuisance adjustment
M2 − M1  =  value of unlabeled target recalibration
M3 − M1  =  value of causal path restriction
M4 − M3  =  value of modularization
M5 − M4  =  value of counterfactual querying      ← the contested one
M6 − M5  =  value of modeling the measurement process
M7 − M6  =  value of abstention
```

**Metrics per row:** top-rank accuracy · false atrial attribution (renal/competing subgroup) · multiclass
Brier · ECE · 95% interval coverage · abstention % · AURC. All with paired Monte Carlo 95% CIs.

**Two settings:** (a) renal/HF/inflammatory confounding — the E1a generator; (b) cross-hospital shift —
existing transport generator.

**What already exists:** rows M0–M4 in the transport setting, M0/M1 in the renal setting. But they were
**not run on a common cohort with paired seeds**, so the increments aren't currently comparable. Re-run
everything under one driver.

Cost: ~16 h build (mostly M6, M7, and the paired-seed harness), ~8 h compute.

---

## 5. E4 — Robustness and refutation

| Test | Spec | Status |
|---|---|---|
| K = 1 null | ✅ **DONE** — 500/500 vs 0/500 vs 0/500, median ΔBIC −277.8 vs +27.3 | complete |
| K = 2/3/4 selection | BIC + ICL selection frequency, prevalence 10–50% | from E1c |
| Overlap 0/10/20/40% | top-2 coverage, multi-label recovery | from E1b |
| MCAR / MAR / MNAR | 10–60% missingness, δ_B = 0 vs ≠ 0 | from E1d |
| Omitted renal path | refit with the renal edge deleted; measure bias in mechanism attribution | ~3 h |
| Alternative DAGs | reverse one uncertain edge · add one prohibited edge · delete one mechanism→marker edge | ~5 h |
| Interval coverage | empirical coverage of nominal 95% Monte Carlo intervals | ~3 h |
| Abstention behavior | risk–coverage curves under each stress | from E2 |

Most of E4 falls out of E1 if the generators are written with the right switches. **Design E1's config
object to expose K, overlap fraction, missingness mechanism, and edge set as parameters** — then E4 is
mostly config sweeps rather than new code. That's the single biggest time saving available.

Cost: ~11 h incremental, ~5 h compute.

---

## 6. Build order and effort

| # | Item | Build | Compute | Gate |
|---|---|---|---|---|
| 0 | Clean env, lockfile, one-command runner, tidy result schema | 10 h | – | everything |
| 1 | Proposition (analytic) | 4 h | – | – |
| 2 | **E1a redundant nuisance** | 14 h | 4 h | **decides the paper** |
| 3 | E2 abstention rule | 12 h | 3 h | M7 row |
| 4 | E1d missingness | 8 h | 3 h | MNAR arm |
| 5 | E1c K = 3/4 | 12 h | 4 h | K-selection arm |
| 6 | E1b co-active | 10 h | 3 h | overlap arm |
| 7 | E3 ablation M0–M7 | 16 h | 8 h | needs 2, 3 |
| 8 | E4 graph sensitivity + coverage | 11 h | 5 h | – |
| 9 | E1e heteroscedastic | 5 h | 2 h | cut first |
| | **Total** | **102 h** | **32 h** | |

Compute runs unattended — the binding constraint is the 102 h of build time.

**Runtime estimate method:** `--repeats` and `--workers` flags already exist. Time a 10-repeat run,
multiply by 50, multiply by 8 rows × 2 settings. Do this before the meeting so the compute ask is
concrete.

### Minimum decisive path — if only one thing gets built
**Item 0 → Item 1 → Item 2 (E1a).** ~28 h. That alone determines whether the paper is a counterfactual
paper or a framework-and-boundary paper, and it produces the redundancy quadrant figure. Everything else
is elaboration.

---

## 7. Scope boundaries

**Yours (technical lead):** all generators, model implementations, the abstention rule, the proposition,
all experiments, all figures, the result manifest, the reproducibility harness, and translating all of it
for the clinical side.

**Not yours — do not absorb these:** cohort access and DUAs · clinical validity of the DAG · biomarker
definitions and thresholds · case selection for the neurologist panel · budget and personnel · proposal
prose outside the Approach subsections. If these land on you, the timeline breaks and the science
doesn't get done.

**Needs a decision before you can proceed:** the nuisance structure in E1a — specifically whether HF
loading on the atrial electrical marker is clinically defensible. **Ask Dr. Khan directly.** If HF cannot
plausibly elevate PTFV1, use a different redundant pair (e.g. age or LA-size-driven remodeling) and say
so. The redundancy experiment only means something if the redundant path is real biology.
