# TRACE-ESUS experiment validation notes

Date checked: 2026-08-06

## Bottom line

The preliminary synthetic evidence supports **nuisance-aware latent recovery and transportability under specified simulations**. It does not yet support the stronger claim that counterfactual querying itself beats posterior inference, nor that the causal model has beaten an adjusted associative model in ARCADIA real data.

## 1. ARCADIA decision-grade notebook

Drive artifact: [arcadia_pipeline_decision_grade.ipynb](https://drive.google.com/file/d/1acAs074Nnl0ynfL9Bo5VgtEBYVJAS3qo/view)

The notebook was downloaded with `arcadia (1).dta` and rerun top-to-bottom in decision mode. The rerun completed with 22/22 cells present, 15/15 code cells executed, and no cell errors.

Verified anchors:

- N randomized: 1,015.
- Treatment allocation: 507 versus 508.
- Recurrent strokes over available follow-up: 80.
- Median follow-up: 564 days.
- Events by the two-year analysis horizon: 71 (7.0%).
- NT-proBNP observed: 998/1,015 (98.3%).
- PTFV1 observed: 1,003/1,015 (98.8%).
- Left atrial diameter observed: 942/1,015 (92.8%).
- All three jointly observed: 919/1,015 (90.5%).

Simulation results from the actual-data-anchored rerun:

| Prespecified K | Null type-I error | 95% CI coverage | Moderate-scenario power |
|---:|---:|---:|---:|
| 2 | 5.0% | 95.0% | 9.3% |
| 3 | 5.3% | 94.7% | 6.4% |
| 4 | 4.5% | 95.5% | 8.1% |
| 5 | 4.8% | 95.2% | 7.1% |

The moderate scenario used a responder treatment hazard ratio of 0.65, score separation of 1.25, and responder prevalence of 30%. The most optimistic tested sensitivity corner (K=2, hazard ratio 0.50, separation 2.0) reached 18.0% power.

Interpretation: the interaction test is calibrated, but the trial has too few events for confirmatory treatment-effect heterogeneity. ARCADIA is defensible for exploratory frozen-score evaluation and planning, not as proof of clinical treatment guidance.

This notebook does **not** fit and compare causal versus associative latent endotyping models on ARCADIA, estimate split-sample fingerprint stability, test a held-out biological anchor, validate the proposed DAG, or validate counterfactual sufficiency/disablement. The original bracketed “outcome-blind ARCADIA pilot” sentence therefore cannot be filled as written.

## 2. Latent endotyping experiment

Reported results in the current Drive artifact are internally consistent with the proposal summary:

- No distortion: associative 82.08%; causal 82.03%.
- Strong renal distortion: pooled associative 57.85%; causal 81.94%; renal-adjusted associative 81.64%.
- False atrial attribution under strong distortion: pooled 75.99%; causal 18.51%.
- K=1 null: pooled association selected K=2 in 500/500 replicates; adjusted association and causal model selected K=1 in 500/500.

However, the current Drive notebook is not a clean executable handoff: only 5 of 10 code cells have execution counts, one cell fails because notebook kernel arguments are passed into `argparse`, and another fails on a missing or incorrect companion-script path. Its result cells are not executed in the saved copy. These numbers should be described as **reported saved results** until the packaged notebook is repaired and reproduced from a clean environment with raw replicate outputs.

Scientific interpretation: the experiment shows that a misspecified pooled associative model is badly fooled by renal distortion. A correctly renal-adjusted associative comparator nearly matches the causal model. The observed gain therefore comes from encoding the relevant nuisance pathway, not from the word “causal” or from counterfactual querying alone.

## 3. Transportability experiment

Saved outputs report:

- No shift: pooled associative 80.23%; target-adjusted associative 81.22%; modular causal 81.52%.
- Strong combined shift: pooled 73.37%; adjusted 77.14%; modular causal 77.60%.
- Strong-shift modular advantage: +4.23 percentage points versus pooled (95% Monte Carlo CI 4.11 to 4.36) and +0.47 versus adjusted (0.38 to 0.55).
- Renal-path shift: pooled −3.90 points; adjusted −0.13; modular −0.03.
- Missingness shift: approximately −4.3 points for every method.
- Confounded-subgroup false atrial classification under strong shift: pooled 44.7%; adjusted 23.4%; modular causal 22.5%.
- The identical-distribution negative control passed: modular minus adjusted was +0.28 points, within the prespecified ±1-point equivalence margin.

The current Drive notebook contains saved output objects but all code-cell execution counts are null, and the companion raw outputs were not independently regenerated in this review. Treat these as **traceable reported outputs**, not a fresh independent reproduction.

Scientific interpretation: modular nuisance handling improves transport under the simulated renal shift and provides a small advantage over a correctly adjusted associative model. It does not solve missingness by itself. Assay-shift loss was zero by construction because calibration was known; robustness to unknown assay drift remains untested.

## 4. Counterfactual claim boundary

The proposal may state that Aim 1 **will test** whether counterfactual sufficiency/disablement adds value by comparing posterior and counterfactual queries from the **same fitted SCM**. The existing preliminary experiments do not yet earn a sentence saying that counterfactual scoring outperformed posterior inference. Comparing a kidney-aware SCM with a kidney-blind associative model confounds model class, nuisance information, and query type.

A valid counterfactual incremental-value experiment must hold the fitted SCM, observations, training data, and nuisance information fixed, then vary only the query:

1. Posterior ranking: rank the latent mechanisms by their posterior probability.
2. Counterfactual scoring: disable each candidate mechanism and quantify whether the observed biomarker pattern would persist.
3. Evaluate paired true-mechanism ranking, false atrial attribution in the renal-confounded subgroup, calibration, and abstention across overlap and confounding strength.
4. Include a K=1 null, graph-misspecification stress test, and a non-identifying regime where the correct answer is uncertainty rather than a forced class.

Until that experiment is executed, “counterfactual querying adds incremental value” remains a prospective hypothesis.

## 5. Unresolved proposal fields

| Field | Status | Defensible action |
|---|---|---|
| ARCADIA N | Verified: 1,015 | Fill. |
| ARCADIA real-data causal-versus-associative endpoint/result | Not produced by the reviewed notebook | Replace the sentence with the feasibility result; do not invent a comparison. |
| ARCADIA split-sample stability | Not found | Leave out until a prespecified stability analysis is run. |
| Held-out biological anchor result | Not found | Leave out until the anchor and test are prespecified and executed. |
| Brown/Rhode Island cohort N, sites, follow-up | Unconfirmed | Obtain a cohort specification and collaborator/access documentation. |
| Temporal cutoff date | Unconfirmed | Derive from the confirmed enrollment extract. |
| Silent-pilot N | Unconfirmed | Power/precision-justify from operational endpoints and expected throughput. |

Drive retrieval confirms the files could be read for this review; it does not independently verify sharing authority, DUA coverage, IRB status, or permission to use the external cohort.
