# Proposal-ready replacement text

## Central hypothesis

We hypothesize that a biologically constrained latent structural causal model
can recover the categorical mechanism responsible for an observed biomarker
profile, even when a nonmechanistic pathway produces a misleading biomarker
elevation.

## Preliminary experiment

We developed a fully scripted hidden-label simulation with two mutually
exclusive latent mechanisms (atrial and competing), one observed renal
distortion variable, and three continuous biomarkers. The atrial mechanism
increased an NT-proBNP-like biomarker by 1.25 residual SD and an atrial
electrical marker by 1.00 SD; the competing mechanism increased a
competing-specific marker by 1.00 SD. Renal dysfunction occurred in 30% of
patients and independently elevated the NT-proBNP-like marker by 0, 0.5, 1.0,
or 1.5 SD. Thus, renal-impaired patients with a true competing mechanism could
display an atrial-appearing biomarker elevation.

The simulator retained the true mechanism only as an evaluation key. No model
received endotype labels during fitting. At each renal-effect level, we
generated 800 unlabeled training patients and an independent 1,000-patient test
cohort and repeated the complete experiment 500 times.

We compared a standard associative latent class model,
\(p(Z)p(R\mid Z)\prod_jp(B_j\mid Z)\), with a biologically constrained latent
SCM, \(p(Z\mid R)\prod_jp(B_j\mid Z,R)\), in which the direct renal path was
restricted to the NT-proBNP-like biomarker. Both primary models used the same
observed inputs and had 12 free parameters. As a stringent fairness control, we
also fit a 14-parameter renal-adjusted associative latent class regression that
estimated renal associations for every biomarker.

Latent classes were named without truth leakage using a prespecified atrial
electrical-minus-competing biomarker anchor; the misleading NT-proBNP-like
marker was excluded from the label rule. The primary outcome was
true-mechanism ranking accuracy in the independent test cohort. The key
subgroup outcome was false atrial classification among renal-impaired patients
whose true mechanism was competing. All comparisons were paired within repeat,
and 95% Monte Carlo confidence intervals were calculated from repeat-level
variation.

With no renal distortion, the standard associative latent class model and
causal model performed similarly (82.08% versus 82.03% accuracy). Under the
strong 1.5-SD renal effect, associative latent-class accuracy fell to 57.85%
(95% MC CI, 57.33%–58.37%), whereas causal-model accuracy remained 81.94%
(81.81%–82.06%), near the data-generating oracle of 82.70%. The paired causal
advantage was 24.08 percentage points (23.55–24.62). False atrial
classification in the renal-impaired competing-mechanism subgroup was 75.99%
with the standard associative model and 18.51% with the causal model, a paired
reduction of 57.48 points (54.77–60.19).

The renal-adjusted associative control achieved 81.64% accuracy and 19.75%
false atrial classification under strong distortion. The causal model improved
accuracy by 0.30 points (0.22–0.37) and reduced false atrial classification by
1.23 points (0.53–1.94) relative to this adjusted control. The fitted causal
model recovered the 1.50-SD renal path with a mean estimate of 1.494 SD.

We then generated 500 K=1 null cohorts with a real 1.5-SD renal biomarker path
but no endotype heterogeneity and compared K=1 versus K=2 by BIC. The standard
associative latent class model selected a spurious K=2 solution in 500/500
repeats. The renal-adjusted associative model and causal SCM selected K=2 in
0/500 repeats each (Wilson 95% CI, 0.00%–0.76%). All recovery and null K=2 fits
converged after prespecified automatic refitting.

These preliminary results support feasibility of hidden causal endotyping under
a correctly specified biological graph: the latent SCM recovered the true
categorical mechanism without label supervision and resisted a nonmechanistic
renal biomarker pathway that redirected a generic latent class model. The
renal-adjusted control bounds the interpretation. Most of the gain arose from
representing the renal-to-biomarker dependency correctly; the causal exclusion
restriction provided a smaller efficiency gain over a flexible associative
adjustment. We therefore do not claim universal superiority over all
associative latent-variable models. Aim 1 will test whether these advantages
persist under graph misspecification, measurement error, missingness, and
external-environment shifts.

## Figure P1 legend

**Figure P1. Hidden-mechanism recovery under renal biomarker distortion.**
At each renal-effect level, 800 unlabeled training patients and an independent
1,000-patient test cohort were generated in each of 500 paired repeats. Each
patient had one true categorical mechanism, atrial or competing; renal
dysfunction independently elevated the NT-proBNP-like biomarker. (A) Percentage
of test patients whose true mechanism was ranked first. (B) False atrial
classification among renal-impaired patients whose true mechanism was
competing. Lines show repeat means and shaded bands show two-sided 95% Monte
Carlo confidence intervals. Both models used renal status and all three
biomarkers, received no mechanism labels during fitting, and had 12 free
parameters. Latent labels were oriented using a prespecified electrical-minus-
competing biomarker anchor that excluded NT-proBNP-like values and truth labels.

## Figure P2 legend

**Figure P2. Patient-level illustration of renal-path removal.** One simulated
renal-impaired patient had a true competing mechanism, high NT-proBNP-like
biomarker, nearly absent atrial electrical evidence, and strong
competing-specific evidence. The associative latent class model assigned
\(P(\mathrm{atrial})=0.99\). After the fitted SCM removed the estimated renal
contribution while preserving the remaining patient-specific biomarker values,
the causal model assigned \(P(\mathrm{atrial})=0.08\). This illustration is a
model-based explanation, not an independent estimate of clinical effect.

## Claims to avoid

- “Causality universally outperforms association.” The adjusted associative
  latent model performed almost as well.
- “Kidney dysfunction is a classical confounder in this simulation.” In this
  version it is an independent alternative cause of the biomarker.
- “The true endotypes were known to the models.” Truth was used only after
  fitting for simulation evaluation.
- “The K=1 result proves the model can never invent endotypes.” It is evidence
  under this null data-generating process and BIC rule.
- Any suggestion that the standardized effect sizes are clinical estimates
  from ARCADIA or another patient cohort.
