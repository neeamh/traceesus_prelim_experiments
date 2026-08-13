# Proposal-ready transportability text

## Hypothesis

We will test whether a biologically constrained latent structural causal model
improves transportability of categorical mechanism recovery relative to
associative latent-variable models when nuisance biomarker relationships change
across hospitals. We do not assume universal causal-model superiority. Under
identical source and target distributions, the causal model is expected to
approximately match a well-specified associative model.

## Approach

We developed a fully scripted hidden-label simulation in which each patient has
one of two mutually exclusive mechanisms, atrial or competing. The simulator
generates an NT-proBNP-like biomarker, an atrial electrical marker, and a
competing-mechanism marker. The atrial and competing mechanism-to-biomarker
effects remain invariant across hospitals. In contrast, hospitals differ in
the prevalence of renal dysfunction, the renal contribution to the
NT-proBNP-like marker, background inflammatory burden, assay calibration, and
biomarker missingness.

Each of 500 paired Monte Carlo repeats contains three source hospitals with 600
unlabeled patients per hospital, a 150-patient unlabeled target calibration
cohort, and an independent 1,000-patient target test cohort. Target shifts are
graded as reference, mild, moderate, and strong. True mechanism labels are
retained only by the simulator and are used solely after prediction to evaluate
recovery; no model receives mechanism supervision.

We compare four deployable strategies. A pooled associative latent class model
ignores hospital-specific nuisance paths. A target-calibrated associative
latent model flexibly estimates associations from renal dysfunction and
inflammation to every biomarker using unlabeled source and target data. A
frozen causal latent SCM uses the prespecified biological graph but retains
source nuisance coefficients. A modular causal latent SCM restricts nuisance
paths to renal dysfunction \(\rightarrow\) NT-proBNP-like biomarker and
inflammation \(\rightarrow\) competing-mechanism biomarker, then recalibrates
only those permitted paths in the unlabeled target cohort. A target oracle
defines the information ceiling but is not treated as a deployable comparator.
All methods use the same assay metadata; laboratory offsets and scales are
assumed known in this preliminary experiment.

The primary outcome is the percentage of target patients for whom the
highest-probability latent class matches the simulated true mechanism. The key
subgroup outcome is false atrial classification among renal-impaired patients
whose true mechanism is competing. Secondary outcomes include adjusted Rand
index, Brier score, calibration error, and accuracy with missing biomarkers.
All comparisons are paired within repeat and summarized with two-sided 95%
Monte Carlo confidence intervals.

## Preliminary results

In the strong-shift target hospital, true-mechanism recovery was 73.37% for the
pooled associative latent class model, 77.14% for the target-calibrated
associative model, 76.18% for the frozen causal model, and 77.60% for the
modular causal model. The modular causal model exceeded the pooled associative
model by 4.23 percentage points (95% Monte Carlo CI, 4.11 to 4.36) and the
target-calibrated associative model by 0.47 points (0.38 to 0.55). Relative to
the reference target, accuracy declined by 6.86 points for the pooled
associative model, 4.08 points for the adjusted associative model, and 3.92
points for the modular causal model.

False atrial classification in the renal-impaired, competing-mechanism subgroup
was 44.73% for the pooled associative model, 23.37% for the adjusted associative
model, and 22.52% for the modular causal model. The modular causal reduction was
22.21 points versus the pooled model (21.67 to 22.75) and 0.85 points versus the
adjusted associative control (0.37 to 1.33).

In a separate identical-distribution negative control, all source and target
hospitals shared the same generating parameters. Accuracy was 81.24% for the
target-calibrated associative model and 81.52% for the modular causal model.
Their paired difference was 0.28 points (0.23 to 0.33), fully contained within
the prespecified \(\pm1\)-point equivalence margin.

One-factor ablations clarified the mechanism of the result. Shifting only the
renal pathway reduced pooled associative accuracy by 3.90 points but modular
causal accuracy by only 0.03 points. Shifting missingness alone reduced accuracy
by approximately 4.3 points for every method, including the oracle, indicating
irreducible information loss. Changing assay calibration alone had no effect
because known laboratory metadata were inverted exactly for all methods.

These results support feasibility and the proposed transportability hypothesis
under a correctly specified graph. They do not establish universal superiority
of causal over associative learning: a well-specified, target-calibrated
associative model recovered most of the same benefit. The preliminary evidence
instead shows that biologically justified exclusion restrictions can modestly
improve recovery beyond flexible adjustment and substantially improve recovery
relative to transport-naive latent classes. The poor performance of the frozen
causal model further demonstrates that graph structure alone is insufficient;
site-specific nuisance coefficients must still be updated.

## Figure T1 legend

**Figure T1. Hidden-mechanism transportability under target-hospital shift.**
Three source hospitals contributed 600 unlabeled patients each. In every target
condition, 150 additional unlabeled patients were used for target calibration
and an independent 1,000-patient cohort was used for evaluation. The complete
experiment was repeated 500 times with paired random draws across shift levels.
The mechanism-to-biomarker effects remained invariant while renal prevalence
and its NT-proBNP-like effect, inflammatory burden, assay calibration, and
missingness changed. (A) Percentage of patients whose true simulated mechanism
was ranked first. (B) False atrial classification among renal-impaired patients
with a true competing mechanism. Lines show repeat means; bands show two-sided
95% Monte Carlo confidence intervals. Mechanism labels were never used in
fitting or target calibration.

## Figure T3 legend

**Figure T3. One-factor target-hospital shift ablations.** Each nuisance
component was moved separately from the reference value to its strong-shift
value while the remaining components were held fixed. Points show paired mean
accuracy change from the reference target across 500 repeats; error bars show
two-sided 95% Monte Carlo confidence intervals. The kidney-path shift selectively
degraded the transport-naive pooled latent class model. Increased missingness
degraded every method, including the oracle. The assay-only effect was zero
because calibration metadata were assumed known and applied equally to all
methods.

## Claims to avoid

- “The causal model proves causality.” This is a simulation under a specified
  causal graph.
- “The causal model always beats associative models.” The adjusted associative
  control nearly matched it.
- “The no-shift point on the main curve is an identical-distribution control.”
  It matches the reference source hospital; the separate exact control is the
  literal source-target identity test.
- “The model is robust to unknown assay drift.” Assay calibration metadata are
  known in this experiment.
- “Missing biomarkers are recovered by causal structure.” Missingness harmed
  even the oracle.
- “Kidney dysfunction is necessarily a classical confounder.” Here it is an
  observed alternative cause of a biomarker and is independent of the hidden
  mechanism.

