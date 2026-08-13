# R21 cross-hospital transportability experiment

## Scientific question

Can a biologically constrained latent structural causal model preserve recovery
of a patient's hidden categorical mechanism when deployed in a hospital whose
nuisance biomarker relationships differ from those in the training hospitals?

The hypothesis is deliberately comparative, not absolute:

> The causal model will preserve latent-mechanism recovery better than
> associative latent models when nuisance biomarker relationships change
> across environments.

The experiment does **not** assume that a causal model must win. A literal
identical-distribution condition is used as a negative control: the modular
causal model should approximately match a well-specified associative latent
model when source and target distributions are the same.

## Estimand and evaluation principle

Each simulated patient has one mutually exclusive mechanism:

- atrial mechanism;
- competing mechanism.

The mechanism label is retained by the simulator only as an evaluation key. It
is never passed to model fitting, nuisance-path estimation, target calibration,
class orientation, or prediction. The primary estimand is the difference in
test-set top-rank accuracy between methods:

\[
\Delta_{\mathrm{accuracy}}
=
P(\widehat Z_{\mathrm{causal}}=Z)
-
P(\widehat Z_{\mathrm{associative}}=Z).
\]

The prespecified safety outcome is false atrial classification among patients
who have renal dysfunction and a true competing mechanism.

## Data-generating structural model

Let \(Z_i\in\{A,C\}\) denote the hidden mechanism, \(R_i\) observed renal
dysfunction, and \(I_i\) observed background inflammation. The three
standardized biomarkers are:

1. NT-proBNP-like marker;
2. atrial electrical marker;
3. competing-mechanism marker.

Before laboratory calibration and missingness:

\[
\mathbf B_i^*
=
\boldsymbol\lambda_{Z_i}
+(\gamma_h R_i,0,0)
+(0,0,\delta_h I_i)
+\boldsymbol\epsilon_i,
\qquad
\boldsymbol\epsilon_i\sim N(\mathbf 0,I_3).
\]

The stable mechanism signatures are:

\[
\boldsymbol\lambda_A=(1.25,1.00,0.00),\qquad
\boldsymbol\lambda_C=(0.00,0.00,1.00).
\]

Hospital \(h\) changes only nuisance relationships:

- prevalence of renal dysfunction;
- renal effect \(\gamma_h\) on the NT-proBNP-like marker;
- prevalence of background inflammation;
- laboratory assay offset and scale;
- biomarker-specific missingness.

The mechanism-to-biomarker signatures \(\boldsymbol\lambda_A\) and
\(\boldsymbol\lambda_C\), residual variances, and mechanism prevalence remain
fixed. Renal dysfunction and background inflammation are generated
independently of \(Z\). That independence is essential: it permits estimation
of nuisance paths from unlabeled target data in this first experiment.

The observed assay value is

\[
B_{ij}^{\mathrm{raw}}=a_{hj}+s_{hj}B_{ij}^{*}.
\]

Assay offsets and scales are assumed to be available as laboratory metadata and
are inverted identically for every method. Therefore, the assay-only ablation
is a manipulation check for correct harmonization, not evidence that the
causal model can discover an unknown assay shift.

Missingness is hospital- and biomarker-specific and also depends on observed
renal dysfunction for the NT-proBNP-like marker and observed inflammation for
the competing marker. Models marginalize missing biomarkers in their
likelihoods; no mean imputation is used.

## Hospitals and sample sizes

Each of 500 paired Monte Carlo repeats contains:

- three source hospitals with 600 unlabeled patients each;
- a held-out target calibration cohort with 150 unlabeled patients;
- an independent held-out target test cohort with 1,000 patients.

The target mechanism labels are used only after all predictions have been made.
Common random numbers are used across shift levels within a repeat so that
paired differences isolate the effect of the environmental intervention.

### Source hospitals

| Source | Renal prevalence | Renal effect on NT-like marker | Inflammation prevalence | Inflammation effect | Base missingness |
|---|---:|---:|---:|---:|---|
| A | 0.25 | 0.80 SD | 0.20 | 0.70 SD | 0.05, 0.04, 0.05 |
| B | 0.30 | 1.00 SD | 0.25 | 0.80 SD | 0.07, 0.05, 0.06 |
| C | 0.35 | 1.20 SD | 0.30 | 0.90 SD | 0.09, 0.06, 0.08 |

### Held-out target sequence

| Target | Renal prevalence | Renal effect | Inflammation prevalence | Assay offset | Assay scale | Base missingness |
|---|---:|---:|---:|---|---|---|
| Reference | 0.30 | 1.00 | 0.25 | 0.00, 0.00, 0.00 | 1.00, 1.00, 1.00 | 0.07, 0.05, 0.06 |
| Mild | 0.40 | 1.25 | 0.35 | 0.15, -0.10, 0.08 | 1.05, 0.98, 1.03 | 0.12, 0.10, 0.10 |
| Moderate | 0.50 | 1.50 | 0.45 | 0.30, -0.20, 0.16 | 1.10, 0.96, 1.06 | 0.22, 0.18, 0.17 |
| Strong | 0.60 | 1.80 | 0.55 | 0.45, -0.30, 0.24 | 1.15, 0.94, 1.09 | 0.35, 0.30, 0.28 |

The main curve's reference target matches Source B. Because a heterogeneous
three-source mixture is not literally identical to Source B, a separate exact
negative control sets all three source hospitals and the target hospital to the
same reference parameters.

## Models

All fitted endotype models are two-component diagonal Gaussian mixtures. They
are estimated by four-start expectation-maximization, support partially
observed biomarker vectors, and orient the atrial component using the
prespecified atrial-electrical minus competing-marker contrast. Neither the
NT-proBNP-like marker nor the true mechanism is used to name the classes.

### 1. Pooled associative latent class model

This model pools assay-harmonized source biomarkers and estimates
\(p(Z)\prod_jp(B_j\mid Z)\). It does not model renal or inflammatory nuisance
paths. It represents a conventional transport-naive latent class analysis.

### 2. Target-calibrated associative latent model

This is the strongest associative control. Within each source hospital and in
the unlabeled target calibration cohort, it estimates every possible linear
association from renal dysfunction and inflammation to every biomarker. It
residualizes these estimated associations before fitting and prediction.

This model receives the same observed nuisance variables and target calibration
sample as the modular causal model. It is intentionally flexible and therefore
the fairest test of whether the causal exclusion restrictions add value beyond
ordinary adjustment.

### 3. Frozen causal latent SCM

This model restricts the nuisance graph to renal dysfunction
\(\rightarrow\) NT-proBNP-like marker and inflammation
\(\rightarrow\) competing marker, but freezes the nuisance coefficients at
their source-hospital average. It tests whether possessing the correct graph
alone is sufficient for transport.

### 4. Modular causal latent SCM

This model uses the same restricted graph but re-estimates the two permitted
nuisance coefficients from the 150 unlabeled target calibration patients. The
source-trained latent mechanism module is then applied to target biomarkers
after removal of the target-specific nuisance contributions.

### 5. Target oracle

The oracle knows the data-generating path coefficients and mechanism
signatures. It is not a deployable competitor. It estimates the ceiling imposed
by biomarker overlap and missingness.

## Outcomes and uncertainty

Primary:

- true-mechanism top-rank accuracy in the target test cohort.

Secondary:

- adjusted Rand index;
- false atrial classification in the renal/competing subgroup;
- Brier score;
- expected calibration error;
- accuracy among patients with any missing biomarker;
- change in accuracy from the reference target to the strong-shift target.

Results are computed within every repeat. All method contrasts and
shift-versus-reference changes are paired by repeat. Two-sided 95% Monte Carlo
confidence intervals use the \(t_{499}\) distribution. These intervals quantify
simulation uncertainty in the mean operating characteristic; they are not
clinical confidence intervals.

The exact no-shift negative control uses a prespecified equivalence margin of
\(\pm1\) percentage point for modular causal minus target-calibrated
associative accuracy. Equivalence is declared only if the entire 95% interval
lies inside that margin.

## Results

### Main transportability result

| Target | Pooled associative accuracy | Adjusted associative accuracy | Modular causal accuracy |
|---|---:|---:|---:|
| Reference | 80.23% | 81.22% | 81.52% |
| Mild | 78.84% | 80.58% | 80.90% |
| Moderate | 76.66% | 79.18% | 79.59% |
| Strong | 73.37% | 77.14% | 77.60% |

At the strong shift, modular causal accuracy exceeded:

- pooled associative accuracy by 4.23 percentage points (95% MC CI,
  4.11 to 4.36);
- target-calibrated associative accuracy by 0.47 points (0.38 to 0.55);
- frozen causal accuracy by 1.42 points (1.33 to 1.52).

From the reference target to the strong-shift target, accuracy fell by:

- 6.86 points for the pooled associative model;
- 4.08 points for the target-calibrated associative model;
- 5.42 points for the frozen causal model;
- 3.92 points for the modular causal model;
- 3.83 points for the target oracle.

The modular causal model therefore preserved 2.94 more points of accuracy than
the pooled model, but only 0.17 more points than the well-specified adjusted
associative model.

### Confounded-looking subgroup

At the strong shift, false atrial classification among renal-impaired patients
with a true competing mechanism was:

- 44.73% for the pooled associative model;
- 23.37% for the target-calibrated associative model;
- 36.60% for the frozen causal model;
- 22.52% for the modular causal model;
- 22.15% for the target oracle.

The modular causal model reduced this error by 22.21 points versus the pooled
model (95% MC CI, 21.67 to 22.75) and by 0.85 points versus the adjusted
associative control (0.37 to 1.33).

### Exact identical-distribution negative control

When all source and target hospitals shared identical generating parameters,
accuracy was 81.24% for the target-calibrated associative model and 81.52% for
the modular causal model. The paired difference was 0.28 points (95% MC CI,
0.23 to 0.33), entirely within the prespecified \(\pm1\)-point equivalence
margin. The negative control therefore passed.

### One-factor ablations

Accuracy change from the paired reference target:

| Strong component changed alone | Pooled associative | Adjusted associative | Modular causal | Oracle |
|---|---:|---:|---:|---:|
| Kidney pathway | -3.90 pp | -0.13 pp | -0.03 pp | +0.05 pp |
| Inflammation prevalence | -0.35 pp | -0.06 pp | +0.01 pp | +0.03 pp |
| Known assay calibration | 0.00 pp | 0.00 pp | 0.00 pp | 0.00 pp |
| Missingness | -4.26 pp | -4.34 pp | -4.35 pp | -4.37 pp |
| Combined strong shift | -6.86 pp | -4.08 pp | -3.92 pp | -3.83 pp |

The kidney-path shift is the component for which structural adjustment matters.
Missingness removes information and harms even the oracle; no modeling choice
can reconstruct biomarkers that were never measured. The zero assay-only
effect confirms exact use of known calibration metadata.

## Validation

- 500 repeats completed for every main target and every ablation.
- All 1,500 source latent fits converged in the main experiment.
- All validation checks passed.
- No fitting interface accepts the true mechanism label.
- Assay inversion was exact to numerical tolerance
  (maximum error \(8.88\times10^{-16}\)).
- In the strong target, the mean target-calibration bias was -0.020 SD for the
  renal path and +0.010 SD for the inflammation path.
- The strong target had 26.67% biomarker missingness and 1.84% of patients with
  all three biomarkers missing.

## Interpretation

The experiment supports the transportability hypothesis under its stated
data-generating assumptions. A modular causal model preserved hidden-mechanism
recovery substantially better than a transport-naive pooled latent class
model. It also produced a small, precisely estimated improvement over a
well-specified associative model given the same nuisance variables and
unlabeled target calibration data.

The result does not show that causality generally dominates association. In
fact, the adjusted associative model nearly matched the causal model. The
stronger conclusion is narrower and defensible: correct exclusion restrictions
can reduce the variance and spurious adjustment of a flexible model while
making target recalibration modular. The frozen causal model's degradation
shows that a correct graph is not enough; changing nuisance coefficients still
have to be measured and updated.

## Boundaries and next sensitivity analyses

This is a proof-of-principle simulation, not clinical validation. It assumes:

- the causal graph is correct;
- renal dysfunction and inflammation are observed without measurement error;
- both are independent of the hidden mechanism;
- the target supplies 150 unlabeled calibration patients;
- laboratory calibration metadata are known;
- the two endotypes are categorical and identifiable through stable markers;
- missingness is ignorable conditional on the observed design.

The next version should vary one assumption at a time:

1. graph misspecification, including a weak forbidden path;
2. noisy or incompletely observed renal and inflammation variables;
3. target calibration sizes of 0, 50, 150, and 500;
4. unknown assay drift rather than known metadata;
5. mechanism-prevalence shift;
6. correlated residual biomarkers and nonlinear nuisance effects;
7. missing-not-at-random mechanisms;
8. a K=1 null under cross-hospital shift.

These are falsification and robustness tests. They should not be folded into the
preliminary figure until the clean primary result is presented.

