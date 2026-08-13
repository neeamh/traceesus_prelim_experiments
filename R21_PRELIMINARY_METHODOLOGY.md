# R21 preliminary experiment: hidden-label mechanism recovery

## The hypothesis

> A biologically constrained latent structural causal model can recover the
> categorical mechanism responsible for an observed biomarker profile, even
> when a nonmechanistic pathway produces a misleading biomarker elevation.

This experiment tests **latent endotyping**, not supervised classification.
The simulator stores each patient's true mechanism as an answer key, but no fit
function receives those labels.

## Plain-language experiment

Each artificial patient has exactly one true mechanism:

- atrial;
- competing.

Posterior probabilities express uncertainty about that single mechanism. They
do not represent fractional biological membership.

Thirty percent of patients have renal dysfunction. Renal dysfunction directly
raises an NT-proBNP-like marker but does not change the true mechanism in this
first experiment. Thus, a competing-mechanism patient can have a strongly
atrial-looking NT-proBNP-like value.

Strictly, renal dysfunction is an **alternative cause of the biomarker**, not a
classical common-cause confounder in this version, because it is generated
independently of the true mechanism. “Renal biomarker distortion” is the
technically correct description.

## Data-generating structural model

Let:

- \(Z_i\in\{A,C\}\) be the unobserved categorical mechanism;
- \(R_i\in\{0,1\}\) indicate observed renal dysfunction;
- \(\mathbf B_i=(B_{\mathrm{NT}},B_{\mathrm{ECG}},B_{\mathrm{COMP}})\)
  be three continuous standardized biomarkers.

The structural biomarker equation is:

\[
\mathbf B_i =
\boldsymbol\lambda_{Z_i}
+ ( \gamma R_i,\ 0,\ 0 )
+ \boldsymbol\epsilon_i,
\qquad
\boldsymbol\epsilon_i\sim N(\mathbf 0,I_3).
\]

Prespecified mechanism signatures:

\[
\boldsymbol\lambda_A=(1.25,\ 1.00,\ 0.00), \qquad
\boldsymbol\lambda_C=(0.00,\ 0.00,\ 1.00).
\]

Other parameters:

- \(P(R=1)=0.30\);
- \(P(Z=A\mid R=0)=P(Z=A\mid R=1)=0.50\);
- independent biomarker residual SDs of 1.00;
- renal effect \(\gamma\in\{0.00,0.50,1.00,1.50\}\) residual SD.

The effects are named and expressed on one standardized scale. The 1.50-SD
“strong” renal path slightly exceeds the atrial mechanism's 1.25-SD path to the
misleading marker. The Bayes error remains nonzero because biomarker
distributions overlap.

## Hidden-label train/test design

At each renal-effect level:

1. Generate 800 unlabeled training patients.
2. Fit every latent model to the same observed renal status and biomarkers.
3. Generate an independent 1,000-patient test cohort.
4. Infer posterior mechanism probabilities in the test cohort.
5. Compare predictions with simulator truth only after fitting.
6. Repeat the complete process 500 times.

Randomness is controlled by master seed 20260728 and independent NumPy
`SeedSequence` streams. Every fit uses four prespecified EM starts. A
nonconverged fit is automatically retried with eight starts and up to 1,200
iterations.

## Models

### 1. Associative latent class model: primary comparator

The standard two-class latent profile model factorizes:

\[
p(Z)\,p(R\mid Z)\prod_j p(B_j\mid Z).
\]

It sees the same renal-status variable as the causal model, but treats renal
status and biomarker values as indicators of class resemblance. It does not
encode a direct \(R\rightarrow B_{\mathrm{NT}}\) path.

It has 12 free parameters:

- one class probability;
- two class-specific renal probabilities;
- six class-specific biomarker means;
- three shared residual variances.

### 2. Biologically constrained latent SCM: primary method

The causal latent model factorizes:

\[
p(Z\mid R)\prod_j p(B_j\mid Z,R),
\]

with a prespecified direct renal path only to the NT-proBNP-like biomarker.
The latent mechanism and structural parameters are estimated jointly by EM;
no endotype labels are supplied.

It also has 12 free parameters:

- two atrial probabilities, one in each renal stratum;
- six mechanism-specific biomarker means;
- one renal-to-NT-proBNP-like path coefficient;
- three shared residual variances.

The primary comparison is therefore matched on observed inputs and parameter
count.

### 3. Renal-adjusted associative latent class model: fairness control

This conditional latent class regression allows a shared renal association for
**every** biomarker rather than imposing the biological exclusion restriction.
It has 14 parameters. It is a stringent control for the possibility that the
causal model wins merely because it accounts for renal status correctly.

### 4. Data-generating oracle: reference ceiling

The oracle uses the true simulator parameters. It is not a fitted competitor.
It quantifies irreducible overlap and the maximum classification performance
available under the generated world.

## Label orientation without truth leakage

Latent classes have arbitrary numeric labels. After each fit, the class with the
larger prespecified anchor contrast

\[
\frac{B_{\mathrm{ECG}}}{\sigma_{\mathrm{ECG}}}
-
\frac{B_{\mathrm{COMP}}}{\sigma_{\mathrm{COMP}}}
\]

is named “atrial.” The misleading NT-proBNP-like marker and the simulator's
truth labels are excluded from this orientation rule.

## Prespecified outcomes

Primary:

- top-1 true-mechanism ranking accuracy in the independent test cohort.

Key subgroup:

- false atrial classification among renal-impaired patients whose true
  mechanism is competing.

Supporting:

- adjusted Rand index;
- Brier score for the atrial posterior;
- expected calibration error;
- posterior entropy;
- predicted atrial prevalence;
- EM convergence, class collapse, anchor separation, and renal-path recovery.

All comparisons are paired within repeat. For repeat-level rates, the reported
95% interval is a two-sided Monte Carlo confidence interval for the expected
rate:

\[
\bar p \pm t_{0.975,499}\frac{s_{\text{repeat}}}{\sqrt{500}}.
\]

Empirical 2.5th and 97.5th repeat quantiles are saved separately and are not
mislabelled as confidence intervals.

## K=1 null experiment

The null simulator generates 800 patients with:

- one homogeneous mechanism;
- the same 30% renal prevalence;
- a real 1.50-SD renal-to-NT-proBNP-like path;
- no latent endotype heterogeneity.

For each model family, K=1 and K=2 are fit without labels and compared by BIC.
The false-endotype discovery rate is the percentage of 500 repeats in which BIC
selects K=2. Binomial uncertainty is reported with Wilson 95% intervals.

This is a genuine null challenge. Merely fixing K=1 would not test whether the
model invents endotypes.

## Results

### Recovery under strong renal distortion

| Model | Accuracy, mean (95% MC CI) | False atrial in renal/competing subgroup |
|---|---:|---:|
| Associative latent class model | 57.85% (57.33–58.37) | 75.99% (73.23–78.76) |
| Renal-adjusted associative latent class model | 81.64% (81.50–81.78) | 19.75% (18.84–20.66) |
| Biologically constrained latent SCM | 81.94% (81.81–82.06) | 18.51% (17.92–19.11) |
| Data-generating oracle | 82.70% (82.59–82.81) | 17.35% (17.07–17.62) |

Compared with the primary associative latent class model, the causal model:

- improved accuracy by 24.08 percentage points
  (95% MC CI 23.55–24.62);
- reduced false atrial classification by 57.48 points
  (54.77–60.19);
- improved adjusted Rand index by 0.371
  (0.366–0.375);
- reduced Brier score by 0.221
  (0.214–0.229).

Compared with the renal-adjusted associative control, the causal model:

- improved accuracy by 0.30 points (0.22–0.37);
- reduced false atrial classification by 1.23 points (0.53–1.94).

The causal model estimated the strong renal path as 1.494 SD on average, versus
the true 1.500 SD.

At zero renal distortion, the primary associative and causal models were
essentially equivalent: 82.08% versus 82.03% accuracy.

### K=1 null

| Model family | False K=2 selections | Rate (Wilson 95% CI) |
|---|---:|---:|
| Associative latent class model | 500/500 | 100.00% (99.24–100.00) |
| Renal-adjusted associative latent class model | 0/500 | 0.00% (0.00–0.76) |
| Biologically constrained latent SCM | 0/500 | 0.00% (0.00–0.76) |

All 6,000 recovery fits and all 1,500 null K=2 fits converged after automatic
refitting. No fitted class fell below 25.2% effective prevalence.

## What the result means

The experiment supports the stated hypothesis **inside this generated world**:
the biologically constrained latent SCM recovered a hidden categorical
mechanism without training labels and remained close to the oracle when a renal
path made one marker misleading.

The standard associative latent class model increasingly organized patients by
the dominant renal-driven biomarker pattern rather than by the true mechanism.
Under the K=1 null it interpreted the real renal-to-biomarker dependency as
latent heterogeneity and selected two classes in every repeat.

The adjusted associative control is the crucial boundary on the claim. Once an
associative latent model was explicitly allowed to adjust biomarkers for renal
status, it performed almost as well as the causal model and also passed the
K=1 null. Therefore the experiment does **not** prove that causal models
generally beat all associative latent-variable models. It shows that correct
biological structure can protect latent recovery against a specific
nonmechanistic biomarker pathway, with a small efficiency gain over a more
flexible adjusted model when the structural restriction is correct.

## Required limitations

- This is synthetic evidence, not clinical validation.
- The renal variable is observed and measured without error.
- The structural graph is correctly specified by construction.
- Biomarker equations are linear Gaussian with independent residuals.
- The main recovery experiment prespecifies K=2.
- The biological anchor used to name the atrial class is assumed valid.
- A wrong renal-path restriction could remove the causal model's advantage;
  graph-misspecification sensitivity is a next experiment.
- The null BIC test is useful but finite-mixture model selection is nonregular;
  later work should add parametric-bootstrap and stability checks.

## Reproduction

```bash
python -m pip install -r requirements.txt
python r21_latent_endotyping_experiment.py \
  --repeats 500 \
  --null-repeats 500 \
  --workers 4 \
  --output-dir outputs_latent_endotyping
```

The executable notebook is `R21_latent_endotyping_experiment.ipynb`.
