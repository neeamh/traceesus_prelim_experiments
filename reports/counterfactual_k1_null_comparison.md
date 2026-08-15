# K=1 null comparison before counterfactual archival

The two K=1 controls are genuinely different tests, not duplicate implementations.

The retired counterfactual control first subtracts the *known* renal contribution from a 1,000-patient cohort generated at 2.25 SD, then compares one unconstrained diagonal Gaussian with a two-component diagonal Gaussian. It reports one model-selection result per repeat, uses five K=2 starts and component-specific variances, requires a minimum component weight of 0.10, and uses the seed root `seed + 1_000_003`.

Endotype discovery instead uses an 800-patient cohort at 1.50 SD and evaluates three separate observed-data factorizations: pooled associative, renal-adjusted conditional, and one-path biology-constrained conditional. Their K=1 likelihoods and parameter counts differ, their K=2 fits use the discovery EM configuration and shared variance, and their seed root is `master_seed + 91_337`. The discovery null therefore cannot replace the archived residualized known-path control; the archived code and `outputs_locked/outputs/` retain that distinct result as provenance.
