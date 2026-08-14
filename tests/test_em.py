"""Contracts for the single missingness-aware EM implementation."""

from __future__ import annotations

import numpy as np

from traceesus.core.em import (
    conditional_e_step,
    conditional_m_step,
    e_step,
    m_step,
)


def test_complete_data_is_exact_all_observed_mask_case() -> None:
    """Require implicit complete data and an explicit all-ones mask to be identical."""

    rng = np.random.default_rng(26_081_303)
    biomarkers = rng.normal(size=(127, 3))
    responsibility = rng.dirichlet((2.0, 2.0), size=127)
    mask = np.ones_like(biomarkers, dtype=bool)
    implicit = m_step(biomarkers, responsibility, 0.0025, 0.02)
    explicit = m_step(
        biomarkers,
        responsibility,
        0.0025,
        0.02,
        measurement_mask=mask,
    )
    np.testing.assert_array_equal(implicit.class_means, explicit.class_means)
    np.testing.assert_array_equal(implicit.nuisance_effects, explicit.nuisance_effects)
    np.testing.assert_array_equal(implicit.variance, explicit.variance)

    log_prior = np.log(np.asarray((0.45, 0.55)))[None, :]
    posterior_implicit = e_step(
        biomarkers, log_prior, implicit.class_means, implicit.variance
    )
    posterior_explicit = e_step(
        biomarkers,
        log_prior,
        implicit.class_means,
        implicit.variance,
        measurement_mask=mask,
    )
    np.testing.assert_array_equal(posterior_implicit[0], posterior_explicit[0])
    assert posterior_implicit[1] == posterior_explicit[1]


def test_unobserved_values_do_not_affect_e_or_m_step() -> None:
    """Prove masked cells are excluded from both shared numerical primitives."""

    rng = np.random.default_rng(71)
    biomarkers = rng.normal(size=(80, 3))
    mask = rng.random(size=biomarkers.shape) > 0.20
    responsibility = rng.dirichlet((3.0, 2.0), size=80)
    hidden_a = biomarkers.copy()
    hidden_b = biomarkers.copy()
    hidden_a[~mask] = 1_000.0
    hidden_b[~mask] = -1_000.0
    fit_a = m_step(hidden_a, responsibility, 0.0025, 0.02, mask)
    fit_b = m_step(hidden_b, responsibility, 0.0025, 0.02, mask)
    np.testing.assert_array_equal(fit_a.class_means, fit_b.class_means)
    np.testing.assert_array_equal(fit_a.variance, fit_b.variance)
    prior = np.log(np.asarray((0.5, 0.5)))[None, :]
    posterior_a = e_step(hidden_a, prior, fit_a.class_means, fit_a.variance, mask)
    posterior_b = e_step(hidden_b, prior, fit_a.class_means, fit_a.variance, mask)
    np.testing.assert_array_equal(posterior_a[0], posterior_b[0])
    assert posterior_a[1] == posterior_b[1]


def test_conditional_complete_data_is_exact_all_observed_mask_case() -> None:
    """Require the conditional path to preserve the general missingness contract."""

    rng = np.random.default_rng(26_081_314)
    biomarkers = rng.normal(size=(101, 3))
    renal = np.tile((0, 1), 51)[:101]
    nuisance = np.column_stack((renal, rng.binomial(1, 0.1, size=101)))
    paths = np.asarray(((True, True, True), (False, True, False)))
    responsibility = rng.dirichlet((2.0, 2.0), size=101)
    observed = np.ones_like(biomarkers, dtype=bool)
    implicit = conditional_m_step(
        biomarkers,
        renal,
        nuisance,
        paths,
        responsibility,
        0.0025,
        0.02,
        0.5,
        1e-6,
    )
    explicit = conditional_m_step(
        biomarkers,
        renal,
        nuisance,
        paths,
        responsibility,
        0.0025,
        0.02,
        0.5,
        1e-6,
        observed,
    )
    np.testing.assert_array_equal(
        implicit.class_probability_by_renal,
        explicit.class_probability_by_renal,
    )
    np.testing.assert_array_equal(
        implicit.emission.class_means, explicit.emission.class_means
    )
    np.testing.assert_array_equal(
        implicit.emission.nuisance_effects, explicit.emission.nuisance_effects
    )
    np.testing.assert_array_equal(implicit.emission.variance, explicit.emission.variance)
    posterior_implicit = conditional_e_step(
        biomarkers, renal, nuisance, paths, implicit
    )
    posterior_explicit = conditional_e_step(
        biomarkers, renal, nuisance, paths, implicit, observed
    )
    np.testing.assert_array_equal(posterior_implicit[0], posterior_explicit[0])
    assert posterior_implicit[1] == posterior_explicit[1]


def test_unused_nuisance_column_reduces_exactly_to_q1() -> None:
    """Prove a masked zero nuisance column does not alter the q=1 configuration."""

    rng = np.random.default_rng(26_081_315)
    biomarkers = rng.normal(size=(120, 3))
    renal = np.tile((0, 1), 60)
    responsibility = rng.dirichlet((2.0, 3.0), size=120)
    q1_design = renal[:, None].astype(float)
    q2_design = np.column_stack((q1_design, np.zeros(120)))
    q1_paths = np.ones((1, 3), dtype=bool)
    q2_paths = np.vstack((q1_paths, np.zeros((1, 3), dtype=bool)))
    q1 = conditional_m_step(
        biomarkers,
        renal,
        q1_design,
        q1_paths,
        responsibility,
        0.0025,
        0.02,
        0.5,
        1e-6,
    )
    q2 = conditional_m_step(
        biomarkers,
        renal,
        q2_design,
        q2_paths,
        responsibility,
        0.0025,
        0.02,
        0.5,
        1e-6,
    )
    np.testing.assert_array_equal(
        q1.class_probability_by_renal, q2.class_probability_by_renal
    )
    np.testing.assert_array_equal(q1.emission.class_means, q2.emission.class_means)
    np.testing.assert_array_equal(
        q1.emission.nuisance_effects, q2.emission.nuisance_effects[:1]
    )
    np.testing.assert_array_equal(q1.emission.variance, q2.emission.variance)
    q1_posterior = conditional_e_step(biomarkers, renal, q1_design, q1_paths, q1)
    q2_posterior = conditional_e_step(biomarkers, renal, q2_design, q2_paths, q2)
    np.testing.assert_array_equal(q1_posterior[0], q2_posterior[0])
    assert q1_posterior[1] == q2_posterior[1]
