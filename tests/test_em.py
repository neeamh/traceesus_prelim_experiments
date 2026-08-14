"""Contracts for the single missingness-aware EM implementation."""

from __future__ import annotations

import numpy as np

from traceesus.core.em import e_step, m_step


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
