"""Exact Monte Carlo summaries, paired contrasts, Wilson intervals, and BIC."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, t


TARGET_MONTE_CARLO_STANDARD_ERROR = 0.005


def required_repeats_for_target_mcse(pilot_variance: float) -> int:
    """Return repeats needed for the single prespecified Monte Carlo SE target."""

    if not np.isfinite(pilot_variance) or pilot_variance < 0.0:
        raise ValueError("pilot_variance must be finite and nonnegative.")
    return max(
        2,
        math.ceil(pilot_variance / TARGET_MONTE_CARLO_STANDARD_ERROR**2),
    )


def monte_carlo_summary(values: np.ndarray) -> dict[str, float | int]:
    """Preserve the legacy t-CI and empirical-quantile operation order."""

    values = np.asarray(values, dtype=float)
    repeat_count = values.size
    mean = float(np.mean(values))
    sample_sd = float(np.std(values, ddof=1))
    standard_error = sample_sd / math.sqrt(repeat_count)
    critical_value = float(t.ppf(0.975, df=repeat_count - 1))
    return {
        "repeat_count": repeat_count,
        "mean": mean,
        "sample_sd": sample_sd,
        "monte_carlo_standard_error": standard_error,
        "ci95_low": mean - critical_value * standard_error,
        "ci95_high": mean + critical_value * standard_error,
        "repeat_quantile_2_5": float(np.quantile(values, 0.025)),
        "repeat_quantile_97_5": float(np.quantile(values, 0.975)),
    }


def bounded_rate_monte_carlo_summary(
    values: np.ndarray,
) -> dict[str, float | int]:
    """Summarize a repeated rate with the legacy clipped t interval.

    The known-SCM experiment uses different public column names from latent
    discovery but shares this exact scalar reduction. Keeping the reduction
    here prevents an ostensibly cosmetic rewrite from changing quantile,
    degrees-of-freedom, or clipping semantics.
    """

    values = np.asarray(values, dtype=float)
    n_repeats = values.size
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    standard_error = standard_deviation / np.sqrt(n_repeats)
    critical_value = float(t.ppf(0.975, df=n_repeats - 1))
    return {
        "n_repeats": n_repeats,
        "mean": mean,
        "sd_across_repeats": standard_deviation,
        "monte_carlo_se": standard_error,
        "ci_low": max(0.0, mean - critical_value * standard_error),
        "ci_high": min(1.0, mean + critical_value * standard_error),
        "repeat_q025": float(np.quantile(values, 0.025)),
        "repeat_q975": float(np.quantile(values, 0.975)),
    }


def paired_rate_contrast(difference: object) -> dict[str, float | int]:
    """Return the legacy unbounded paired t interval for a pandas Series.

    ``Series.mean`` and ``Series.std`` are invoked deliberately.  The cited
    pipelines used pandas reductions here, so replacing them with NumPy would
    be an unnecessary exactness risk even when no values are missing.
    """

    n_repeats = int(getattr(difference, "size"))
    mean = float(getattr(difference, "mean")())
    standard_error = float(
        getattr(difference, "std")(ddof=1) / np.sqrt(n_repeats)
    )
    critical_value = float(t.ppf(0.975, df=n_repeats - 1))
    return {
        "n_repeats": n_repeats,
        "mean_difference": mean,
        "ci_low": mean - critical_value * standard_error,
        "ci_high": mean + critical_value * standard_error,
    }


def paired_mean_contrast(difference: np.ndarray) -> dict[str, float | int]:
    """Summarize a precomputed within-repeat difference without reordering it."""

    difference = np.asarray(difference, dtype=float)
    mean = float(np.mean(difference))
    sample_sd = float(np.std(difference, ddof=1))
    standard_error = sample_sd / math.sqrt(difference.size)
    critical_value = float(t.ppf(0.975, df=difference.size - 1))
    return {
        "repeat_count": difference.size,
        "mean_difference": mean,
        "ci95_low": mean - critical_value * standard_error,
        "ci95_high": mean + critical_value * standard_error,
    }


def paired_nanmean_contrast(difference: np.ndarray) -> dict[str, float | int]:
    """Summarize a contrast after the transport kernel's finite-value filter.

    The mean is intentionally computed with ``nanmean`` before filtering, then
    the SD, standard error, and degrees of freedom use the finite subset.  This
    slightly unusual order is part of the locked transportability artifacts.
    """

    difference = np.asarray(difference, dtype=float)
    mean = float(np.nanmean(difference))
    valid = difference[np.isfinite(difference)]
    sample_sd = float(np.std(valid, ddof=1))
    standard_error = sample_sd / math.sqrt(valid.size)
    critical_value = float(t.ppf(0.975, df=valid.size - 1))
    return {
        "repeat_count": valid.size,
        "mean_difference": mean,
        "ci95_low": mean - critical_value * standard_error,
        "ci95_high": mean + critical_value * standard_error,
    }


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return the same untruncated 95% Wilson interval used by latent discovery."""

    z_value = float(norm.ppf(0.975))
    proportion = successes / trials
    denominator = 1.0 + z_value**2 / trials
    center = (proportion + z_value**2 / (2.0 * trials)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_value**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def bic(log_likelihood: float, parameter_count: int, patient_count: int) -> float:
    """Compute BIC without changing the historical scalar expression."""

    return -2.0 * log_likelihood + parameter_count * math.log(patient_count)
