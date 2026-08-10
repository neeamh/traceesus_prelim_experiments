"""Presentation figures for the renal x heart-failure grid.

Every function takes the repeat-level grid frame, reduces it internally with
t-based 95% intervals, and writes PNG + PDF.  Nothing here re-runs a model:
figures regenerate from saved CSVs alone.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import chi2, t

ACCENT = "#1C3F5F"       # method under test
ACCENT_CF = "#C25A2E"    # counterfactual query
BASELINE_GRAY = "#9AA4AD"

# Mechanism colours for patient-level figures.  Deliberately the same two hues
# as the method accents so the whole deck reads as one palette.
MECHANISM_COLORS = {"atrial": "#1C3F5F", "competing": "#C25A2E"}
# Deliberately unlike the mechanism hues: a discovered class is not a mechanism,
# and the colours should not invite the reader to assume it is.
DISCOVERED_COLORS = {"class A": "#5B4B8A", "class B": "#3F7D6E"}
SCATTER_FACET_WORDS = {
    0.0: "No renal distortion",
    0.75: "Moderate renal distortion",
    1.5: "Strong renal distortion",
}
PROFILE_ORDER = ("uncomplicated", "renal_only", "heart_failure_only", "redundant")
PROFILE_LABELS = {
    "uncomplicated": "Uncomplicated",
    "renal_only": "Renal only",
    "heart_failure_only": "HF only",
    "redundant": "Redundant\n(renal + HF)",
}


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    half = float(t.ppf(0.975, values.size - 1)) * float(np.std(values, ddof=1)) / math.sqrt(values.size)
    return mean, mean - half, mean + half


def _save(figure: plt.Figure, path: Path, source: str) -> None:
    # Slightly below the axes box: ``bbox_inches="tight"`` still captures it,
    # and it stops the provenance note landing on top of an x-axis label.
    figure.text(0.005, -0.02, source, fontsize=6, color="#8A939B", style="italic")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


# Manuscript Figure 2 line styles.  The associative family is warm and broken;
# the fully adjusted and causal models are cool and solid.  A reader sees the
# grouping before reading a single label.
RECOVERY_LINE_STYLES: dict[str, dict[str, object]] = {
    "Associative latent class model": {
        "color": "#C25A2E", "linestyle": "--", "marker": "s",
    },
    "Renal-adjusted associative latent class model": {
        "color": "#D18A3D", "linestyle": "--", "marker": "^",
    },
    "Two-nuisance adjusted associative latent model": {
        "color": "#6E8A9C", "linestyle": "-.", "marker": "D",
    },
    "Two-path biologically constrained latent SCM": {
        "color": "#1C3F5F", "linestyle": "-", "marker": "o", "filled": True,
    },
}
RECOVERY_SHORT_NAMES = {
    "Associative latent class model": "Pooled associative LCM",
    "Renal-adjusted associative latent class model": "Renal-adjusted associative LCM",
    "Two-nuisance adjusted associative latent model": "Two-nuisance adjusted LCM",
    "Two-path biologically constrained latent SCM": "Two-path biology-constrained SCM",
}


def _place_end_labels(
    axis,
    endpoints: list[tuple[float, str, str]],
    x_position: float,
    *,
    gap_fraction: float = 0.062,
    fontsize: float = 8.5,
) -> None:
    """Write value labels at the right edge, pushed apart so none overlap."""

    low_limit, high_limit = axis.get_ylim()
    gap = gap_fraction * (high_limit - low_limit)
    placed = -np.inf
    for value, text, color in sorted(endpoints):
        target = max(value, placed + gap)
        placed = target
        axis.annotate(
            text, (x_position, value),
            xytext=(x_position + 0.08, target), textcoords="data",
            fontsize=fontsize, fontweight="bold", color=color, va="center",
            arrowprops={
                "arrowstyle": "-", "color": color, "linewidth": 0.6,
                "shrinkA": 1.0, "shrinkB": 1.0,
            } if abs(target - value) > 0.25 * gap else None,
        )


def plot_recovery_lines(
    raw: pd.DataFrame,
    path: Path,
    *,
    methods: tuple[str, ...] | None = None,
    hf_effect_sd: float = 1.5,
    level_column: str = "renal_effect_sd",
    level_labels: tuple[str, ...] = ("None", "Weak", "Moderate", "Strong"),
    accuracy_metric: str = "accuracy",
    false_atrial_metric: str = "false_atrial__redundant",
    panel_a_title: str = "True-mechanism ranking accuracy",
    panel_b_title: str = "False atrial calls, redundant subgroup",
    figure_number: str = "Figure 2",
    title: str = "Hidden-mechanism recovery under renal biomarker distortion",
    caption: str = "",
    show_intervals: bool = True,
    source: str = "",
) -> None:
    """Two-panel accuracy / false-attribution lines across renal distortion.

    Direct successor to the locked Figure 2, with the heart-failure path held
    on and the two adjustment tiers the single-nuisance figure could not show.
    Both panels are percentages on a common 0-100 axis so the two failure modes
    are read at the same visual scale.
    """

    methods = methods or tuple(RECOVERY_LINE_STYLES)
    block = raw[raw["heart_failure_effect_sd"] == hf_effect_sd]
    levels = sorted(block[level_column].unique())
    positions = np.arange(len(levels), dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    panels = (
        (axes[0], "a", panel_a_title, accuracy_metric, "Patients correctly assigned (%)"),
        (axes[1], "b", panel_b_title, false_atrial_metric, "Called atrial (%)"),
    )

    for axis, letter, panel_title, metric, y_label in panels:
        endpoints: list[tuple[float, str, str]] = []
        for method in methods:
            style = dict(RECOVERY_LINE_STYLES.get(method, {}))
            filled = bool(style.pop("filled", False))
            color = str(style.get("color", BASELINE_GRAY))
            means, halfwidths = [], []
            for level in levels:
                cell = block[
                    (block["method"] == method) & (block[level_column] == level)
                ][metric].to_numpy(dtype=float)
                mean, low, _ = _mean_ci(cell)
                means.append(100.0 * mean)
                halfwidths.append(100.0 * (mean - low) if np.isfinite(low) else 0.0)
            axis.errorbar(
                positions, means,
                yerr=halfwidths if show_intervals else None,
                linewidth=1.9, markersize=6, capsize=2.5,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color, markeredgewidth=1.4,
                elinewidth=0.8, ecolor=color, **style,
            )
            if np.isfinite(means[-1]):
                endpoints.append((means[-1], f"{means[-1]:.1f}", color))

        axis.set_ylim(0, 100)
        axis.set_xlim(-0.25, len(levels) - 0.35)
        axis.set_xticks(positions, level_labels[: len(levels)])
        axis.set_xlabel("Direct renal effect on NT-proBNP")
        axis.set_ylabel(y_label)
        axis.set_title(panel_title, fontsize=11, fontweight="bold", pad=12)
        axis.yaxis.grid(True, color="#DCE1E5", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.16, 1.06, letter, transform=axis.transAxes,
            fontsize=13, fontweight="bold", va="top",
        )
        _place_end_labels(axis, endpoints, positions[-1])

    handles = []
    for method in methods:
        style = dict(RECOVERY_LINE_STYLES.get(method, {}))
        filled = bool(style.pop("filled", False))
        color = str(style.get("color", BASELINE_GRAY))
        handles.append(plt.Line2D(
            [], [], linewidth=1.9, markersize=6,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color, markeredgewidth=1.4,
            label=RECOVERY_SHORT_NAMES.get(method, method), **style,
        ))
    figure.legend(
        handles=handles, loc="lower center", ncol=2,
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.14),
    )
    # Mathtext bold for the figure number only, so the heading reads like the
    # manuscript's.  Spaces must be escaped inside a mathtext group.
    bold_number = figure_number.replace(" ", "\\ ")
    figure.suptitle(
        f"$\\bf{{{bold_number}.}}$ {title}",
        x=0.012, ha="left", fontsize=13, y=1.10,
    )
    figure.text(0.012, 1.015, caption, fontsize=8.5, color="#8A939B", ha="left")
    figure.tight_layout()
    _save(figure, path, source)


DRIFT_COLORS = {"mechanism": "#1C3F5F", "kidney": "#D18A3D"}
# Stack order runs correct-and-clean at the bottom to false-attribution on top,
# so the growing band is the one a clinician should worry about.
COMPOSITION_STACK = (
    ("composition__atrial_normal_kidneys", "Atrial, kidneys normal", "#1C3F5F"),
    ("composition__atrial_impaired_kidneys", "Atrial, kidneys impaired", "#6E8A9C"),
    ("composition__competing_normal_kidneys", "Competing, kidneys normal", "#E3B08A"),
    ("composition__competing_impaired_kidneys", "Competing, kidneys impaired", "#C25A2E"),
)


def plot_identity_drift(
    drift: pd.DataFrame,
    path: Path,
    *,
    title: str = "What the pooled model's latent class comes to mean",
    subtitle: str = "",
    source: str = "",
) -> None:
    """Show a latent class losing its mechanism identity and gaining a renal one.

    Panel a plots how well the unadjusted associative model's split matches the
    true mechanism versus kidney status, as renal distortion grows.  The two
    curves cross: past that point the class the model would read as
    "atrial-like" is better described as "renally impaired".

    Panel b opens that class up.  It is not that the model finds nothing — the
    class stays large and confident.  It fills with competing-mechanism
    patients who happen to have bad kidneys, which is precisely the false
    attribution the score has to avoid.
    """

    levels = sorted(drift["renal_effect_sd"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))

    curves: dict[str, list[float]] = {}
    for key, column, label in (
        ("mechanism", "agreement_with_mechanism", "Matches the true mechanism"),
        ("kidney", "agreement_with_kidney_status", "Matches kidney status"),
    ):
        means, lows, highs = [], [], []
        for level in levels:
            mean, low, high = _mean_ci(
                drift[drift["renal_effect_sd"] == level][column].to_numpy(float)
            )
            means.append(100.0 * mean)
            lows.append(100.0 * low)
            highs.append(100.0 * high)
        curves[key] = means
        color = DRIFT_COLORS[key]
        axes[0].plot(levels, means, color=color, linewidth=2.3, marker="o", markersize=5)
        axes[0].fill_between(levels, lows, highs, color=color, alpha=0.16, linewidth=0)
        axes[0].annotate(
            label, (levels[-1], means[-1]), xytext=(6, 0), textcoords="offset points",
            fontsize=9, fontweight="bold", color=color, va="center",
        )

    # Where the class stops being a mechanism and starts being a comorbidity.
    difference = np.asarray(curves["kidney"]) - np.asarray(curves["mechanism"])
    crossings = np.nonzero(np.diff(np.sign(difference)))[0]
    if crossings.size:
        index = int(crossings[0])
        span = difference[index + 1] - difference[index]
        weight = 0.0 if span == 0 else -difference[index] / span
        crossover = levels[index] + weight * (levels[index + 1] - levels[index])
        axes[0].axvline(crossover, color="#42474D", linewidth=0.9, linestyle=":")
        axes[0].annotate(
            f"the class changes meaning\nat {crossover:.2f} SD",
            (crossover, 52), xytext=(6, 0), textcoords="offset points",
            fontsize=8, color="#42474D", va="bottom",
        )

    axes[0].axhline(50, color="#B7BEC4", linewidth=0.8, linestyle="--")
    axes[0].set_ylim(40, 102)
    axes[0].set_xlim(levels[0], levels[-1] + 0.62)
    axes[0].set_xlabel("Direct renal effect on NT-proBNP (SD)")
    axes[0].set_ylabel("Agreement with the discovered split (%)")
    axes[0].set_title("a   Identity of the latent class", fontsize=11,
                      fontweight="bold", loc="left")
    axes[0].yaxis.grid(True, color="#DCE1E5", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)

    stacks, labels, colors = [], [], []
    for column, label, color in COMPOSITION_STACK:
        stacks.append([
            100.0 * float(np.mean(
                drift[drift["renal_effect_sd"] == level][column].to_numpy(float)
            ))
            for level in levels
        ])
        labels.append(label)
        colors.append(color)
    axes[1].stackplot(levels, *stacks, colors=colors, edgecolor="white", linewidth=0.4)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlim(levels[0], levels[-1])
    axes[1].set_xlabel("Direct renal effect on NT-proBNP (SD)")
    axes[1].set_ylabel("Composition of that class (%)")
    axes[1].set_title(
        "b   Who is actually in the class the model calls atrial-like",
        fontsize=11, fontweight="bold", loc="left",
    )

    # Label each band in place at its thickest point.  A legend here would sit
    # on top of the data whichever corner it went in, and an in-place label
    # removes the round trip between key and figure.
    bottoms = np.zeros(len(levels))
    dark_bands = {"#1C3F5F", "#C25A2E"}
    for band, label, color in zip(stacks, labels, colors):
        band = np.asarray(band, dtype=float)
        index = int(np.argmax(band))
        if band[index] < 7.0:      # too thin to carry text legibly
            bottoms = bottoms + band
            continue
        alignment = "left" if index == 0 else "right" if index == len(levels) - 1 else "center"
        offset = 0.03 * (levels[-1] - levels[0])
        x = levels[index] + (offset if alignment == "left" else -offset if alignment == "right" else 0.0)
        axes[1].text(
            x, bottoms[index] + band[index] / 2.0, label,
            ha=alignment, va="center", fontsize=8.5, fontweight="bold",
            color="white" if color in dark_bands else "#22282E",
        )
        bottoms = bottoms + band
    axes[1].spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        title + (f"\n{subtitle}" if subtitle else ""),
        x=0.012, ha="left", fontsize=12.5, y=1.06,
    )
    figure.tight_layout()
    _save(figure, path, source)


def _moment_matched_ellipse(
    points: np.ndarray, coverage: float
) -> tuple[np.ndarray, float, float, float]:
    """Centre, width, height and rotation of a moment-matched Gaussian contour.

    The ellipse is the ``coverage``-probability contour of the bivariate normal
    whose first two moments equal the sample's.  It is a summary, not the true
    class density: marginalized over renal and heart-failure status each class
    is a mixture, so the contour is intentionally the *simplest honest* shape
    that answers "where does this cloud sit and how wide is it".
    """

    mean = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    scale = float(np.sqrt(chi2.ppf(coverage, df=2)))
    width, height = 2.0 * scale * np.sqrt(np.maximum(values, 0.0))
    return mean, float(width), float(height), angle


def _gaussian_separability(points: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy of the best quadratic rule that uses only these two markers.

    Quadratic discriminant analysis fitted on the *true* labels with empirical
    priors, scored on the same points.  This is deliberately optimistic — it is
    an upper bound on what any method could extract from this plane, which is
    exactly the quantity that makes the overlap argument concrete.
    """

    classes = np.unique(labels)
    log_scores = []
    for label in classes:
        block = points[labels == label]
        mean = block.mean(axis=0)
        covariance = np.cov(block, rowvar=False) + 1e-9 * np.eye(points.shape[1])
        deviation = points - mean
        mahalanobis = np.einsum(
            "ij,jk,ik->i", deviation, np.linalg.inv(covariance), deviation
        )
        log_scores.append(
            -0.5 * mahalanobis
            - 0.5 * float(np.log(np.linalg.det(covariance)))
            + float(np.log(block.shape[0] / points.shape[0]))
        )
    predicted = classes[np.argmax(np.vstack(log_scores), axis=0)]
    return float(np.mean(predicted == labels))


def _agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Agreement between two binary labellings, invariant to label switching."""

    match = float(np.mean(left == right))
    return max(match, 1.0 - match)


def _draw_marker_plane(
    axis,
    points: np.ndarray,
    labels: np.ndarray,
    palette: dict[str, str],
    *,
    shown: np.ndarray,
    renal: np.ndarray | None,
    coverage: tuple[float, ...],
) -> None:
    """Scatter one labelling with its moment-matched Gaussian contours."""

    for label, color in palette.items():
        member = labels == label
        if not member.any():
            continue
        if renal is None:
            groups = ((None, {"facecolor": color, "edgecolor": "none", "alpha": 0.34}),)
        else:
            groups = (
                (0, {"facecolor": color, "edgecolor": "none", "alpha": 0.34}),
                (1, {"facecolor": "none", "edgecolor": color,
                     "alpha": 0.75, "linewidths": 0.7}),
            )
        for impaired, style in groups:
            selection = shown[member[shown]] if impaired is None else shown[
                member[shown] & (renal[shown] == impaired)
            ]
            if selection.size:
                axis.scatter(
                    points[selection, 0], points[selection, 1],
                    s=13, rasterized=True, **style,
                )
        if member.sum() > 10:
            for index, level in enumerate(coverage):
                mean, width, height, angle = _moment_matched_ellipse(
                    points[member], level
                )
                axis.add_patch(Ellipse(
                    mean, width, height, angle=angle,
                    facecolor=color if index == 0 else "none",
                    edgecolor=color,
                    alpha=0.16 if index == 0 else 1.0,
                    linewidth=1.0 if index == 0 else 1.6,
                    zorder=3,
                ))
            axis.plot(
                *points[member].mean(axis=0), marker="+", color=color,
                markersize=9, markeredgewidth=1.8, zorder=4,
            )


def plot_mechanism_scatter(
    sample: pd.DataFrame,
    path: Path,
    *,
    x_column: str = "nt_probnp",
    y_column: str = "ptfv1",
    x_label: str = "NT-proBNP (SD units)",
    y_label: str = "PTFV1 (SD units)",
    facet_column: str = "renal_effect_sd",
    facet_values: tuple[float, ...] | None = None,
    coverage: tuple[float, ...] = (0.5, 0.9),
    points_per_facet: int = 900,
    mark_renal: bool = True,
    show_discovered_row: bool = True,
    title: str = "The information is in the plane; the model splits the wrong way",
    subtitle: str = "",
    source: str = "",
    seed: int = 0,
) -> None:
    """Marker plane by truth, and — below it — by what an LCM actually recovers.

    Top row colours patients by true mechanism, with moment-matched Gaussian
    contours.  The separation annotation barely moves across facets: renal
    distortion shifts *both* classes' impaired patients by the same amount, so
    it does not destroy the mechanism signal.

    The bottom row colours the identical points by the hard assignment of an
    unadjusted associative latent class model.  That is where the failure lives:
    as renal distortion grows, the loudest axis of variation stops being the
    mechanism contrast and becomes kidney status, and an unsupervised model
    follows the variance.  Showing only the top row would imply the problem is
    lost information; it is not, and the proposal should not claim it is.
    """

    facet_values = facet_values or tuple(sorted(sample[facet_column].unique()))
    has_discovered = show_discovered_row and "discovered_class" in sample
    row_count = 2 if has_discovered else 1
    rng = np.random.default_rng(seed)

    figure, axes = plt.subplots(
        row_count, len(facet_values),
        figsize=(4.0 * len(facet_values), 4.3 * row_count),
        sharex=True, sharey=True, squeeze=False,
    )
    limits = (
        (sample[x_column].quantile(0.002), sample[x_column].quantile(0.998)),
        (sample[y_column].quantile(0.002), sample[y_column].quantile(0.998)),
    )

    for column, value in enumerate(facet_values):
        block = sample[sample[facet_column] == value]
        points = block[[x_column, y_column]].to_numpy(dtype=float)
        mechanism = block["true_mechanism"].to_numpy()
        renal = (
            block["renal_dysfunction"].to_numpy()
            if mark_renal and "renal_dysfunction" in block else None
        )
        shown = (
            rng.choice(block.shape[0], points_per_facet, replace=False)
            if points_per_facet and block.shape[0] > points_per_facet
            else np.arange(block.shape[0])
        )

        top = axes[0][column]
        _draw_marker_plane(
            top, points, mechanism, MECHANISM_COLORS,
            shown=shown, renal=renal, coverage=coverage,
        )
        heading = SCATTER_FACET_WORDS.get(value, f"Renal {value:g} SD")
        top.set_title(f"{heading}\n({value:g} SD on NT-proBNP)", fontsize=10)
        top.text(
            0.03, 0.03,
            f"best 2-marker separation {_gaussian_separability(points, mechanism):.2f}",
            transform=top.transAxes, fontsize=8, color="#42474D",
        )

        if has_discovered:
            discovered = block["discovered_class"].to_numpy()
            bottom = axes[1][column]
            # Same open/filled convention as the top row.  Without it the
            # "matches kidney status" number is an assertion the reader cannot
            # check; with it, the discovered classes visibly *become* the
            # renal split as distortion grows.
            _draw_marker_plane(
                bottom, points, discovered, DISCOVERED_COLORS,
                shown=shown, renal=renal, coverage=coverage,
            )
            # Both labellings are reduced to booleans before comparison so the
            # agreement is well defined across their different value domains.
            found = discovered == "class A"
            note = f"matches mechanism {_agreement(found, mechanism == 'atrial'):.2f}"
            if renal is not None:
                note += (
                    "\nmatches kidney status "
                    f"{_agreement(found, renal == 1):.2f}"
                )
            bottom.text(
                0.03, 0.03, note, transform=bottom.transAxes,
                fontsize=8, color="#42474D",
            )

        for row in range(row_count):
            axis = axes[row][column]
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            axis.spines[["top", "right"]].set_visible(False)
        axes[row_count - 1][column].set_xlabel(x_label)

    row_titles = ["Coloured by true mechanism"]
    if has_discovered:
        row_titles.append("Coloured by what an unadjusted latent class model finds")
    for row, row_title in enumerate(row_titles):
        axes[row][0].set_ylabel(f"{row_title}\n\n{y_label}", fontsize=9.5)

    mechanism_handles = [
        plt.Line2D([], [], marker="o", linestyle="none",
                   color=MECHANISM_COLORS["atrial"], label="Atrial mechanism"),
        plt.Line2D([], [], marker="o", linestyle="none",
                   color=MECHANISM_COLORS["competing"], label="Competing mechanism"),
    ]
    if mark_renal:
        mechanism_handles.append(plt.Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor="#42474D", label="Renally impaired (open)",
        ))
    axes[0][0].legend(handles=mechanism_handles, fontsize=8, frameon=False,
                      loc="upper left")
    if has_discovered:
        discovered_handles = [
            plt.Line2D([], [], marker="o", linestyle="none",
                       color=color, label=f"Discovered {label}")
            for label, color in DISCOVERED_COLORS.items()
        ]
        if mark_renal:
            discovered_handles.append(plt.Line2D(
                [], [], marker="o", linestyle="none", markerfacecolor="none",
                markeredgecolor="#42474D", label="Renally impaired (open)",
            ))
        axes[1][0].legend(
            handles=discovered_handles, fontsize=8, frameon=False, loc="upper left",
        )
    figure.suptitle(
        title + (f"\n{subtitle}" if subtitle else ""), fontsize=12.5, y=1.0,
    )
    figure.tight_layout()
    _save(figure, path, source)


def plot_false_atrial_heatmaps(
    raw: pd.DataFrame,
    methods: list[str],
    path: Path,
    *,
    metric: str = "false_atrial__redundant",
    title: str = "False atrial attribution in the redundant profile",
    source: str = "",
) -> None:
    """One renal x HF heat map per method, shared color scale, annotated cells."""

    renal_levels = sorted(raw["renal_effect_sd"].unique())
    hf_levels = sorted(raw["heart_failure_effect_sd"].unique())
    grids = {}
    for method in methods:
        grid = np.full((len(renal_levels), len(hf_levels)), np.nan)
        block = raw[raw["method"] == method]
        for i, r in enumerate(renal_levels):
            for j, h in enumerate(hf_levels):
                cell = block[
                    (block["renal_effect_sd"] == r)
                    & (block["heart_failure_effect_sd"] == h)
                ][metric].to_numpy(dtype=float)
                grid[i, j] = _mean_ci(cell)[0]
        grids[method] = grid

    maximum = np.nanmax([np.nanmax(g) for g in grids.values()])
    # Method names are long by design; wrap them rather than letting adjacent
    # subplot titles run into one another.
    wrapped = {m: textwrap.fill(m, 24) for m in methods}
    title_lines = max(w.count("\n") + 1 for w in wrapped.values())
    figure, axes = plt.subplots(
        1, len(methods),
        figsize=(3.1 * len(methods), 3.4 + 0.22 * title_lines),
        sharey=True,
    )
    if len(methods) == 1:
        axes = [axes]
    for axis, method in zip(axes, methods):
        grid = grids[method]
        image = axis.imshow(
            grid, origin="lower", cmap="YlOrRd", vmin=0.0, vmax=maximum, aspect="auto"
        )
        for i in range(len(renal_levels)):
            for j in range(len(hf_levels)):
                if np.isfinite(grid[i, j]):
                    axis.text(
                        j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if grid[i, j] > 0.6 * maximum else "#22282E",
                    )
        axis.set_xticks(range(len(hf_levels)), [f"{h:g}" for h in hf_levels])
        axis.set_yticks(range(len(renal_levels)), [f"{r:g}" for r in renal_levels])
        axis.set_xlabel("HF effect on PTFV1 (SD)")
        axis.set_title(wrapped[method], fontsize=9)
    axes[0].set_ylabel("Renal effect on NT-proBNP (SD)")
    figure.suptitle(title, fontsize=12, y=1.04 + 0.02 * title_lines)
    figure.colorbar(image, ax=axes, shrink=0.85, label="False atrial rate")
    _save(figure, path, source)


def plot_hf_slice(
    raw: pd.DataFrame,
    path: Path,
    *,
    renal_effect_sd: float = 1.5,
    metric: str = "false_atrial__redundant",
    accent_methods: dict[str, str] | None = None,
    title: str = "",
    ylabel: str = "False atrial rate (redundant profile)",
    source: str = "",
) -> None:
    """Metric versus HF strength at fixed renal distortion, CI bands shaded."""

    accent_methods = accent_methods or {}
    block = raw[raw["renal_effect_sd"] == renal_effect_sd]
    hf_levels = sorted(block["heart_failure_effect_sd"].unique())
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    endpoints: list[tuple[float, str, str]] = []
    for method in block["method"].unique():
        means, lows, highs = [], [], []
        for h in hf_levels:
            cell = block[
                (block["method"] == method)
                & (block["heart_failure_effect_sd"] == h)
            ][metric].to_numpy(dtype=float)
            mean, low, high = _mean_ci(cell)
            means.append(mean), lows.append(low), highs.append(high)
        color = accent_methods.get(method, BASELINE_GRAY)
        width = 2.4 if method in accent_methods else 1.3
        axis.plot(hf_levels, means, color=color, linewidth=width, marker="o", markersize=4)
        axis.fill_between(hf_levels, lows, highs, color=color, alpha=0.15, linewidth=0)
        if np.isfinite(means[-1]):
            endpoints.append((float(means[-1]), method, color))

    axis.set_xlabel("Heart-failure effect on PTFV1 (SD)")
    axis.set_ylabel(ylabel)
    axis.set_title(title or f"Renal effect fixed at {renal_effect_sd:g} SD", fontsize=11)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlim(hf_levels[0], hf_levels[-1] + 1.05)

    # End labels beat a legend for line charts, but only if they do not stack.
    # Push them apart from the bottom up, keeping each one tied to its line by
    # a leader, so a curve is still identifiable when several converge.
    low_limit, high_limit = axis.get_ylim()
    gap = 0.055 * (high_limit - low_limit)
    placed = -np.inf
    for value, method, color in sorted(endpoints):
        target = max(value, placed + gap)
        placed = target
        axis.annotate(
            method, (hf_levels[-1], value),
            xytext=(hf_levels[-1] + 0.09, target), textcoords="data",
            fontsize=7.5, color=color, va="center",
            arrowprops={
                "arrowstyle": "-", "color": color,
                "linewidth": 0.6, "shrinkA": 1.0, "shrinkB": 1.0,
            } if abs(target - value) > 0.2 * gap else None,
        )
    _save(figure, path, source)


def plot_subgroup_bars(
    raw: pd.DataFrame,
    path: Path,
    *,
    renal_effect_sd: float = 1.5,
    hf_effect_sd: float = 1.5,
    methods: list[str] | None = None,
    accent_methods: dict[str, str] | None = None,
    title: str = "",
    source: str = "",
) -> None:
    """False-atrial attribution by nuisance profile at one grid cell."""

    accent_methods = accent_methods or {}
    block = raw[
        (raw["renal_effect_sd"] == renal_effect_sd)
        & (raw["heart_failure_effect_sd"] == hf_effect_sd)
    ]
    methods = methods or list(block["method"].unique())
    x = np.arange(len(PROFILE_ORDER))
    width = 0.8 / len(methods)
    # Distinct greys so stacked baselines stay separable in print and greyscale.
    baseline_shades = ["#C9D0D6", "#A9B3BB", "#8A949C", "#6E7880"]
    baseline_index = 0
    figure, axis = plt.subplots(figsize=(8.6, 4.6))
    for index, method in enumerate(methods):
        means, lows, highs = [], [], []
        for profile in PROFILE_ORDER:
            cell = block[block["method"] == method][
                f"false_atrial__{profile}"
            ].to_numpy(dtype=float)
            mean, low, high = _mean_ci(cell)
            means.append(mean)
            # A rate cannot be negative; clip the interval rather than drawing
            # an error bar into impossible territory.
            lows.append(max(low, 0.0) if np.isfinite(low) else mean)
            highs.append(min(high, 1.0) if np.isfinite(high) else mean)
        means = np.asarray(means, dtype=float)
        error = np.vstack((means - np.asarray(lows), np.asarray(highs) - means))
        error = np.clip(error, 0.0, None)
        if method in accent_methods:
            color, alpha = accent_methods[method], 1.0
        else:
            color = baseline_shades[baseline_index % len(baseline_shades)]
            baseline_index += 1
            alpha = 0.95
        offset = (index - (len(methods) - 1) / 2.0) * width
        axis.bar(
            x + offset, means, width * 0.92, yerr=error, capsize=2,
            color=color, alpha=alpha, label=method,
            error_kw={"linewidth": 0.8, "ecolor": "#3A424A"},
        )
    axis.set_ylim(bottom=0.0)
    axis.set_xticks(x, [PROFILE_LABELS[p] for p in PROFILE_ORDER])
    axis.set_ylabel("False atrial rate (competing-mechanism patients)")
    axis.set_title(
        title
        or f"Renal {renal_effect_sd:g} SD, HF {hf_effect_sd:g} SD",
        fontsize=11,
    )
    axis.legend(fontsize=7.5, frameon=False, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, path, source)


def plot_query_divergence(
    raw: pd.DataFrame,
    path: Path,
    *,
    posterior_method: str,
    counterfactual_method: str,
    renal_effect_sd: float = 1.5,
    metric: str = "false_atrial__redundant",
    title: str = "Counterfactual minus posterior, same fitted model",
    source: str = "",
) -> None:
    """Paired within-repeat query difference versus HF strength, with CI.

    Both rows come from the identical fit on the identical cohort, so each
    repeat yields one paired difference and the interval is a genuine paired
    comparison — the same construction as the locked contrast files.
    """

    block = raw[raw["renal_effect_sd"] == renal_effect_sd]
    hf_levels = sorted(block["heart_failure_effect_sd"].unique())
    means, lows, highs = [], [], []
    for h in hf_levels:
        level = block[block["heart_failure_effect_sd"] == h]
        wide = level.pivot(index="repeat", columns="method", values=metric)
        paired = (
            wide[counterfactual_method].to_numpy(dtype=float)
            - wide[posterior_method].to_numpy(dtype=float)
        )
        mean, low, high = _mean_ci(paired)
        means.append(mean), lows.append(low), highs.append(high)

    figure, axis = plt.subplots(figsize=(6.6, 4.0))
    axis.axhline(0.0, color="#22282E", linewidth=0.8, linestyle="--")
    axis.plot(hf_levels, means, color=ACCENT_CF, linewidth=2.2, marker="o")
    axis.fill_between(hf_levels, lows, highs, color=ACCENT_CF, alpha=0.18, linewidth=0)
    for h, m in zip(hf_levels, means):
        if np.isfinite(m):
            axis.annotate(
                f"{m:+.3f}", (h, m), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=ACCENT_CF,
            )
    axis.set_xlabel("Heart-failure effect on PTFV1 (SD)")
    axis.set_ylabel(f"Δ {metric}\n(counterfactual − posterior)")
    axis.set_title(f"{title}\nRenal fixed at {renal_effect_sd:g} SD", fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, path, source)
