"""Provide the sole shared visual style and save helper for CSV-driven notebooks."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


PALETTE = MappingProxyType(
    {
        "associative": "#D97706",
        "adjusted": "#6B7280",
        "causal": "#1D4ED8",
        "oracle": "#111827",
        "grid": "#D1D5DB",
        "text": "#111827",
        "muted_text": "#4B5563",
    }
)

DEFAULT_FONT = "DejaVu Sans"

FIGURE_SIZES_INCHES = MappingProxyType(
    {
        "counterfactual_primary": (11.2, 4.6),
        "endotype_recovery": (12.2, 5.1),
        "endotype_controls": (12.2, 5.0),
        "endotype_example_patient": (8.8, 5.1),
        "transportability_primary": (12.4, 5.2),
        "transportability_controls": (12.2, 5.0),
        "transportability_ablation": (10.4, 5.4),
    }
)


def configure_style() -> None:
    """Apply the stable font and restrained publication defaults."""

    plt.rcParams.update(
        {
            "font.family": DEFAULT_FONT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": PALETTE["text"],
            "text.color": PALETTE["text"],
        }
    )


def save_figure(figure: Figure, path: Path, *, dpi: int = 240) -> None:
    """Save one notebook figure as matched PNG and PDF files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


__all__ = [
    "DEFAULT_FONT",
    "FIGURE_SIZES_INCHES",
    "PALETTE",
    "configure_style",
    "save_figure",
]
