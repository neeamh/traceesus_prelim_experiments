"""Stable visual constants for TRACE-ESUS figures.

The proposal-locked plotting kernels still spell out their historical values
locally. Redirecting those calls during the identity refactor would create a
needless compatibility risk, so this module is the canonical palette and size
inventory for new panels while the old kernels remain frozen.
"""

from __future__ import annotations

PALETTE: dict[str, str] = {
    "associative": "#D97706",
    "adjusted": "#6B7280",
    "causal": "#1D4ED8",
    "oracle": "#111827",
    "grid": "#D1D5DB",
    "text": "#111827",
    "muted_text": "#4B5563",
}

DEFAULT_FONT: str = "DejaVu Sans"

# Exact inch dimensions used by the eight proposal-locked figure families.
# Separate keys are intentional: superficially similar two-panel plots differ
# in width or height, and normalizing them would alter rendered artifacts.
FIGURE_SIZES_INCHES: dict[str, tuple[float, float]] = {
    "counterfactual_primary": (11.2, 4.6),
    "model_comparison_primary": (11.4, 4.7),
    "endotype_recovery": (12.2, 5.1),
    "endotype_controls": (12.2, 5.0),
    "endotype_example_patient": (8.8, 5.1),
    "transportability_primary": (12.4, 5.2),
    "transportability_controls": (12.2, 5.0),
    "transportability_ablation": (10.4, 5.4),
}

__all__ = ["DEFAULT_FONT", "FIGURE_SIZES_INCHES", "PALETTE"]
