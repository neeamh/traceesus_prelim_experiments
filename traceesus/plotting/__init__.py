"""Expose the notebook-only visual style without package figure generation."""

from .style import (
    DEFAULT_FONT,
    FIGURE_SIZES_INCHES,
    PALETTE,
    configure_style,
    save_figure,
)

__all__ = [
    "DEFAULT_FONT",
    "FIGURE_SIZES_INCHES",
    "PALETTE",
    "configure_style",
    "save_figure",
]
