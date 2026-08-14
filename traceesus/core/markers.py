"""Define stable biomarker positions and names shared across models."""

from __future__ import annotations

from enum import IntEnum


class Biomarker(IntEnum):
    """Column positions shared by every simulator and fitted model."""

    NT_PROBNP = 0
    PTFV1 = 1
    COMPETING_VASCULAR = 2

    NT_PROBNP_LIKE = 0
    ATRIAL_ELECTRICAL = 1
    COMPETING_SPECIFIC = 2

# Provenance-stable — do not change, appears in locked outputs
BIOMARKER_NAMES = (
    "NT-proBNP-like biomarker",
    "Atrial electrical evidence",
    "Competing-mechanism evidence",
)

# Figure/table display only
BIOMARKER_DISPLAY_NAMES = (
    "NT-proBNP",
    "PTFV1",
    "Competing-vascular",
)


__all__ = ["BIOMARKER_DISPLAY_NAMES", "BIOMARKER_NAMES", "Biomarker"]
