class Biomarker(IntEnum):
    NT_PROBNP = 0
    PTFV1 = 1
    COMPETING_VASCULAR = 2

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