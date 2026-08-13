"""Compatibility entry point for the packaged transportability experiment.

The numerical kernel moved without algorithmic changes. This module preserves
the archived CLI and the names imported by the notebook builder.
"""

from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from traceesus.experiments.transportability.kernel import *  # noqa: E402,F401,F403
from traceesus.experiments.transportability.kernel import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
