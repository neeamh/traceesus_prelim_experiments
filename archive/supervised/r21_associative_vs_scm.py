"""Compatibility entry point for the packaged supervised model comparison.

All three methods in this experiment use true synthetic mechanism labels during
training. It remains supervised classification, not endotype discovery.
"""

from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from traceesus.experiments.model_comparison.kernel import *  # noqa: E402,F401,F403
from traceesus.experiments.model_comparison.kernel import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
