"""Compatibility entry point for the packaged latent-endotyping experiment.

The numerical implementation moved without modification to the TRACE-ESUS
package.  This module remains so the existing notebook builders and archived
commands continue to import the same public names.
"""

from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from traceesus.experiments.endotype_discovery.kernel import *  # noqa: E402,F401,F403
from traceesus.experiments.endotype_discovery.kernel import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
