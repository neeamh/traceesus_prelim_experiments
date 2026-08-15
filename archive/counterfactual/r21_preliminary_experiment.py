"""Compatibility entry point for the packaged preliminary experiment kernel.

Existing notebook builders and command lines import this module by its
historical name.  Numerical work remains in the packaged kernel so those
imports keep working without maintaining a second implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from archive.counterfactual.kernel import *  # noqa: F401,F403,E402
from archive.counterfactual.kernel import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
