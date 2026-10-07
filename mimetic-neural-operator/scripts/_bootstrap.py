"""Make `src/` importable when the package has not been pip-installed.

Every stage script imports this first, so `python scripts/stageN_*.py` works
straight out of a fresh clone with no `pip install -e .` step.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
