"""Test fixtures.

Token counting is now dependency-free (word-count based estimate in
``specgraph.tokens``), so there is nothing to stub — tests exercise the real
sizing path.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ layout importable without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
