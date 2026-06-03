"""specgraph — offline GraphRAG (Local Search) index builder for technical specs.

Importing this package immediately enforces offline mode (see ``offline.py``) so
that no dependency can silently reach the public internet. The only permitted
network egress is to the configured local LLM endpoint.
"""

from __future__ import annotations

# Enforce offline behaviour for HF / tokenizers / litellm before anything else
# imports those libraries.
from specgraph import offline as _offline  # noqa: F401  (side effects on import)

__all__ = ["__version__"]
__version__ = "0.1.0"
