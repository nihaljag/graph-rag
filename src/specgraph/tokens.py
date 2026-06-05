"""Lightweight, dependency-free token estimation.

Exact token counts are unnecessary for section sizing, so we avoid a hard
dependency on tiktoken (and its offline encoding-file requirement) entirely.
We estimate tokens from word count using a fixed average ratio. This is good
enough for deciding when to split a section and keeps the project lean and
fully offline with zero extra setup.
"""

from __future__ import annotations

import math

# Empirical average for English/technical prose: ~1.3 tokens per whitespace word.
TOKENS_PER_WORD = 1.3


def count_tokens(text: str, encoding_model: str = "cl100k_base") -> int:
    """Estimate the number of tokens in ``text`` from its word count.

    ``encoding_model`` is accepted for signature compatibility but ignored.
    """
    if not text:
        return 0
    words = len(text.split())
    return max(1, math.ceil(words * TOKENS_PER_WORD))
