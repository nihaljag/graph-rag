"""Test fixtures.

The real tiktoken encoding is unavailable offline (and in CI here), so we inject
a deterministic whitespace token counter. This exercises all chunk-sizing logic
without needing the BPE files; the production path uses the real offline cache.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the src/ layout importable without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/tmp/does-not-matter-for-tests")


def _word_count(text: str, encoding_model: str = "cl100k_base") -> int:
    return len(text.split())


@pytest.fixture(autouse=True)
def _patch_tokenizer(monkeypatch):
    import specgraph.tokens as tokens
    import specgraph.chunk.section_chunker as sc

    monkeypatch.setattr(tokens, "count_tokens", _word_count)
    monkeypatch.setattr(sc, "count_tokens", _word_count)
    yield
