"""Offline token counting via tiktoken.

Per project policy there is NO network fallback: if the encoding files are not
present in the offline tiktoken cache, this raises rather than downloading.
"""

from __future__ import annotations

import functools


class TokenizerUnavailable(RuntimeError):
    """Raised when the tiktoken encoding cannot be loaded offline."""


@functools.lru_cache(maxsize=4)
def _encoding(name: str):
    try:
        import tiktoken
    except Exception as exc:  # pragma: no cover
        raise TokenizerUnavailable(
            "tiktoken is not installed. It is required for section-aware token counting."
        ) from exc
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:
        raise TokenizerUnavailable(
            f"Could not load tiktoken encoding '{name}' offline. Set TIKTOKEN_CACHE_DIR to a "
            "directory containing the pre-downloaded encoding files (no network is used)."
        ) from exc


def count_tokens(text: str, encoding_model: str = "cl100k_base") -> int:
    """Return the number of tokens in ``text`` for the given encoding."""
    if not text:
        return 0
    return len(_encoding(encoding_model).encode(text, disallowed_special=()))
