"""Pure, dependency-free LLM call logging helpers (no graphrag imports).

Kept separate from ``graphrag_chat`` (which imports graphrag_llm) so the labelling
and counter logic can be unit-tested anywhere.
"""

from __future__ import annotations

import threading
from typing import Any

from specgraph.logging import get_logger

log = get_logger()

# Tunables (overridden from config by register_models()).
SNIPPET_CHARS = 90
SLOW_CALL_SECONDS = 20.0

_LOCK = threading.Lock()
_STATS = {"started": 0, "done": 0, "failed": 0, "in_flight": 0}

# Heuristic, specification-agnostic mapping of prompt content -> task label.
# Order matters: more specific tasks are checked before the generic graph
# extraction (whose prompts also mention "entity").
_LABELS = [
    ("claim", "extract_claims"),
    ("covariate", "extract_claims"),
    ("community report", "community_report"),
    ("summar", "summarize_descriptions"),
    ("report", "community_report"),
    ("entit", "extract_graph"),
    ("relationship", "extract_graph"),
]


def call_stats() -> dict[str, int]:
    with _LOCK:
        return dict(_STATS)


def reset_stats() -> None:
    with _LOCK:
        for k in _STATS:
            _STATS[k] = 0


def first_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, (list, tuple)):
        for role in ("system", "user"):
            for m in messages:
                if isinstance(m, dict) and m.get("role") == role and isinstance(m.get("content"), str):
                    return m["content"]
        for m in messages:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                return m["content"]
    return ""


def label(messages: Any) -> str:
    text = first_text(messages)
    low = text.lower()
    for needle, name in _LABELS:
        if needle in low:
            return name
    return " ".join(text.split())[:SNIPPET_CHARS] or "llm-call"


def begin(messages: Any) -> tuple[int, str]:
    lbl = label(messages)
    with _LOCK:
        _STATS["started"] += 1
        _STATS["in_flight"] += 1
        n = _STATS["started"]
        inflight = _STATS["in_flight"]
    log.info("LLM #%d [%s] (in-flight %d)", n, lbl, inflight)
    return n, lbl


def end(n: int, lbl: str, dt: float, failed: bool) -> None:
    with _LOCK:
        _STATS["in_flight"] = max(0, _STATS["in_flight"] - 1)
        _STATS["done"] += 1
        if failed:
            _STATS["failed"] += 1
    if failed:
        log.warning("LLM #%d [%s] FAILED after %.1fs", n, lbl, dt)
    elif dt >= SLOW_CALL_SECONDS:
        log.info("LLM #%d [%s] slow: %.0fs", n, lbl, dt)
