"""Deterministic retrieval-time expansion over the cross-reference graph.

Given seed chunk ids (selected by GraphRAG Local Search), follow the explicit
``cross_references.parquet`` edges — both out-edges (sections this chunk
references) and in-edges (sections that reference this chunk) — up to a bounded
BFS depth. Direct references (hop 1) always outrank indirect ones (hop 2+) via
hop-decay, and a token budget caps how much expansion is admitted.

This module is pure-python and unit-tested; it does not import GraphRAG.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class XrefGraph:
    """Adjacency over resolved chunk->chunk edges (both directions)."""

    out_edges: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    in_edges: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> "XrefGraph":
        g = cls()
        for r in rows:
            if not r.get("resolved"):
                continue
            src, dst = r.get("src_chunk_id"), r.get("dst_chunk_id")
            if not src or not dst:
                continue
            g.out_edges.setdefault(src, []).append(r)
            g.in_edges.setdefault(dst, []).append(r)
        return g

    @classmethod
    def from_parquet(cls, path: Any) -> "XrefGraph":
        import pandas as pd
        from pathlib import Path

        if not path or not Path(path).exists():
            return cls()
        df = pd.read_parquet(path)
        return cls.from_rows(df.to_dict("records"))


@dataclass
class XrefHit:
    chunk_id: str
    hop: int
    direction: str          # 'xref_out' | 'xref_in'
    via: str                # the reference text, e.g. "Section 5.21.1"
    edge_type: str
    score: float


def expand(graph: XrefGraph, seeds: dict[str, float], *, depth: int, hop_decay: float,
           token_of: Any, token_budget: int) -> list[XrefHit]:
    """BFS from seed chunk ids over the xref graph.

    Args
    ----
        seeds: mapping seed_chunk_id -> seed relevance score (higher = better).
        depth: max hops.
        hop_decay: multiply score by this per hop.
        token_of: callable chunk_id -> token count (for budgeting).
        token_budget: stop admitting hits once this many tokens are gathered.
    """
    visited: set[str] = set(seeds)
    hits: dict[str, XrefHit] = {}
    queue: deque[tuple[str, int, float]] = deque(
        (cid, 0, score) for cid, score in seeds.items()
    )
    spent = 0

    while queue:
        cid, hop, score = queue.popleft()
        if hop >= depth:
            continue
        for direction, edges in (("xref_out", graph.out_edges.get(cid, [])),
                                  ("xref_in", graph.in_edges.get(cid, []))):
            for edge in edges:
                neighbor = edge["dst_chunk_id"] if direction == "xref_out" else edge["src_chunk_id"]
                if not neighbor or neighbor in visited:
                    continue
                hop_n = hop + 1
                n_score = score * (hop_decay ** hop_n) * float(edge.get("weight", 1.0))
                cost = int(token_of(neighbor) or 0)
                if token_budget and spent + cost > token_budget and hits:
                    continue
                visited.add(neighbor)
                spent += cost
                hits[neighbor] = XrefHit(
                    chunk_id=neighbor, hop=hop_n, direction=direction,
                    via=str(edge.get("ref_text", "")), edge_type=str(edge.get("edge_type", "")),
                    score=n_score,
                )
                queue.append((neighbor, hop_n, n_score))

    return sorted(hits.values(), key=lambda h: (h.hop, -h.score))
