"""Generic, dictionary-free entity normalization (anti-fragmentation).

Produces ``output/entity_aliases.parquet`` mapping every entity-name variant to a
canonical name. This is applied at RETRIEVAL TIME (folding seed/related entity
names to canonical) rather than by rewriting the indexed graph + vector store —
which keeps the portable DB intact and version-robust while still collapsing
fragmented nodes ("Identify" / "Identify Command" / "the Identify command") for
retrieval.

Tiers (both specification-agnostic):
- Tier 2 (deterministic, default ON): merge entities whose canonical key is equal
  after lowercasing, article/possessive stripping, punctuation collapse, and
  conservative singularization.
- Tier 3 (embedding, optional): additionally merge a shorter name into a longer
  one when their local embeddings are highly similar AND the shorter key's tokens
  are a subset of the longer key's tokens (guards against false merges like
  "Write" vs "Write Zeroes").
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from specgraph.config import Settings
from specgraph.logging import get_logger

log = get_logger()

_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_POSS_RE = re.compile(r"[’']s\b")
_NONWORD_RE = re.compile(r"[^a-z0-9]+")


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and not token.endswith(("ses", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def canonical_key(name: str) -> str:
    """Aggressive, generic canonicalization key used for grouping."""
    s = (name or "").strip().lower()
    s = _POSS_RE.sub("", s)
    s = _ARTICLE_RE.sub("", s)
    s = _NONWORD_RE.sub(" ", s).strip()
    return " ".join(_singular(t) for t in s.split())


def build_entity_aliases(settings: Settings) -> Path | None:
    """Compute and persist the alias map. Returns the parquet path (or None)."""
    import pandas as pd

    entities_path = settings.output_dir / "entities.parquet"
    if not entities_path.exists():
        log.warning("entities.parquet not found; skipping normalization.")
        return None

    df = pd.read_parquet(entities_path)
    name_col = "title" if "title" in df.columns else ("name" if "name" in df.columns else None)
    if name_col is None:
        log.warning("entities parquet has no title/name column; skipping normalization.")
        return None

    names: list[str] = [str(n) for n in df[name_col].tolist()]
    id_col = "id" if "id" in df.columns else None
    ids = [str(i) for i in df[id_col].tolist()] if id_col else [""] * len(names)

    # Tier 2: group by canonical key; canonical name = the longest variant.
    key_to_names: dict[str, list[str]] = {}
    for nm in names:
        key_to_names.setdefault(canonical_key(nm), []).append(nm)
    canonical_of_key: dict[str, str] = {
        k: max(v, key=lambda s: (len(s), s)) for k, v in key_to_names.items()
    }

    # Tier 3 (optional): merge shorter-key entities into a longer superset key
    # when embeddings agree.
    if settings.normalize.embed_merge:
        canonical_of_key = _embedding_merge(
            settings, canonical_of_key, key_to_names,
        )

    rows: list[dict[str, Any]] = []
    for nm, _id in zip(names, ids):
        k = canonical_key(nm)
        canonical = canonical_of_key.get(k, nm)
        rows.append({
            "name": nm,
            "name_key": k,
            "canonical": canonical,
            "canonical_key": canonical_key(canonical),
            "id": _id,
        })
    out = pd.DataFrame(rows).drop_duplicates(subset=["name"])
    path = settings.output_dir / "entity_aliases.parquet"
    out.to_parquet(path, index=False)
    n_merged = sum(1 for r in rows if r["name"] != r["canonical"])
    log.info("Entity normalization: %d names, %d folded to a canonical form.",
             len(out), n_merged)
    return path


def _embedding_merge(settings: Settings, canonical_of_key: dict[str, str],
                     key_to_names: dict[str, list[str]]) -> dict[str, str]:
    """Tier 3: fold a shorter key into a longer one if embeddings are close AND
    the shorter key's tokens are a subset of the longer key's tokens."""
    import numpy as np

    from specgraph.llm.graphrag_embed import load_local_model

    keys = list(canonical_of_key.keys())
    if len(keys) < 2:
        return canonical_of_key
    model = load_local_model(settings.embeddings.local_path, settings.embeddings.device)
    canon_names = [canonical_of_key[k] for k in keys]
    vecs = model.encode(canon_names, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)
    thr = settings.normalize.embed_threshold

    token_sets = {k: set(k.split()) for k in keys}
    # Sort by token count so shorter keys look for a longer superset partner.
    order = sorted(range(len(keys)), key=lambda i: len(token_sets[keys[i]]))
    remap: dict[str, str] = {}
    for ai in order:
        ka = keys[ai]
        ta = token_sets[ka]
        if not ta:
            continue
        best = None
        best_sim = thr
        for bi in range(len(keys)):
            if bi == ai:
                continue
            kb = keys[bi]
            tb = token_sets[kb]
            if not ta.issubset(tb) or len(tb) <= len(ta):
                continue
            sim = float(np.dot(vecs[ai], vecs[bi]))
            if sim >= best_sim:
                best_sim = sim
                best = kb
        if best is not None:
            remap[ka] = canonical_of_key[best]

    merged = dict(canonical_of_key)
    for k, canonical in remap.items():
        merged[k] = canonical
    return merged


def load_aliases(path: Path) -> dict[str, str]:
    """Return ``canonical_key(name) -> canonical_name`` for retrieval-time folding."""
    import pandas as pd

    if not path or not Path(path).exists():
        return {}
    df = pd.read_parquet(path)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        mapping[str(row["name_key"])] = str(row["canonical"])
    return mapping
