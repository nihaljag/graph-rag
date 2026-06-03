"""Deterministic cross-reference graph + generic concept metadata.

This stage does NOT use an LLM. It guarantees that explicit references between
sections ("See Section 5.21.1", "Figure 12", "refer to Status Code Definitions")
become traversable chunk->chunk edges, regardless of what the LLM extraction
infers. It also populates a generic, specification-agnostic ``mentioned_entities``
field on each chunk, derived from the document's own vocabulary.

Outputs:
- mutates each :class:`~specgraph.chunk.section_chunker.Chunk` in place
  (``refs_out``, ``refs_in``, ``mentioned_entities``, and optionally enriched
  ``text``),
- returns an edge list (list of dict rows) for ``cross_references.parquet``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from specgraph.chunk.section_chunker import Chunk

# Reference detectors (case-insensitive). High recall: any explicit pointer counts.
_SECTION_RE = re.compile(r"\b(?:section|clause|subclause|§)\s*(\d+(?:\.\d+)*)", re.I)
_FIGURE_RE = re.compile(r"\bfigure\s+(\d+(?:[.\-]\d+)*)", re.I)
_TABLE_RE = re.compile(r"\btable\s+(\d+(?:[.\-]\d+)*)", re.I)
# "see/refer to <Title Case Phrase>" — used to resolve NAMED references against titles.
_NAMED_CUE_RE = re.compile(
    r"\b(?:see|refer to|refer|defined in|specified in|described in|listed in)\s+"
    r"([A-Z][A-Za-z0-9][A-Za-z0-9 ,/()\-]{2,60})",
)
# Caption lines that DEFINE a figure/table, e.g. "Figure 12: ..." / "Table 45 -".
_FIG_CAPTION_RE = re.compile(r"^\s*figure\s+(\d+(?:[.\-]\d+)*)\s*[:.\-–—]", re.I | re.M)
_TBL_CAPTION_RE = re.compile(r"^\s*table\s+(\d+(?:[.\-]\d+)*)\s*[:.\-–—]", re.I | re.M)

_STOP_TITLES = {"introduction", "overview", "scope", "general", "notes", "definitions",
                "description", "references", "requirements"}


def norm_title(text: str) -> str:
    """Light normalization for title/phrase matching (NOT entity canonicalization)."""
    s = text.strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


@dataclass
class SectionIndex:
    by_section_no: dict[str, str] = field(default_factory=dict)
    by_title: dict[str, str] = field(default_factory=dict)
    by_figure: dict[str, str] = field(default_factory=dict)
    by_table: dict[str, str] = field(default_factory=dict)
    # (normalized_title, original_title, chunk_id) for vocabulary matching
    titles: list[tuple[str, str, str]] = field(default_factory=list)
    _title_re: re.Pattern[str] | None = None

    def build_title_regex(self, min_chars: int = 6) -> None:
        vocab = sorted(
            {(nt, ot) for nt, ot, _ in self.titles
             if len(nt) >= min_chars and nt not in _STOP_TITLES},
            key=lambda x: -len(x[0]),
        )
        if not vocab:
            self._title_re = None
            return
        alt = "|".join(re.escape(ot) for _, ot in vocab)
        self._title_re = re.compile(rf"\b({alt})\b", re.I)


def build_section_index(chunks: list[Chunk]) -> SectionIndex:
    """Index section numbers, titles, and figure/table captions to chunk ids."""
    idx = SectionIndex()
    for ch in chunks:
        # Prefer the first (part 1) chunk of a section as the canonical target.
        if ch.section_no and ch.section_no not in idx.by_section_no:
            idx.by_section_no[ch.section_no] = ch.chunk_id
        nt = norm_title(ch.title)
        if nt and nt not in idx.by_title:
            idx.by_title[nt] = ch.chunk_id
        if nt:
            idx.titles.append((nt, ch.title, ch.chunk_id))
        for m in _FIG_CAPTION_RE.finditer(ch.text):
            idx.by_figure.setdefault(m.group(1), ch.chunk_id)
        for m in _TBL_CAPTION_RE.finditer(ch.text):
            idx.by_table.setdefault(m.group(1), ch.chunk_id)
    idx.build_title_regex()
    return idx


def process_chunks(chunks: list[Chunk], *, named_references: list[str],
                   inline_resolved_titles: bool,
                   mentioned_entities: bool) -> list[dict[str, Any]]:
    """Detect/resolve references, populate metadata, return the edge list."""
    idx = build_section_index(chunks)
    named_norm = {norm_title(n): n for n in named_references}
    edges: list[dict[str, Any]] = []
    refs_in: dict[str, list[dict[str, Any]]] = {c.chunk_id: [] for c in chunks}

    for ch in chunks:
        seen_targets: set[tuple[str, str]] = set()
        out: list[dict[str, Any]] = []

        def add_ref(edge_type: str, raw: str, target_no: str | None, target_id: str | None) -> None:
            key = (edge_type, target_id or raw.lower())
            if key in seen_targets:
                return
            seen_targets.add(key)
            resolved = target_id is not None and target_id != ch.chunk_id
            ref = {"ref_type": edge_type, "raw_text": raw,
                   "target_section_no": target_no, "target_chunk_id": target_id if resolved else None,
                   "resolved": resolved}
            out.append(ref)
            weight = {"references_section": 1.0, "references_figure": 0.9,
                      "references_table": 0.9, "references_named": 0.8}.get(edge_type, 0.7)
            edges.append({
                "src_chunk_id": ch.chunk_id,
                "dst_chunk_id": target_id if resolved else None,
                "edge_type": edge_type,
                "ref_text": raw,
                "resolved": resolved,
                "weight": weight,
            })
            if resolved and target_id is not None:
                refs_in[target_id].append({"src_chunk_id": ch.chunk_id, "ref_type": edge_type,
                                           "raw_text": raw})

        body = ch.text
        for m in _SECTION_RE.finditer(body):
            no = m.group(1)
            add_ref("references_section", m.group(0).strip(), no, idx.by_section_no.get(no))
        for m in _FIGURE_RE.finditer(body):
            no = m.group(1)
            add_ref("references_figure", m.group(0).strip(), no, idx.by_figure.get(no))
        for m in _TABLE_RE.finditer(body):
            no = m.group(1)
            add_ref("references_table", m.group(0).strip(), no, idx.by_table.get(no))
        # Named references: explicit config phrases + cue-based title resolution.
        for nt, original in named_norm.items():
            if nt and nt in norm_title(body):
                add_ref("references_named", original, None, idx.by_title.get(nt))
        for m in _NAMED_CUE_RE.finditer(body):
            phrase = m.group(1).strip().rstrip(".,;:")
            tid = _resolve_phrase(phrase, idx)
            if tid:
                add_ref("references_named", phrase, None, tid)

        ch.refs_out = out
        if mentioned_entities:
            ch.mentioned_entities = _mentioned(ch, idx)
        if inline_resolved_titles:
            ch.text = _inline_titles(ch.text, out, chunks_by_id(chunks))

    for ch in chunks:
        ch.refs_in = refs_in.get(ch.chunk_id, [])
    return edges


def chunks_by_id(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def _resolve_phrase(phrase: str, idx: SectionIndex) -> str | None:
    """Resolve a free-text phrase to a chunk id via the title index."""
    nt = norm_title(phrase)
    if not nt:
        return None
    if nt in idx.by_title:
        return idx.by_title[nt]
    # Try progressively shorter prefixes (e.g. "status code definitions table" -> ...).
    words = nt.split()
    for end in range(len(words), 1, -1):
        cand = " ".join(words[:end])
        if cand in idx.by_title:
            return idx.by_title[cand]
    return None


def _mentioned(ch: Chunk, idx: SectionIndex) -> list[dict[str, Any]]:
    """Generic per-chunk concept list from the document's own title vocabulary."""
    found: dict[str, dict[str, Any]] = {}
    self_nt = norm_title(ch.title)
    if idx._title_re is not None:
        for m in idx._title_re.finditer(ch.text):
            ot = m.group(1)
            nt = norm_title(ot)
            if not nt or nt == self_nt:
                continue
            tid = idx.by_title.get(nt)
            found.setdefault(nt, {"name": ot, "source": "title_vocab", "canonical_id": tid})
            if len(found) >= 40:
                break
    # Resolved xref targets are strong "mentions".
    for ref in ch.refs_out:
        if ref.get("resolved") and ref.get("target_chunk_id"):
            name = ref.get("raw_text", "")
            found.setdefault("xref:" + (ref["target_chunk_id"]),
                             {"name": name, "source": "xref",
                              "canonical_id": ref["target_chunk_id"]})
    return list(found.values())


def _inline_titles(text: str, refs: list[dict[str, Any]], by_id: dict[str, Chunk]) -> str:
    """Append the resolved target's title next to a bare 'Section N' reference."""
    for ref in refs:
        if ref["ref_type"] != "references_section" or not ref["resolved"]:
            continue
        target = by_id.get(ref["target_chunk_id"] or "")
        if not target:
            continue
        raw = ref["raw_text"]
        annotated = f"{raw} ({target.title})"
        # Only annotate the first bare occurrence not already followed by '('.
        pattern = re.compile(re.escape(raw) + r"(?!\s*\()")
        text, n = pattern.subn(annotated, text, count=1)
    return text
