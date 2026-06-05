"""Turn a Markdown heading tree into logical, section-based chunks.

One chunk == one logical section (or a size-bounded part of a large section).
Tables are never split mid-table unless a single table exceeds the hard token
cap, in which case it is split at row boundaries with the header row repeated.

This module is dependency-free (token counts are estimated from word counts) and
entirely specification-agnostic.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from specgraph.chunk.md_tree import Block, Section, build_tree, split_section_number
from specgraph.tokens import count_tokens

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


@dataclass
class Chunk:
    """A single emitted chunk and its metadata (one row of the manifest)."""

    order: int
    chunk_id: str
    title: str
    breadcrumb: str
    section_no: str | None
    level: int
    part: int
    n_parts: int
    token_count: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    source_path: str | None = None
    original_name: str | None = None
    refs_out: list[dict[str, Any]] = field(default_factory=list)
    refs_in: list[dict[str, Any]] = field(default_factory=list)
    mentioned_entities: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def chunk_markdown(markdown: str, *, doc_title: str, target_tokens: int,
                   max_tokens: int, prepend_breadcrumb: bool,
                   encoding_model: str = "cl100k_base") -> list[Chunk]:
    """Chunk a single Markdown document into section-based chunks."""
    root = build_tree(markdown, doc_title=doc_title)
    chunks: list[Chunk] = []
    order = 0
    for sec in root.iter_sections():
        if not sec.has_body():
            continue  # heading-only parents contribute via their subsections
        bodies = _emit_section_bodies(
            sec, target_tokens=target_tokens, max_tokens=max_tokens,
            prepend_breadcrumb=prepend_breadcrumb, encoding_model=encoding_model,
        )
        n_parts = len(bodies)
        for part_idx, text in enumerate(bodies, start=1):
            order += 1
            chunks.append(Chunk(
                order=order,
                chunk_id=_make_id(order, sec, part_idx),
                title=sec.title,
                breadcrumb=" > ".join(sec.breadcrumb),
                section_no=sec.number,
                level=sec.level,
                part=part_idx,
                n_parts=n_parts,
                token_count=count_tokens(text, encoding_model),
                text=text,
            ))
    return chunks


def chunks_from_directory(directory: Path, *, target_tokens: int, max_tokens: int,
                          prepend_breadcrumb: bool,
                          encoding_model: str = "cl100k_base") -> list[Chunk]:
    """Build chunks from a user-supplied directory of pre-made chunk files.

    One file == one chunk. We re-derive a breadcrumb/section number from the
    file's first heading (or its filename) and freely rename to deterministic
    ids, preserving the original filename in metadata. Files that still exceed
    the hard cap are split the same way as section chunks.
    """
    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".markdown", ".txt", ".text", ""}
    )
    chunks: list[Chunk] = []
    order = 0
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            continue
        title, number, breadcrumb = _derive_heading(raw, path)
        # Wrap the file content as a single synthetic section so we reuse the
        # same size-guard / sub-splitting machinery.
        sec = Section(level=1, title=title, raw_title=(f"{number} {title}" if number else title),
                      number=number, blocks=_blocks_from_text(raw))
        sec.breadcrumb = breadcrumb
        bodies = _emit_section_bodies(
            sec, target_tokens=target_tokens, max_tokens=max_tokens,
            prepend_breadcrumb=prepend_breadcrumb, encoding_model=encoding_model,
        )
        n_parts = len(bodies)
        for part_idx, text in enumerate(bodies, start=1):
            order += 1
            chunks.append(Chunk(
                order=order,
                chunk_id=_make_id(order, sec, part_idx),
                title=sec.title,
                breadcrumb=" > ".join(sec.breadcrumb),
                section_no=sec.number,
                level=sec.level,
                part=part_idx,
                n_parts=n_parts,
                token_count=count_tokens(text, encoding_model),
                text=text,
                source_path=str(path),
                original_name=path.name,
            ))
    return chunks


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _emit_section_bodies(sec: Section, *, target_tokens: int, max_tokens: int,
                         prepend_breadcrumb: bool, encoding_model: str) -> list[str]:
    """Return the final chunk text(s) for one section."""
    heading_line = "" if sec.level == 0 else f"{'#' * max(sec.level, 1)} {sec.raw_title}\n\n"
    prefix = ""
    if prepend_breadcrumb and sec.breadcrumb:
        prefix = f"> Context: {' > '.join(sec.breadcrumb)}\n\n"

    body = "\n\n".join(b.text for b in sec.blocks if b.text.strip())
    whole = f"{prefix}{heading_line}{body}".strip()
    if count_tokens(whole, encoding_model) <= max_tokens:
        return [whole]

    # Oversized: sub-split the body, then re-attach prefix + heading to each part.
    overhead = count_tokens(f"{prefix}{heading_line}", encoding_model) + 8
    budget = max(target_tokens - overhead, 128)
    body_parts = _pack_blocks(sec.blocks, budget, max_tokens - overhead, encoding_model)
    n = len(body_parts)
    out: list[str] = []
    for idx, part_body in enumerate(body_parts, start=1):
        cont = f"_(continued — part {idx} of {n})_\n\n" if n > 1 and idx > 1 else ""
        out.append(f"{prefix}{heading_line}{cont}{part_body}".strip())
    return out


def _pack_blocks(blocks: list[Block], soft: int, hard: int, enc: str) -> list[str]:
    """Greedily pack atomic blocks into parts ~``soft`` tokens, never exceeding ``hard``."""
    parts: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    def flush() -> None:
        nonlocal cur, cur_tok
        if cur:
            parts.append("\n\n".join(cur))
            cur = []
            cur_tok = 0

    for block in blocks:
        text = block.text
        if not text.strip():
            continue
        bt = count_tokens(text, enc)
        if bt > hard:
            flush()
            pieces = (_split_table(text, hard, enc) if block.kind == "table"
                      else _split_text(text, hard, enc))
            parts.extend(pieces)
            continue
        if cur and cur_tok + bt > soft:
            flush()
        cur.append(text)
        cur_tok += bt
    flush()
    return parts or [""]


def _split_table(table_text: str, budget: int, enc: str) -> list[str]:
    """Split a large table at row boundaries, repeating the header in each part."""
    rows = table_text.split("\n")
    if len(rows) < 3:
        return [table_text]
    header, sep, body_rows = rows[0], rows[1], rows[2:]
    base = count_tokens(header + "\n" + sep + "\n", enc)
    parts: list[str] = []
    cur: list[str] = []
    cur_tok = base
    for r in body_rows:
        rt = count_tokens(r, enc) + 1
        if cur and cur_tok + rt > budget:
            parts.append("\n".join([header, sep, *cur]))
            cur = []
            cur_tok = base
        cur.append(r)
        cur_tok += rt
    if cur:
        parts.append("\n".join([header, sep, *cur]))
    return parts


def _split_text(text: str, budget: int, enc: str) -> list[str]:
    """Split prose at line boundaries (words as a last resort)."""
    parts: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for line in text.split("\n"):
        lt = count_tokens(line, enc) + 1
        if lt > budget:
            if cur:
                parts.append("\n".join(cur))
                cur = []
                cur_tok = 0
            parts.extend(_split_words(line, budget, enc))
            continue
        if cur and cur_tok + lt > budget:
            parts.append("\n".join(cur))
            cur = []
            cur_tok = 0
        cur.append(line)
        cur_tok += lt
    if cur:
        parts.append("\n".join(cur))
    return parts


def _split_words(line: str, budget: int, enc: str) -> list[str]:
    words = line.split(" ")
    parts: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for w in words:
        wt = count_tokens(w, enc) + 1
        if cur and cur_tok + wt > budget:
            parts.append(" ".join(cur))
            cur = []
            cur_tok = 0
        cur.append(w)
        cur_tok += wt
    if cur:
        parts.append(" ".join(cur))
    return parts


def _blocks_from_text(raw: str) -> list[Block]:
    """Re-tokenize a raw chunk file into atomic blocks (table-aware)."""
    from specgraph.chunk.md_tree import parse_blocks
    blocks: list[Block] = []
    for token in parse_blocks(raw):
        if isinstance(token, Block):
            blocks.append(token)
        else:
            # A heading inside a supplied chunk file becomes a text block so its
            # content is preserved (we do not re-tree user-provided chunks).
            _level, raw_title, line = token
            blocks.append(Block("text", line))
    return blocks


def _derive_heading(raw: str, path: Path) -> tuple[str, str | None, list[str]]:
    """Best-effort title/number/breadcrumb for a user-supplied chunk file."""
    for line in raw.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            number, title = split_section_number(m.group(2).strip())
            return title, number, [title]
    # No heading: fall back to a cleaned filename.
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    number, title = split_section_number(stem)
    return (title or stem or path.name), number, [title or stem or path.name]


def _slug(text: str, maxlen: int = 48) -> str:
    s = _SLUG_RE.sub("-", text).strip("-").lower()
    return s[:maxlen] or "section"


def _make_id(order: int, sec: Section, part: int) -> str:
    label = f"{sec.number + '-' if sec.number else ''}{sec.title}"
    h = hashlib.sha1((" > ".join(sec.breadcrumb) + f"#p{part}").encode()).hexdigest()[:6]
    return f"{order:04d}__{_slug(label)}__{h}"
