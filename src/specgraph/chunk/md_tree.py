"""Parse Markdown into a heading tree with table-aware, atomic blocks.

This is the backbone of section-based chunking. It is deliberately dependency
free and specification-agnostic: it keys only on Markdown heading structure and
GitHub-flavored tables, so it works for any technical document.

Key guarantees:
- GFM tables (including multi-row / nested headers and tables that Docling
  stitched across page breaks) are kept as a single atomic block and are never
  split mid-table.
- Fenced code blocks are atomic.
- Section numbers like ``5.21.1`` are parsed from headings when present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
# A markdown table separator row, e.g. | --- | :--: | ---: |
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
# Leading dotted section number in a heading, e.g. "5.21.1 Abort command".
_SECTION_NO_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.*\S)\s*$")


@dataclass
class Block:
    """An atomic piece of section body content."""

    kind: str  # 'text' | 'table' | 'code'
    text: str


@dataclass
class Section:
    """A node in the heading tree."""

    level: int                       # 0 for the synthetic root / preamble
    title: str                       # heading text with any section number stripped
    raw_title: str                   # full heading text as written
    number: str | None               # parsed dotted section number, if any
    blocks: list[Block] = field(default_factory=list)   # body directly under heading
    children: list["Section"] = field(default_factory=list)
    breadcrumb: list[str] = field(default_factory=list)  # root..self titles

    def iter_sections(self):
        """Depth-first iteration over this section and its descendants."""
        yield self
        for child in self.children:
            yield from child.iter_sections()

    def has_body(self) -> bool:
        return any(b.text.strip() for b in self.blocks)


def _is_pipe_row(line: str) -> bool:
    s = line.strip()
    return "|" in s and not s.startswith(("#",))


def parse_blocks(markdown: str) -> list[Block | tuple[int, str, str]]:
    """Tokenize markdown into a flat list.

    Headings are emitted as ``(level, raw_title, full_line)`` tuples; body content
    is emitted as :class:`Block`. Kept internal; callers use :func:`build_tree`.
    """
    lines = markdown.split("\n")
    n = len(lines)
    out: list[Block | tuple[int, str, str]] = []
    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block (atomic).
        fence = _FENCE_RE.match(line)
        if fence:
            token = fence.group(1)
            buf = [line]
            i += 1
            while i < n and not lines[i].strip().startswith(token):
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i])
                i += 1
            out.append(Block("code", "\n".join(buf)))
            continue

        # Heading.
        hm = _HEADING_RE.match(line)
        if hm:
            level = len(hm.group(1))
            raw_title = hm.group(2).strip()
            out.append((level, raw_title, line))
            i += 1
            continue

        # Table: a pipe row immediately followed by a separator row (atomic block).
        if _is_pipe_row(line) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            buf = [line, lines[i + 1]]
            i += 2
            while i < n and lines[i].strip() and _is_pipe_row(lines[i]):
                buf.append(lines[i])
                i += 1
            out.append(Block("table", "\n".join(buf)))
            continue

        # Blank line.
        if not stripped:
            i += 1
            continue

        # Paragraph / list: accumulate until a structural boundary.
        buf = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            s = nxt.strip()
            if not s:
                break
            if _HEADING_RE.match(nxt) or _FENCE_RE.match(nxt):
                break
            if _is_pipe_row(nxt) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
                break
            buf.append(nxt)
            i += 1
        out.append(Block("text", "\n".join(buf)))
    return out


def split_section_number(raw_title: str) -> tuple[str | None, str]:
    """Return ``(number, title_without_number)`` for a heading."""
    m = _SECTION_NO_RE.match(raw_title)
    if m:
        return m.group(1), m.group(2).strip()
    return None, raw_title.strip()


def build_tree(markdown: str, doc_title: str = "Document") -> Section:
    """Build a heading tree from markdown. Returns the synthetic root section."""
    root = Section(level=0, title=doc_title, raw_title=doc_title, number=None)
    root.breadcrumb = [doc_title]
    stack: list[Section] = [root]

    for token in parse_blocks(markdown):
        if isinstance(token, Block):
            stack[-1].blocks.append(token)
            continue
        level, raw_title, _line = token
        number, title = split_section_number(raw_title)
        node = Section(level=level, title=title, raw_title=raw_title, number=number)
        # Pop until we find a parent with a strictly smaller level.
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1]
        node.breadcrumb = [*parent.breadcrumb, title]
        parent.children.append(node)
        stack.append(node)

    return root
