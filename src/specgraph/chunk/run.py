"""Phase B driver — produce chunks, run the cross-reference stage, persist.

Two entry points mirror the skippable pipeline:
- ``run_chunk_from_markdown`` — section-chunk a single Markdown file.
- ``run_chunk_from_directory`` — ingest a user-supplied directory of pre-made
  chunks (one file == one chunk); chunking is skipped but xref still runs.
"""

from __future__ import annotations

from pathlib import Path

from specgraph.config import Settings
from specgraph.logging import get_logger, phase
from specgraph.chunk.manifest import write_chunk_files, write_manifest, write_xref
from specgraph.chunk.section_chunker import (
    Chunk, chunk_markdown, chunks_from_directory,
)
from specgraph.chunk.xref import process_chunks

log = get_logger()


def _finalize(chunks: list[Chunk], settings: Settings) -> list[Chunk]:
    if not chunks:
        raise RuntimeError("No chunks were produced from the input.")
    edges = process_chunks(
        chunks,
        named_references=(settings.xref.named_references if settings.xref.enabled else []),
        inline_resolved_titles=(settings.xref.inline_resolved_titles and settings.xref.enabled),
        mentioned_entities=settings.mentioned_entities_enabled,
    )
    write_chunk_files(chunks, settings.chunks_dir)
    write_manifest(chunks, settings.manifest_path)
    write_xref(edges if settings.xref.enabled else [], settings.xref_path)
    resolved = sum(1 for e in edges if e.get("resolved"))
    log.info("Chunks: %d | xref edges: %d (resolved %d) | chunks dir: %s",
             len(chunks), len(edges) if settings.xref.enabled else 0, resolved,
             settings.chunks_dir)
    return chunks


def run_chunk_from_markdown(md_path: Path, settings: Settings) -> list[Chunk]:
    md_path = Path(md_path)
    markdown = md_path.read_text(encoding="utf-8", errors="replace")
    doc_title = md_path.stem.replace("_", " ").strip() or "Specification"
    with phase(f"section chunking ({md_path.name})"):
        chunks = chunk_markdown(
            markdown,
            doc_title=doc_title,
            target_tokens=settings.chunking.target_tokens,
            max_tokens=settings.chunking.max_tokens,
            prepend_breadcrumb=settings.chunking.prepend_breadcrumb,
            encoding_model=settings.chunking.encoding_model,
        )
    with phase("cross-reference extraction"):
        return _finalize(chunks, settings)


def run_chunk_from_directory(directory: Path, settings: Settings) -> list[Chunk]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Chunks directory not found: {directory}")
    with phase(f"ingest chunk directory ({directory})"):
        chunks = chunks_from_directory(
            directory,
            target_tokens=settings.chunking.target_tokens,
            max_tokens=settings.chunking.max_tokens,
            prepend_breadcrumb=settings.chunking.prepend_breadcrumb,
            encoding_model=settings.chunking.encoding_model,
        )
    with phase("cross-reference extraction"):
        return _finalize(chunks, settings)
