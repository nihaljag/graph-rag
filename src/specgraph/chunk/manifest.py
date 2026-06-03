"""Persist chunks + cross-reference edges, and build the GraphRAG input frame.

Artifacts written:
- ``artifacts/chunks/<chunk_id>.md``   — one file per chunk (human-inspectable; also
  doubles as a valid "directory of chunks" for the --from chunks entry point)
- ``artifacts/chunks_manifest.jsonl``  — canonical, machine-readable chunk records
- ``artifacts/cross_references.parquet`` — the deterministic chunk->chunk edge list
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from specgraph.chunk.section_chunker import Chunk


def chunk_to_record(ch: Chunk) -> dict[str, Any]:
    return dataclasses.asdict(ch)


def write_chunk_files(chunks: list[Chunk], chunks_dir: Path) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale chunk files so renames/re-runs are clean.
    for old in chunks_dir.glob("*.md"):
        old.unlink()
    for ch in chunks:
        (chunks_dir / f"{ch.chunk_id}.md").write_text(ch.text, encoding="utf-8")


def write_manifest(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(json.dumps(chunk_to_record(ch), ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk manifest not found: {path}. Run the chunk phase first."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_xref(edges: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["src_chunk_id", "dst_chunk_id", "edge_type", "ref_text", "resolved", "weight"]
    df = pd.DataFrame(edges, columns=cols) if edges else pd.DataFrame(columns=cols)
    df.to_parquet(path, index=False)


def documents_dataframe(rows: list[dict[str, Any]]):
    """Build the GraphRAG ``documents`` input frame (one of our chunks per row).

    ``id`` carries our ``chunk_id`` so that, after indexing, each text-unit's
    ``document_id`` maps back to the originating chunk. Chunk size in settings is
    set large enough that each document becomes exactly one text-unit, preserving
    section boundaries.
    """
    import pandas as pd

    records = []
    for r in rows:
        records.append({
            "id": r["chunk_id"],
            "title": r.get("breadcrumb") or r.get("title") or r["chunk_id"],
            "text": r["text"],
            "creation_date": "",
            "raw_data": {
                "chunk_id": r["chunk_id"],
                "section_no": r.get("section_no"),
                "breadcrumb": r.get("breadcrumb"),
                "level": r.get("level"),
                "part": r.get("part"),
                "n_parts": r.get("n_parts"),
                "page_start": r.get("page_start"),
                "page_end": r.get("page_end"),
            },
        })
    return pd.DataFrame.from_records(records)
