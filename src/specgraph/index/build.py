"""Phase C — build the GraphRAG index and assemble the portable DB.

Steps:
1. Validate config (LLM, embeddings, tokenizer) — fail loud.
2. Register the local-HF embedding provider with GraphRAG's factory.
3. Render ``settings.yaml`` and load it into a GraphRagConfig.
4. Run ``build_index`` feeding our section chunks via ``input_documents`` so our
   section boundaries are preserved (one chunk == one text-unit).
5. Normalize entities -> ``entity_aliases.parquet``.
6. Copy ``cross_references.parquet`` into the output and write ``db_manifest.json``.

The contents of ``graphrag_workspace/output/`` after this runs ARE the portable,
offline GraphRAG DB.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from specgraph.config import Settings
from specgraph.logging import get_logger, phase

log = get_logger()


def run_build(settings: Settings) -> dict[str, Any]:
    """Synchronous entry point for the indexing phase."""
    settings.require_llm()
    settings.require_embeddings()

    from specgraph.chunk.manifest import documents_dataframe, load_manifest
    from specgraph.index.settings_render import render_settings
    from specgraph.index.normalize_entities import build_entity_aliases
    from specgraph.llm.register import register_models

    rows = load_manifest(settings.manifest_path)
    if not rows:
        raise RuntimeError("Chunk manifest is empty; run the chunk phase first.")

    register_models()

    with phase("render settings.yaml"):
        render_settings(settings)

    with phase("load GraphRAG config"):
        from graphrag.config.load_config import load_config

        config = load_config(settings.workspace_dir)

    documents = documents_dataframe(rows)
    log.info("Indexing %d section chunks via GraphRAG build_index …", len(documents))

    with phase("GraphRAG build_index (entities, relationships, claims, communities, embeddings)"):
        asyncio.run(_run_index(config, documents))

    if settings.normalize.enabled:
        with phase("entity normalization (alias map)"):
            build_entity_aliases(settings)

    with phase("assemble portable DB"):
        _bundle(settings, n_chunks=len(rows))

    log.info("Portable GraphRAG DB ready at: %s", settings.output_dir)
    return {"output_dir": str(settings.output_dir), "n_chunks": len(rows)}


async def _run_index(config: Any, documents: Any) -> None:
    from graphrag.api.index import build_index

    results = await build_index(config, input_documents=documents)
    errors = [r for r in results if getattr(r, "errors", None)]
    if errors:
        # Surface the first error loudly; the logs dir has full detail.
        raise RuntimeError(f"GraphRAG indexing reported errors: {errors}")


def _bundle(settings: Settings, *, n_chunks: int) -> None:
    out = settings.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Copy the deterministic cross-reference graph into the portable DB.
    if settings.xref_path.exists():
        shutil.copy2(settings.xref_path, out / "cross_references.parquet")

    # Copy the chunk manifest too, so the retriever can map text-units -> chunks
    # and read mentioned_entities/breadcrumbs without the artifacts/ dir.
    if settings.manifest_path.exists():
        shutil.copy2(settings.manifest_path, out / "chunks_manifest.jsonl")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "graphrag_version": _version("graphrag"),
        "specgraph_version": _version("specgraph"),
        "embedding_model_path": settings.embeddings.local_path,
        "embedding_dim": settings.embeddings.dim,
        "n_chunks": n_chunks,
        "xref_edges": _parquet_len(out / "cross_references.parquet"),
        "entity_aliases": (out / "entity_aliases.parquet").exists(),
        "extract_claims": settings.graphrag.extract_claims,
        "files": sorted(p.name for p in out.glob("*.parquet")),
    }
    (out / "db_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _version(pkg: str) -> str:
    try:
        import importlib.metadata as m
        return m.version(pkg)
    except Exception:
        return "unknown"


def _parquet_len(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import pandas as pd
        return int(len(pd.read_parquet(path)))
    except Exception:
        return 0
