"""Load a portable GraphRAG DB and run Local Search context retrieval (offline).

This wires GraphRAG's ``LocalSearchMixedContext`` so we can obtain the Local
Search *context* (seed entities, relationships, covariates/claims, and selected
text-units) WITHOUT spending an LLM call — ideal for a retrieval-only
``BaseRetriever``. A separate :meth:`LocalSearchDB.answer` runs the full engine
(which does call the LLM) for demos / agent tools that want a synthesized answer.

The portable DB is the ``output/`` directory produced by the build phase:
``*.parquet`` + ``lancedb/`` + ``cross_references.parquet`` +
``entity_aliases.parquet`` + ``chunks_manifest.jsonl`` + ``db_manifest.json``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from specgraph.config import Settings
from specgraph.index.normalize_entities import canonical_key, load_aliases
from specgraph.index.settings_render import build_settings_dict
from specgraph.query.xref_expand import XrefGraph


def _read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path) if path.exists() else None


def build_query_config(settings: Settings, db_dir: Path):
    """Construct a GraphRagConfig pointing at the (possibly relocated) DB dir."""
    from graphrag.config.load_config import load_config

    cfg = build_settings_dict(settings)
    cfg["vector_store"]["db_uri"] = str(Path(db_dir) / "lancedb")
    cfg["output_storage"] = {"type": "file", "base_dir": str(db_dir)}
    tmp = Path(tempfile.mkdtemp(prefix="specgraph_query_"))
    (tmp / "settings.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return load_config(tmp)


class LocalSearchDB:
    """Loaded portable DB + a built Local Search engine."""

    def __init__(self, settings: Settings, db_dir: Path | None = None):
        settings.require_llm()
        settings.require_embeddings()
        self.settings = settings
        self.db_dir = Path(db_dir) if db_dir else settings.output_dir
        if not self.db_dir.exists():
            raise FileNotFoundError(
                f"Portable DB directory not found: {self.db_dir}. Run the build phase, "
                "or point --db at the copied output/ folder."
            )

        from specgraph.llm.register import register_models
        register_models(settings)

        # Load tables.
        self._entities = _read_parquet(self.db_dir / "entities.parquet")
        self._communities = _read_parquet(self.db_dir / "communities.parquet")
        self._reports = _read_parquet(self.db_dir / "community_reports.parquet")
        self._text_units = _read_parquet(self.db_dir / "text_units.parquet")
        self._relationships = _read_parquet(self.db_dir / "relationships.parquet")
        self._covariates = _read_parquet(self.db_dir / "covariates.parquet")
        if self._entities is None or self._text_units is None:
            raise RuntimeError(
                f"{self.db_dir} does not look like a GraphRAG DB (missing entities/text_units)."
            )

        # Side artifacts for retrieval-time expansion + provenance.
        self.xref = XrefGraph.from_parquet(self.db_dir / "cross_references.parquet")
        self.aliases = load_aliases(self.db_dir / "entity_aliases.parquet")
        self.chunks = self._load_manifest()
        self._textunit_to_chunk = self._build_tu_map()

        self._config = build_query_config(settings, self.db_dir)
        self._engine = None  # built lazily

    # ----- public API ------------------------------------------------------ #
    def fold(self, name: str) -> str:
        """Fold an entity name to its canonical form via the alias map."""
        return self.aliases.get(canonical_key(name), name)

    def token_of(self, chunk_id: str) -> int:
        rec = self.chunks.get(chunk_id)
        return int(rec.get("token_count", 0)) if rec else 0

    def chunk_text(self, chunk_id: str) -> str:
        rec = self.chunks.get(chunk_id)
        return rec.get("text", "") if rec else ""

    def context(self, query: str) -> Any:
        """Return GraphRAG's Local Search ContextBuilderResult (no LLM call)."""
        engine = self._get_engine()
        params = dict(getattr(engine, "context_builder_params", {}) or {})
        return engine.context_builder.build_context(query=query, **params)

    def answer(self, query: str) -> tuple[str, Any]:
        """Run the full Local Search engine (calls the LLM). Returns (text, context)."""
        engine = self._get_engine()
        result = engine.search(query=query)
        return getattr(result, "response", str(result)), getattr(result, "context_data", None)

    def map_text_unit(self, tu_id: Any, text: str | None = None) -> str | None:
        """Map a Local Search text-unit row back to our chunk id."""
        key = str(tu_id)
        if key in self._textunit_to_chunk:
            return self._textunit_to_chunk[key]
        if text:
            prefix = " ".join(text.split())[:80]
            return self._text_prefix_to_chunk.get(prefix)
        return None

    # ----- internals ------------------------------------------------------- #
    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        from graphrag.config.embeddings import entity_description_embedding
        from graphrag.query.factory import get_local_search_engine
        from graphrag.query.indexer_adapters import (
            read_indexer_covariates, read_indexer_entities, read_indexer_relationships,
            read_indexer_reports, read_indexer_text_units,
        )
        from graphrag.utils.api import get_embedding_store

        level = self.settings.graphrag.community_level
        store = get_embedding_store(config=self._config.vector_store,
                                    embedding_name=entity_description_embedding)
        covariates = (read_indexer_covariates(self._covariates)
                      if self._covariates is not None else [])
        reports = (read_indexer_reports(self._reports, self._communities, level)
                   if self._reports is not None and self._communities is not None else [])
        self._engine = get_local_search_engine(
            config=self._config,
            reports=reports,
            text_units=read_indexer_text_units(self._text_units),
            entities=read_indexer_entities(self._entities, self._communities, level),
            relationships=read_indexer_relationships(self._relationships),
            covariates={"claims": covariates},
            description_embedding_store=store,
            response_type=self.settings.graphrag.response_type,
        )
        return self._engine

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        path = self.db_dir / "chunks_manifest.jsonl"
        if not path.exists():
            path = self.settings.manifest_path
        out: dict[str, dict[str, Any]] = {}
        self._text_prefix_to_chunk: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[rec["chunk_id"]] = rec
                prefix = " ".join(rec.get("text", "").split())[:80]
                if prefix:
                    self._text_prefix_to_chunk[prefix] = rec["chunk_id"]
        return out

    def _build_tu_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        df = self._text_units
        if df is None:
            return mapping
        cols = set(df.columns)
        for row in df.to_dict("records"):
            chunk_id = None
            if "document_ids" in cols and isinstance(row.get("document_ids"), (list, tuple)) \
                    and len(row["document_ids"]):
                chunk_id = str(row["document_ids"][0])
            elif "document_id" in cols and row.get("document_id"):
                chunk_id = str(row["document_id"])
            if not chunk_id:
                continue
            if "id" in cols and row.get("id") is not None:
                mapping[str(row["id"])] = chunk_id
            if "human_readable_id" in cols and row.get("human_readable_id") is not None:
                mapping[str(row["human_readable_id"])] = chunk_id
        return mapping
