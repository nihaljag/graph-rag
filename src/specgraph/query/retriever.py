"""LangChain retriever over the portable GraphRAG DB.

Retrieval = GraphRAG Local Search  ⊕  deterministic cross-reference expansion.

For a query like "requirements and error conditions for Abort Command" the
retriever returns, with provenance:
- the Abort section (Local Search seed),
- related entities/commands surfaced by the entity graph (Local Search),
- explicitly referenced sections via the xref graph (guaranteed, hop<=depth),
- normative claims/covariates attached to the involved entities.

It is retrieval-only (no answer LLM); use ``.as_tool()`` for an agent tool, or
``LocalSearchDB.answer`` for a synthesized answer.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from specgraph.config import Settings
from specgraph.index.normalize_entities import canonical_key
from specgraph.query.local_search import LocalSearchDB
from specgraph.query.xref_expand import expand


def _col(df: Any, *candidates: str) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


class GraphRAGLocalRetriever(BaseRetriever):
    """Local Search + xref expansion, returning provenance-rich Documents."""

    db: Any
    k: int = 12
    mention_boost: float = 0.25
    include_claims: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(  # noqa: D401
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        db: LocalSearchDB = self.db
        ctx = db.context(query)
        records: dict[str, Any] = getattr(ctx, "context_records", {}) or {}

        seed_ids = self._seed_chunk_ids(db, records)
        seed_entities = self._seed_entities(db, records)

        docs: list[Document] = []
        seen: set[str] = set()

        # 1) Seed chunks from Local Search (entity-graph context).
        n = len(seed_ids)
        for i, cid in enumerate(seed_ids):
            if cid in seen:
                continue
            seen.add(cid)
            docs.append(self._doc(db, cid, source="local_search", hop=0,
                                  score=float(n - i), via=""))

        # 2) Normative claims/covariates (critical for test generation).
        if self.include_claims:
            claim_doc = self._claims_doc(records)
            if claim_doc is not None:
                docs.append(claim_doc)

        # 3) Deterministic cross-reference expansion (depth-2, budget-bounded).
        seeds_scores = {cid: float(n - i) for i, cid in enumerate(seed_ids)}
        budget = int(self.db.settings.xref.expand_token_budget * self._context_window())
        hits = expand(
            db.xref, seeds_scores,
            depth=db.settings.xref.expand_depth,
            hop_decay=db.settings.xref.hop_decay,
            token_of=db.token_of,
            token_budget=budget,
        )
        # mentioned_entities boost: a neighbor that mentions a seed entity ranks up.
        for h in hits:
            if self._mentions_any(db, h.chunk_id, seed_entities):
                h.score *= (1.0 + self.mention_boost)
        hits.sort(key=lambda h: (h.hop, -h.score))
        for h in hits:
            if h.chunk_id in seen:
                continue
            seen.add(h.chunk_id)
            docs.append(self._doc(db, h.chunk_id, source=h.direction, hop=h.hop,
                                  score=h.score, via=h.via, edge_type=h.edge_type))

        return docs[: self.k] if self.k else docs

    # ----- helpers --------------------------------------------------------- #
    def _context_window(self) -> int:
        try:
            return int(getattr(self.db._config.local_search, "max_context_tokens", 8000))
        except Exception:
            return 8000

    def _seed_chunk_ids(self, db: LocalSearchDB, records: dict[str, Any]) -> list[str]:
        df = records.get("sources")
        if df is None:
            df = records.get("text_units")
        if df is None or len(df) == 0:
            return []
        id_col = _col(df, "id", "human_readable_id", "short_id")
        text_col = _col(df, "text")
        ordered: list[str] = []
        for row in df.to_dict("records"):
            tu_id = row.get(id_col) if id_col else None
            text = row.get(text_col) if text_col else None
            cid = db.map_text_unit(tu_id, text)
            if cid and cid not in ordered:
                ordered.append(cid)
        return ordered

    def _seed_entities(self, db: LocalSearchDB, records: dict[str, Any]) -> set[str]:
        df = records.get("entities")
        if df is None or len(df) == 0:
            return set()
        name_col = _col(df, "entity", "title", "name")
        if not name_col:
            return set()
        return {canonical_key(db.fold(str(v))) for v in df[name_col].tolist()}

    def _mentions_any(self, db: LocalSearchDB, chunk_id: str, seed_entities: set[str]) -> bool:
        rec = db.chunks.get(chunk_id)
        if not rec or not seed_entities:
            return False
        for m in rec.get("mentioned_entities", []) or []:
            if canonical_key(db.fold(str(m.get("name", "")))) in seed_entities:
                return True
        return False

    def _doc(self, db: LocalSearchDB, chunk_id: str, *, source: str, hop: int,
             score: float, via: str, edge_type: str = "") -> Document:
        rec = db.chunks.get(chunk_id, {})
        metadata = {
            "chunk_id": chunk_id,
            "title": rec.get("title"),
            "breadcrumb": rec.get("breadcrumb"),
            "section_no": rec.get("section_no"),
            "page_start": rec.get("page_start"),
            "page_end": rec.get("page_end"),
            "source": source,
            "hop": hop,
            "score": round(float(score), 4),
        }
        if via:
            metadata["via"] = via
        if edge_type:
            metadata["edge_type"] = edge_type
        return Document(page_content=db.chunk_text(chunk_id) or rec.get("text", ""),
                        metadata=metadata)

    def _claims_doc(self, records: dict[str, Any]) -> Document | None:
        df = records.get("claims")
        if df is None:
            df = records.get("covariates")
        if df is None or len(df) == 0:
            return None
        text_col = _col(df, "description", "statement", "text", "content")
        subj_col = _col(df, "subject_id", "subject", "object_id")
        if not text_col:
            return None
        lines = []
        for row in df.to_dict("records")[:40]:
            subj = f"[{row.get(subj_col)}] " if subj_col and row.get(subj_col) else ""
            stmt = str(row.get(text_col) or "").strip()
            if stmt:
                lines.append(f"- {subj}{stmt}")
        if not lines:
            return None
        return Document(
            page_content="Normative requirements / claims relevant to the query:\n"
                         + "\n".join(lines),
            metadata={"source": "claims", "hop": 0, "score": float("inf"),
                      "n_claims": len(lines)},
        )


def make_retriever(settings: Settings, db_dir: Any = None, k: int = 12) -> GraphRAGLocalRetriever:
    """Build a retriever over the portable DB (defaults to the build output dir)."""
    db = LocalSearchDB(settings, db_dir=db_dir)
    return GraphRAGLocalRetriever(db=db, k=k)


def make_tool(retriever: GraphRAGLocalRetriever, name: str = "spec_search",
              description: str | None = None):
    """Wrap the retriever as a LangChain tool for agents (e.g. test generation)."""
    from langchain_core.tools import Tool

    description = description or (
        "Search the technical specification knowledge graph for the requirements, "
        "fields, error/status conditions, related commands/structures, and explicitly "
        "referenced sections relevant to a question. Returns spec excerpts with "
        "section provenance. Use this before writing or deriving tests."
    )

    def _run(query: str) -> str:
        docs = retriever.invoke(query)
        blocks = []
        for d in docs:
            m = d.metadata
            head = m.get("breadcrumb") or m.get("title") or m.get("chunk_id", "")
            tag = m.get("source", "")
            via = f" via {m['via']}" if m.get("via") else ""
            blocks.append(f"### {head}  [{tag}{via}]\n{d.page_content}")
        return "\n\n".join(blocks) if blocks else "No relevant specification content found."

    return Tool.from_function(func=_run, name=name, description=description)
