# specgraph — Offline GraphRAG (Local Search) index builder for specifications

Turn a technical specification (PDF, Markdown, or a directory of chunks) into a
**portable, fully offline Microsoft GraphRAG "Local Search" knowledge base** that
you can copy to any machine and query from LangChain/LangGraph.

Primary use case: **NVMe specifications → retrieval for test generation** — but the
design is **generic and specification-agnostic** (PCIe, storage, firmware, protocol,
standards, internal design / requirements docs).

> **Runtime is strictly offline.** The only network egress is to your local,
> OpenAI-compatible LLM endpoint (with custom headers). Embeddings come from a
> **local HuggingFace model on disk**. If anything is misconfigured, the tool
> **fails loud** — there are no silent fallbacks and no cloud calls.

---

## What it does

```
            ┌─ --from pdf ──► [A] Docling PDF→Markdown ─┐
input ──────┼─ --from md ───────────────────────────────┼─► [B] Section chunking ─┐
            └─ --from chunks ─────────────────────────────────────────(skip A,B)──┤
                                                                                   ▼
                                            [B] + deterministic cross-reference graph
                                                 → chunks/ + chunks_manifest.jsonl + cross_references.parquet
                                                                                   │
                                                  [C] GraphRAG build_index  ◄───────┘
                                                  → PORTABLE DB: output/*.parquet + lancedb/
                                                                   │
                                  [D] LangChain retriever = Local Search ⊕ xref expansion (+ @tool)
```

### Key properties
- **Section-based chunking** (not fixed-size): one logical section = one chunk;
  oversized sections are split at sub-headings / table row boundaries; tables are
  never split mid-table; multi-page/nested tables are preserved by Docling.
- **Explicit cross-references are guaranteed traversable.** A deterministic (no-LLM)
  stage detects `Section X.Y`, `Figure/Table N`, and named references, resolves them
  to chunks, and stores them as `cross_references.parquet` edges. Retrieval follows
  these even if the LLM never inferred the relationship — with **depth-2 BFS**,
  **token-budget** caps, and **hop-decay** ranking (direct refs outrank indirect).
- **Behavioral dependencies** ("the Abort section mentions Identify") become entity
  edges via GraphRAG extraction, hardened by tuned entity types + (optional) inline
  reference-title enrichment, then traversed by Local Search.
- **Generic `mentioned_entities`** metadata per chunk (from the document's own
  vocabulary) improves ranking and debugging — no new retrieval system, no dictionaries.
- **Entity normalization** (anti-fragmentation): generic canonical-key merge + optional
  embedding/token-subset merge → `entity_aliases.parquet`, applied at retrieval time
  so the indexed store/embeddings are never corrupted.
- **Custom headers**: delivered to the LLM on every call via GraphRAG's LiteLLM
  `call_args.extra_headers` (no proxy, no monkeypatch).
- **Portable**: copy `graphrag_workspace/output/` + the local embedding model + your
  config to any offline machine and query there.

---

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

For the **offline target**, build a wheel cache on an online box and install from it:

```bash
# online build box
pip download -r requirements.txt -d wheelhouse
# offline target
pip install --no-index --find-links wheelhouse -r requirements.txt
```

### One-time offline model prep (on a machine WITH internet)

```bash
DOCLING_ARTIFACTS=/path/docling_models python scripts/prefetch_docling_models.py
```

Copy `docling_models/` to the offline machine. Also place your **HuggingFace
embedding model** (safetensors) on disk there. (No tokenizer download is needed —
token counts are estimated from word counts, so there is no tiktoken dependency.)

---

## Configure

```bash
cp .env.example .env            # endpoint, custom headers, local model paths
cp config.example.yaml config.yaml
```

Edit `.env`:
- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- `LLM_HEADER_1_NAME` / `LLM_HEADER_1_VALUE` (repeat `_2_`, `_3_`, … for more headers)
- `EMBED_MODEL_PATH`, `EMBED_DIM`, `EMBED_DEVICE`
- `DOCLING_ARTIFACTS`

`config.yaml` is the single source of truth; `ds_api.py` reads the same config so the
indexer and your LangChain queries use one endpoint + headers.

Verify everything before running:

```bash
gridx doctor
```

---

## Run

```bash
# Full pipeline from a PDF:
gridx build path/to/spec.pdf --from pdf

# …or start from a single Markdown file (skip PDF parsing):
gridx build path/to/spec.md --from md

# …or start from a directory of pre-made chunks (skip parsing + chunking):
gridx build path/to/chunks_dir --from chunks
```

Or run phases individually: `gridx convert`, `gridx chunk --from md|chunks`, `gridx index`.

### Progress & logging (long indexing runs)

Indexing a large spec makes many LLM calls and can take a while. The build prints:
- each **workflow** start/end (`extract_graph`, `extract_claims`, `community_report`, …),
- each **LLM call** with a running counter, in-flight count, and a label describing what
  it's for (e.g. `LLM #842 [extract_graph] (in-flight 4)`), plus slow-call flags,
- a periodic **heartbeat** so even a single long task shows liveness:
  `⏳ working [community_report] | elapsed 6m12s | LLM: 840 done, 4 in-flight, 0 failed`.

Tune or disable via the `logging:` block in `config.yaml` (`llm_calls`,
`heartbeat_seconds`, `slow_call_seconds`, `snippet_chars`). Setting `llm_calls: false`
uses the plain LiteLLM provider (custom headers are unaffected either way).

The portable DB is written to `graphrag_workspace/output/`:
`entities/relationships/communities/community_reports/text_units/documents/covariates.parquet`
+ `lancedb/` + `cross_references.parquet` + `entity_aliases.parquet` +
`chunks_manifest.jsonl` + `db_manifest.json`.

---

## Query (LangChain)

```bash
gridx query "What are the requirements and error conditions for the Abort command?" --answer
```

Programmatically:

```python
from specgraph.config import load_settings
from specgraph.query.retriever import make_retriever, make_tool

settings = load_settings()
retriever = make_retriever(settings, k=12)        # langchain BaseRetriever
docs = retriever.invoke("Abort command requirements and error conditions")

tool = make_tool(retriever)                       # langchain @tool for agents
```

Each returned `Document` carries provenance: `source` (`local_search` / `xref_out` /
`xref_in` / `claims`), `hop`, `via` (e.g. `Section 5.1`), `breadcrumb`, `section_no`.

---

## Move the DB to another machine

Copy `graphrag_workspace/output/` (the whole folder). On the target you need:
this repo + deps, the **same** local embedding model at `EMBED_MODEL_PATH`, a
`config.yaml`/`.env` pointing at that machine's local LLM endpoint + headers, and
`ds_api.py`. Then:

```bash
gridx query "…" --db /path/to/copied/output
```

No internet, no re-indexing.

---

## Testing

```bash
pip install -e ".[dev]"
pytest                                   # unit tests for chunking + xref (offline)
python scripts/make_sample_pdf.py sample_spec.pdf   # synthetic spec for end-to-end trials
```

## Notes & limitations
- Pinned to `graphrag==3.1.0` (the model-registry/config API is version-sensitive).
- Embedding model must be loadable by `sentence-transformers`. For asymmetric models
  (e5-style) set `embeddings.passage_prefix` / `query_prefix` consistently.
- Set `llm.supports_json: true` only if your model reliably returns valid JSON.
