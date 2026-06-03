"""Runnable example: query the portable GraphRAG DB for test-generation context.

Shows two modes:
1. Retrieval only — the LangChain ``GraphRAGLocalRetriever`` (no LLM), printing
   provenance for each result (Local Search seed vs xref hop, via which reference).
2. RAG answer — a minimal LangChain chain that feeds retrieved context to your
   ds_api ChatOpenAI to draft test-relevant output.

Usage:
    python -m specgraph.query.example_query "What are the requirements and error \
        conditions for the Abort command?"
"""

from __future__ import annotations

import sys

from specgraph.config import load_settings
from specgraph.query.retriever import make_retriever, make_tool

DEFAULT_QUERY = "What are the requirements and error conditions for the Abort command?"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    query = " ".join(argv) if argv else DEFAULT_QUERY

    settings = load_settings()
    retriever = make_retriever(settings, k=12)

    print(f"\n=== Query ===\n{query}\n")
    docs = retriever.invoke(query)

    print(f"=== Retrieved {len(docs)} chunks (Local Search ⊕ xref) ===")
    for d in docs:
        m = d.metadata
        prov = m.get("source", "")
        via = f" via {m['via']}" if m.get("via") else ""
        hop = m.get("hop", "")
        head = m.get("breadcrumb") or m.get("title") or m.get("chunk_id")
        print(f"\n[{prov}{via} | hop={hop} | score={m.get('score')}]\n  {head}")
        snippet = " ".join(d.page_content.split())[:240]
        print(f"  {snippet}…")

    # Optional: synthesize a test-oriented answer with your local LLM (ds_api).
    try:
        from ds_api import chat  # your configured ChatOpenAI with custom headers
        from langchain_core.prompts import ChatPromptTemplate

        context = "\n\n".join(
            f"## {d.metadata.get('breadcrumb') or d.metadata.get('title')}\n{d.page_content}"
            for d in docs
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a verification engineer. Using ONLY the provided "
                       "specification context, list the testable requirements, fields, "
                       "and error/status conditions, citing section breadcrumbs."),
            ("human", "Question: {q}\n\nSpecification context:\n{ctx}"),
        ])
        chain = prompt | chat
        print("\n=== Draft test-relevant answer (local LLM) ===")
        print(chain.invoke({"q": query, "ctx": context}).content)
    except Exception as exc:  # LLM endpoint not reachable / ds_api not configured
        print(f"\n(Skipped LLM answer: {exc})")

    # The same retriever is available as a LangChain tool for agents:
    _ = make_tool(retriever)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
