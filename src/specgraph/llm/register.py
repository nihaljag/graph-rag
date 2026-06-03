"""Register custom GraphRAG model providers (idempotent).

- Embeddings: our offline local-HF provider, registered as type ``local_hf``.
- Chat: NO custom provider is required. GraphRAG's stock LiteLLM provider passes
  ``call_args.extra_headers`` straight through to ``litellm.completion``, which is
  exactly how we deliver the custom headers your endpoint needs.

If the GraphRAG factory API does not match the pinned version, we FAIL LOUD with
the detected version (there is no proxy/monkeypatch fallback).
"""

from __future__ import annotations

_REGISTERED = False


def _graphrag_version() -> str:
    try:
        import importlib.metadata as m
        return m.version("graphrag")
    except Exception:
        return "unknown"


def register_models() -> None:
    """Register the local-HF embedding provider with GraphRAG's factory."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from graphrag_llm.embedding import register_embedding
    except Exception as exc:  # pragma: no cover - version guard
        raise RuntimeError(
            "Could not import graphrag_llm.embedding.register_embedding "
            f"(graphrag version {_graphrag_version()}). This project targets graphrag==3.1.0; "
            "pin that version. No fallback is attempted by design."
        ) from exc

    from specgraph.llm.graphrag_embed import LocalHFEmbedding

    register_embedding(LocalHFEmbedding.PROVIDER_TYPE, LocalHFEmbedding, scope="singleton")
    _REGISTERED = True
