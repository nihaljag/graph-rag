"""Register custom GraphRAG model providers (idempotent).

- Embeddings: our offline local-HF provider, registered as type ``local_hf``.
- Chat: a thin logging wrapper over GraphRAG's stock LiteLLM provider, registered
  as ``logging_litellm``. It adds per-call console logging (what each call is for)
  while still delivering custom headers via ``call_args.extra_headers``. If call
  logging is disabled in config, settings render the plain ``litellm`` type and
  this wrapper is simply unused.

If the GraphRAG factory API does not match the pinned version, we FAIL LOUD with
the detected version (there is no proxy/monkeypatch fallback).
"""

from __future__ import annotations

from typing import Any

_REGISTERED = False


def _graphrag_version() -> str:
    try:
        import importlib.metadata as m
        return m.version("graphrag")
    except Exception:
        return "unknown"


def register_models(settings: Any = None) -> None:
    """Register the local-HF embedding and logging chat providers."""
    global _REGISTERED

    # Apply logging tunables from config (safe to do on every call).
    if settings is not None:
        from specgraph.llm import call_log
        call_log.SNIPPET_CHARS = int(settings.logging.snippet_chars)
        call_log.SLOW_CALL_SECONDS = float(settings.logging.slow_call_seconds)

    if _REGISTERED:
        return
    try:
        from graphrag_llm.completion import register_completion
        from graphrag_llm.embedding import register_embedding
    except Exception as exc:  # pragma: no cover - version guard
        raise RuntimeError(
            "Could not import graphrag_llm register_completion/register_embedding "
            f"(graphrag version {_graphrag_version()}). This project targets graphrag==3.1.0; "
            "pin that version. No fallback is attempted by design."
        ) from exc

    from specgraph.llm.graphrag_chat import LoggingLiteLLMCompletion, PROVIDER_TYPE as CHAT_TYPE
    from specgraph.llm.graphrag_embed import LocalHFEmbedding

    register_embedding(LocalHFEmbedding.PROVIDER_TYPE, LocalHFEmbedding, scope="singleton")
    register_completion(CHAT_TYPE, LoggingLiteLLMCompletion, scope="singleton")
    _REGISTERED = True
