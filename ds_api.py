"""Shared LLM handle for LangChain/LangGraph usage (TEMPLATE).

If you already have your own ``ds_api.py`` exposing a configured ``ChatOpenAI``,
keep yours — just make sure it points at the same endpoint + custom headers as
``config.yaml`` so indexing and querying use one source of truth.

This template builds the ChatOpenAI from the same config.yaml/.env that the
indexer uses, so there is a single place to configure the endpoint and headers.
Importing it enforces offline mode for everything except the local LLM endpoint.
"""

from __future__ import annotations

import specgraph.offline  # noqa: F401  (enforces offline env on import)
from langchain_openai import ChatOpenAI

from specgraph.config import load_settings

_settings = load_settings()
_llm = _settings.llm

# The configured local, OpenAI-compatible chat model with your custom headers.
chat = ChatOpenAI(
    base_url=_llm.base_url,
    api_key=_llm.api_key or "not-needed",
    model=_llm.model,
    temperature=_llm.temperature,
    timeout=_llm.request_timeout,
    max_retries=_llm.max_retries,
    default_headers=dict(_llm.default_headers or {}),
)


def ChatOpenAIFactory() -> ChatOpenAI:
    """Return the shared ChatOpenAI instance (kept as a function for convenience)."""
    return chat


__all__ = ["chat", "ChatOpenAIFactory"]
