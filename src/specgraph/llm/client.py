"""The single place that constructs a raw OpenAI-compatible client.

Used only for the ``doctor`` connectivity check. Indexing goes through GraphRAG's
LiteLLM provider; queries go through ds_api.py's ChatOpenAI. All three point at the
same configured endpoint + custom headers (one source of truth in config.yaml).

There is NO default endpoint: a missing base_url raises.
"""

from __future__ import annotations

from typing import Any

from specgraph.config import LLMSettings


def build_openai_client(llm: LLMSettings) -> Any:
    """Return an ``openai.OpenAI`` configured with base_url + custom headers."""
    if not llm.base_url:
        raise ValueError("llm.base_url is required; refusing to use any default endpoint.")
    from openai import OpenAI

    return OpenAI(
        base_url=llm.base_url,
        api_key=llm.api_key or "not-needed",
        default_headers=dict(llm.default_headers or {}),
        timeout=llm.request_timeout,
        max_retries=0,  # the doctor does a single explicit probe
    )


def ping_chat(llm: LLMSettings) -> str:
    """Send one tiny chat completion to verify endpoint + headers. Raises on failure."""
    client = build_openai_client(llm)
    resp = client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
        temperature=0,
    )
    return resp.choices[0].message.content or ""
