"""A logging wrapper around GraphRAG's LiteLLM chat provider.

Indexing a large spec makes thousands of LLM calls inside long-running workflows.
The stock provider logs nothing, so the console looks frozen. This subclass logs
each call with a running counter, in-flight count, and a short label describing
WHAT the call is for (entity extraction, claim extraction, community report, …),
and flags slow calls. The labelling/counter logic lives in ``call_log`` so it can
be unit-tested without graphrag installed.

Custom headers still flow through unchanged via ``call_args.extra_headers`` on the
base ``LiteLLMCompletion`` — this only adds observability.
"""

from __future__ import annotations

import time
from typing import Any

from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion

from specgraph.llm import call_log

PROVIDER_TYPE = "logging_litellm"


class LoggingLiteLLMCompletion(LiteLLMCompletion):
    """LiteLLM completion that logs each call's purpose, count, and duration."""

    def completion(self, /, **kwargs: Any) -> Any:
        n, lbl = call_log.begin(kwargs.get("messages"))
        t0 = time.perf_counter()
        failed = False
        try:
            return super().completion(**kwargs)
        except Exception:
            failed = True
            raise
        finally:
            call_log.end(n, lbl, time.perf_counter() - t0, failed)

    async def completion_async(self, /, **kwargs: Any) -> Any:
        n, lbl = call_log.begin(kwargs.get("messages"))
        t0 = time.perf_counter()
        failed = False
        try:
            return await super().completion_async(**kwargs)
        except Exception:
            failed = True
            raise
        finally:
            call_log.end(n, lbl, time.perf_counter() - t0, failed)
