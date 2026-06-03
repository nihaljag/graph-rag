"""Enforce strict offline operation.

This module sets environment flags (idempotently, without overriding values the
user has already exported) so that the heavy ML/LLM libraries we depend on never
attempt to reach the public internet. The ONLY network egress this project makes
is to the configured local LLM endpoint via LiteLLM.

Per project policy there are NO fallbacks: if a model/tokenizer/artifact is not
present locally, the relevant library raises instead of downloading.
"""

from __future__ import annotations

import os

# Flags that force HuggingFace / transformers / datasets to use only local files.
_OFFLINE_FLAGS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    # Disable assorted telemetry / phone-home behaviours.
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    # LiteLLM: use the bundled model-cost map instead of fetching it over HTTP,
    # and disable its telemetry.
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LITELLM_DONT_SHOW_FEEDBACK_BOX": "True",
    # Avoid tokenizers fork warnings/threads doing unexpected work.
    "TOKENIZERS_PARALLELISM": "false",
}


def enforce_offline() -> None:
    """Apply offline environment flags without clobbering explicit user values."""
    for key, value in _OFFLINE_FLAGS.items():
        os.environ.setdefault(key, value)

    # Honour a user-provided tiktoken cache so encodings load without network.
    # (config.py validates that the directory actually exists when required.)
    cache = os.environ.get("TIKTOKEN_CACHE_DIR")
    if cache:
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", cache)

    # Best-effort: disable LiteLLM telemetry attribute if the lib is imported.
    try:  # pragma: no cover - litellm may not be installed during unit tests
        import litellm  # type: ignore

        litellm.telemetry = False
        litellm.suppress_debug_info = True
    except Exception:
        pass


enforce_offline()
