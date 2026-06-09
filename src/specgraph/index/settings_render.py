"""Render a GraphRAG 3.1 ``settings.yaml`` from our config.

Design choices baked in here:
- Local Search only: community reports ON (Local Search context uses them),
  Global/DRIFT/Basic prompts left at defaults but never invoked by us.
- Custom headers: delivered via the stock LiteLLM provider's
  ``call_args.extra_headers`` (no custom chat provider needed).
- Local embeddings: our registered ``local_hf`` provider; extra fields
  (local_path/device/…) ride on the model entry (ModelConfig allows extras).
- Section-preserving chunking: ``chunking.size`` is set very large with zero
  overlap so each input document (one of OUR section chunks) becomes exactly one
  text-unit. We bypass GraphRAG's own document loading by passing
  ``input_documents`` to ``build_index``.
- Built-in default prompts are used for robustness; entity types + claim
  extraction (the high-leverage knobs) are tuned from config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from specgraph.config import Settings

COMPLETION_MODEL_ID = "default_completion_model"
EMBEDDING_MODEL_ID = "default_embedding_model"

# Large enough that each of our pre-made section chunks stays a single text-unit.
_ONE_CHUNK_SIZE = 1_000_000


def build_settings_dict(settings: Settings) -> dict[str, Any]:
    llm = settings.llm
    emb = settings.embeddings
    gr = settings.graphrag

    call_args: dict[str, Any] = {"temperature": llm.temperature}
    if llm.default_headers:
        call_args["extra_headers"] = dict(llm.default_headers)

    # Use the logging chat provider when call logging is enabled (default), else
    # the stock litellm provider. Both carry custom headers via call_args.
    chat_type = "logging_litellm" if settings.logging.llm_calls else "litellm"
    completion_model: dict[str, Any] = {
        "type": chat_type,
        "model_provider": "openai",   # OpenAI-compatible local endpoint
        "model": llm.model,
        "api_base": llm.base_url,
        "api_key": llm.api_key or "not-needed",
        "auth_method": "api_key",
        "call_args": call_args,
        "retry": {"type": "exponential_backoff"},
    }

    embedding_model: dict[str, Any] = {
        "type": "local_hf",            # our registered offline provider
        "model_provider": "local",
        "model": Path(emb.local_path).name or "local-embedding",
        # extra fields consumed by LocalHFEmbedding:
        "local_path": emb.local_path,
        "device": emb.device,
        "batch_size": emb.batch_size,
        "normalize": emb.normalize,
        "passage_prefix": emb.passage_prefix,
    }

    cfg: dict[str, Any] = {
        "completion_models": {COMPLETION_MODEL_ID: completion_model},
        "embedding_models": {EMBEDDING_MODEL_ID: embedding_model},

        "input": {"type": "json"},
        "input_storage": {"type": "file", "base_dir": "input"},
        "output_storage": {"type": "file", "base_dir": "output"},
        "reporting": {"type": "file", "base_dir": "logs"},
        "cache": {"type": "none"},   # caching disabled (valid types: json|memory|none)

        "chunking": {
            "type": "tokens",
            "size": _ONE_CHUNK_SIZE,
            "overlap": 0,
            "encoding_model": settings.chunking.encoding_model,
            "prepend_metadata": [],   # our chunk text already carries the breadcrumb
        },

        "vector_store": {
            "type": "lancedb",
            "db_uri": str(settings.output_dir / "lancedb"),
        },

        "embed_text": {"embedding_model_id": EMBEDDING_MODEL_ID},

        "extract_graph": {
            "completion_model_id": COMPLETION_MODEL_ID,
            "entity_types": gr.all_entity_types,
            "max_gleanings": gr.max_gleanings,
        },
        "summarize_descriptions": {"completion_model_id": COMPLETION_MODEL_ID},
        "cluster_graph": {"max_cluster_size": 10},
        "extract_claims": {
            "enabled": bool(gr.extract_claims),
            "completion_model_id": COMPLETION_MODEL_ID,
            "description": gr.claims_description or
                "Any normative requirement, constraint, dependency, or behavioral rule "
                "stated about an entity.",
            "max_gleanings": gr.max_gleanings,
        },
        "community_reports": {"completion_model_id": COMPLETION_MODEL_ID},

        "local_search": {
            "completion_model_id": COMPLETION_MODEL_ID,
            "embedding_model_id": EMBEDDING_MODEL_ID,
        },

        "snapshots": {"graphml": False, "embeddings": False},
        "umap": {"enabled": False},
    }
    return cfg


def render_settings(settings: Settings) -> Path:
    """Write ``graphrag_workspace/settings.yaml`` and return its path."""
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    settings.workspace_input_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_settings_dict(settings)
    path = settings.workspace_settings_path
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path
