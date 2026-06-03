"""A GraphRAG embedding provider backed by a LOCAL HuggingFace model.

GraphRAG 3.x has no built-in way to embed with an on-disk safetensors model, so
we implement its ``LLMEmbedding`` contract and register it under the type
``local_hf``. The model is loaded with sentence-transformers in strict offline
mode (``HF_HUB_OFFLINE`` is set in ``specgraph.offline``); there is NO network
fallback — a missing/unloadable model raises.

Used at BOTH index time and query time, which keeps the resulting DB fully
portable and offline.
"""

from __future__ import annotations

from typing import Any

# graphrag_llm is only importable where the full deps are installed (the target
# machine / a full env). Import lazily-friendly at module top is fine because this
# module is only imported by the index/query phases.
from graphrag_llm.embedding.embedding import LLMEmbedding
from graphrag_llm.types import LLMEmbeddingResponse
from openai.types.embedding import Embedding
from openai.types.create_embedding_response import Usage


class LocalHFEmbedding(LLMEmbedding):
    """Embed text with a local sentence-transformers model."""

    PROVIDER_TYPE = "local_hf"

    def __init__(
        self,
        *,
        model_id: str,
        model_config: Any,
        tokenizer: Any,
        metrics_store: Any,
        metrics_processor: Any = None,
        rate_limiter: Any = None,
        retrier: Any = None,
        cache: Any = None,
        cache_key_creator: Any = None,
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store

        # Extra fields ride on the (extra="allow") ModelConfig and/or kwargs.
        def opt(name: str, default: Any) -> Any:
            val = getattr(model_config, name, None)
            if val is None:
                val = kwargs.get(name, default)
            return val

        local_path = opt("local_path", None)
        if not local_path:
            raise ValueError(
                "local_hf embedding requires 'local_path' (the on-disk HuggingFace "
                "model directory). Set embeddings.local_path in config.yaml."
            )
        self._device = str(opt("device", "cpu") or "cpu")
        self._batch_size = int(opt("batch_size", 16) or 16)
        self._normalize = bool(opt("normalize", True))
        self._passage_prefix = str(opt("passage_prefix", "") or "")

        from sentence_transformers import SentenceTransformer

        # local_files_only guarantees no network access regardless of env flags.
        self._model = SentenceTransformer(
            str(local_path), device=self._device,
            local_files_only=True, trust_remote_code=False,
        )

    # -- core --------------------------------------------------------------- #
    def _encode(self, inputs: list[str]) -> list[list[float]]:
        texts = [f"{self._passage_prefix}{t}" if self._passage_prefix else t for t in inputs]
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def _response(self, inputs: list[str]) -> LLMEmbeddingResponse:
        vectors = self._encode(inputs)
        data = [
            Embedding(index=i, embedding=vec, object="embedding")
            for i, vec in enumerate(vectors)
        ]
        return LLMEmbeddingResponse(
            data=data,
            model=self._model_id,
            object="list",
            usage=Usage(prompt_tokens=0, total_tokens=0),
        )

    def embedding(self, /, **kwargs: Any) -> LLMEmbeddingResponse:
        inputs = list(kwargs.get("input") or [])
        return self._response(inputs)

    async def embedding_async(self, /, **kwargs: Any) -> LLMEmbeddingResponse:
        # sentence-transformers is synchronous; run inline (the indexer manages
        # its own thread pool around this).
        return self.embedding(**kwargs)

    # -- required properties ----------------------------------------------- #
    @property
    def metrics_store(self) -> Any:
        return self._metrics_store

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    # -- convenience for our own query-side embedding ---------------------- #
    def embed_query(self, text: str, query_prefix: str = "") -> list[float]:
        prefixed = f"{query_prefix}{text}" if query_prefix else text
        vec = self._model.encode(
            [prefixed], normalize_embeddings=self._normalize, convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vec.tolist()


def load_local_model(local_path: str, device: str = "cpu"):
    """Load a sentence-transformers model from disk, strictly offline.

    Shared by the entity-normalization (Tier 3) and any other code that needs the
    raw model. Raises if the model cannot be loaded locally — no network fallback.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        str(local_path), device=device, local_files_only=True, trust_remote_code=False,
    )
