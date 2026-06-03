"""Strict configuration loader.

Loads ``config.yaml`` (with ``${VAR}`` interpolation from the environment / .env),
validates it into typed dataclasses, and FAILS LOUD on anything missing or any
local path that does not exist. There are deliberately no defaults that point at
a cloud endpoint and no silent fallbacks.

Validation is split so the CLI only requires the inputs a given phase needs
(e.g. ``require_docling`` for PDF conversion, ``require_embeddings`` for indexing
and querying).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:  # python-dotenv is a hard dependency, but keep import resilient for tests
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_a: Any, **_k: Any) -> bool:  # type: ignore
        return False


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid. Always fatal."""


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


# --------------------------------------------------------------------------- #
# Typed settings
# --------------------------------------------------------------------------- #
@dataclass
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    default_headers: dict[str, str]
    supports_json: bool = False
    concurrency: int = 4
    max_retries: int = 5
    request_timeout: int = 180
    temperature: float = 0.0


@dataclass
class EmbeddingSettings:
    local_path: str
    dim: int
    device: str = "cpu"
    batch_size: int = 16
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True


@dataclass
class DoclingSettings:
    artifacts_path: str
    do_table_structure: bool = True
    do_ocr: bool = False


@dataclass
class ChunkingSettings:
    target_tokens: int = 1200
    max_tokens: int = 1800
    prepend_breadcrumb: bool = True
    encoding_model: str = "cl100k_base"


@dataclass
class XrefSettings:
    enabled: bool = True
    inline_resolved_titles: bool = True
    expand_depth: int = 2
    hop_decay: float = 0.5
    expand_token_budget: float = 0.4
    named_references: list[str] = field(default_factory=list)


@dataclass
class GraphRagSettings:
    extract_claims: bool = True
    claims_description: str = ""
    community_reports: bool = True
    community_level: int = 2
    max_gleanings: int = 1
    entity_types: list[str] = field(default_factory=list)
    domain_entity_types: list[str] = field(default_factory=list)
    response_type: str = "multiple paragraphs"

    @property
    def all_entity_types(self) -> list[str]:
        """Generic types plus optional domain overlay, de-duplicated, order-stable."""
        seen: dict[str, None] = {}
        for t in [*self.entity_types, *self.domain_entity_types]:
            seen.setdefault(t, None)
        return list(seen)


@dataclass
class NormalizeSettings:
    enabled: bool = True
    embed_merge: bool = False
    embed_threshold: float = 0.92


@dataclass
class Settings:
    root: Path
    llm: LLMSettings
    embeddings: EmbeddingSettings
    docling: DoclingSettings
    chunking: ChunkingSettings
    xref: XrefSettings
    mentioned_entities_enabled: bool
    graphrag: GraphRagSettings
    normalize: NormalizeSettings
    artifacts_dir: Path
    workspace_dir: Path

    # ---- convenient derived paths ----
    @property
    def markdown_dir(self) -> Path:
        return self.artifacts_dir / "markdown"

    @property
    def chunks_dir(self) -> Path:
        return self.artifacts_dir / "chunks"

    @property
    def manifest_path(self) -> Path:
        return self.artifacts_dir / "chunks_manifest.jsonl"

    @property
    def xref_path(self) -> Path:
        return self.artifacts_dir / "cross_references.parquet"

    @property
    def workspace_input_dir(self) -> Path:
        return self.workspace_dir / "input"

    @property
    def workspace_settings_path(self) -> Path:
        return self.workspace_dir / "settings.yaml"

    @property
    def output_dir(self) -> Path:
        return self.workspace_dir / "output"

    # ---- phase-scoped validation (fail loud) ----
    def require_tiktoken(self) -> None:
        cache = os.environ.get("TIKTOKEN_CACHE_DIR")
        if not cache or not Path(cache).is_dir():
            raise ConfigError(
                "TIKTOKEN_CACHE_DIR is required and must point at a directory holding "
                "the pre-downloaded tiktoken encoding files (offline). "
                "Run scripts/prefetch_docling_models.py's tiktoken step on an online box."
            )

    def require_docling(self) -> None:
        p = Path(self.docling.artifacts_path)
        if not self.docling.artifacts_path or not p.is_dir():
            raise ConfigError(
                f"docling.artifacts_path '{self.docling.artifacts_path}' does not exist. "
                "Pre-cache Docling models offline with scripts/prefetch_docling_models.py."
            )

    def require_embeddings(self) -> None:
        p = Path(self.embeddings.local_path)
        if not self.embeddings.local_path or not p.is_dir():
            raise ConfigError(
                f"embeddings.local_path '{self.embeddings.local_path}' does not exist. "
                "Provide a local HuggingFace embedding model directory (safetensors)."
            )
        if not (p / "config.json").exists():
            raise ConfigError(
                f"embeddings.local_path '{p}' is not a HuggingFace model dir "
                "(missing config.json)."
            )
        if not self.embeddings.dim or int(self.embeddings.dim) <= 0:
            raise ConfigError("embeddings.dim must be a positive integer.")

    def require_llm(self) -> None:
        if not self.llm.base_url:
            raise ConfigError("llm.base_url is required (the local OpenAI-format endpoint).")
        if not self.llm.model:
            raise ConfigError("llm.model is required.")


# --------------------------------------------------------------------------- #
# Loading / interpolation
# --------------------------------------------------------------------------- #
def _interpolate(value: Any) -> Any:
    """Recursively replace ``${VAR}`` with environment values; error if unset."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            var = m.group(1)
            env = os.environ.get(var)
            if env is None:
                raise ConfigError(f"Environment variable ${{{var}}} referenced in config is not set.")
            return env
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _collect_env_headers() -> dict[str, str]:
    """Assemble headers from LLM_HEADER_n_NAME / LLM_HEADER_n_VALUE env pairs."""
    headers: dict[str, str] = {}
    idx = 1
    while True:
        name = os.environ.get(f"LLM_HEADER_{idx}_NAME")
        if name is None:
            break
        name = name.strip()
        value = os.environ.get(f"LLM_HEADER_{idx}_VALUE", "").strip()
        if name:
            headers[name] = value
        idx += 1
    return headers


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be an integer, got {value!r}.")


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    """Load and validate settings from ``config.yaml`` (and ``.env``)."""
    root = Path.cwd()
    # Load .env from CWD if present (does not override already-exported vars).
    load_dotenv(root / ".env")

    cfg_path = Path(config_path) if config_path else root / "config.yaml"
    if not cfg_path.exists():
        raise ConfigError(
            f"Config file not found: {cfg_path}. Copy config.example.yaml to config.yaml "
            "and edit it."
        )
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    raw = _interpolate(raw)

    llm_raw = raw.get("llm", {})
    headers = dict(llm_raw.get("default_headers") or {})
    headers.update(_collect_env_headers())  # env pairs take precedence / augment

    llm = LLMSettings(
        base_url=str(llm_raw.get("base_url", "")).rstrip("/"),
        api_key=str(llm_raw.get("api_key", "") or ""),
        model=str(llm_raw.get("model", "")),
        default_headers=headers,
        supports_json=bool(llm_raw.get("supports_json", False)),
        concurrency=_as_int(llm_raw.get("concurrency", 4), "llm.concurrency"),
        max_retries=_as_int(llm_raw.get("max_retries", 5), "llm.max_retries"),
        request_timeout=_as_int(llm_raw.get("request_timeout", 180), "llm.request_timeout"),
        temperature=float(llm_raw.get("temperature", 0)),
    )

    emb_raw = raw.get("embeddings", {})
    embeddings = EmbeddingSettings(
        local_path=str(emb_raw.get("local_path", "")),
        dim=_as_int(emb_raw.get("dim", 0), "embeddings.dim"),
        device=str(emb_raw.get("device", "cpu") or "cpu"),
        batch_size=_as_int(emb_raw.get("batch_size", 16), "embeddings.batch_size"),
        query_prefix=str(emb_raw.get("query_prefix", "") or ""),
        passage_prefix=str(emb_raw.get("passage_prefix", "") or ""),
        normalize=bool(emb_raw.get("normalize", True)),
    )

    doc_raw = raw.get("docling", {})
    docling = DoclingSettings(
        artifacts_path=str(doc_raw.get("artifacts_path", "")),
        do_table_structure=bool(doc_raw.get("do_table_structure", True)),
        do_ocr=bool(doc_raw.get("do_ocr", False)),
    )

    ch_raw = raw.get("chunking", {})
    chunking = ChunkingSettings(
        target_tokens=_as_int(ch_raw.get("target_tokens", 1200), "chunking.target_tokens"),
        max_tokens=_as_int(ch_raw.get("max_tokens", 1800), "chunking.max_tokens"),
        prepend_breadcrumb=bool(ch_raw.get("prepend_breadcrumb", True)),
        encoding_model=str(ch_raw.get("encoding_model", "cl100k_base")),
    )

    xr_raw = raw.get("xref", {})
    xref = XrefSettings(
        enabled=bool(xr_raw.get("enabled", True)),
        inline_resolved_titles=bool(xr_raw.get("inline_resolved_titles", True)),
        expand_depth=_as_int(xr_raw.get("expand_depth", 2), "xref.expand_depth"),
        hop_decay=float(xr_raw.get("hop_decay", 0.5)),
        expand_token_budget=float(xr_raw.get("expand_token_budget", 0.4)),
        named_references=list(xr_raw.get("named_references") or []),
    )

    gr_raw = raw.get("graphrag", {})
    graphrag = GraphRagSettings(
        extract_claims=bool(gr_raw.get("extract_claims", True)),
        claims_description=str(gr_raw.get("claims_description", "") or ""),
        community_reports=bool(gr_raw.get("community_reports", True)),
        community_level=_as_int(gr_raw.get("community_level", 2), "graphrag.community_level"),
        max_gleanings=_as_int(gr_raw.get("max_gleanings", 1), "graphrag.max_gleanings"),
        entity_types=list(gr_raw.get("entity_types") or []),
        domain_entity_types=list(gr_raw.get("domain_entity_types") or []),
        response_type=str(gr_raw.get("response_type", "multiple paragraphs")),
    )

    nm_raw = raw.get("normalize", {})
    normalize = NormalizeSettings(
        enabled=bool(nm_raw.get("enabled", True)),
        embed_merge=bool(nm_raw.get("embed_merge", False)),
        embed_threshold=float(nm_raw.get("embed_threshold", 0.92)),
    )

    paths_raw = raw.get("paths", {})
    artifacts_dir = _resolve(root, paths_raw.get("artifacts_dir", "artifacts"))
    workspace_dir = _resolve(root, paths_raw.get("workspace_dir", "graphrag_workspace"))

    return Settings(
        root=root,
        llm=llm,
        embeddings=embeddings,
        docling=docling,
        chunking=chunking,
        xref=xref,
        mentioned_entities_enabled=bool(raw.get("mentioned_entities", {}).get("enabled", True)),
        graphrag=graphrag,
        normalize=normalize,
        artifacts_dir=artifacts_dir,
        workspace_dir=workspace_dir,
    )


def _resolve(root: Path, p: str | os.PathLike[str]) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)
