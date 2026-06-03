"""Offline end-to-end smoke test of the chunk -> xref -> manifest -> index-prep path.

Heavy deps (graphrag/docling/sentence-transformers) are not exercised here; this
validates the pure-python pipeline, the manifest/IO, the documents frame fed to
GraphRAG, and the settings rendering.
"""

from __future__ import annotations

import textwrap

import pytest

from specgraph.config import load_settings

MD = textwrap.dedent("""
    # 5 Command Set

    ## 5.1 Identify command

    The Identify command returns a data structure. See Status Code Definitions.

    ## 5.2 Abort command

    The Abort command aborts a command. If the command being aborted is an
    Identify command, complete per Section 5.1. Status is returned per Section 3.

    # 3 Status Code Definitions

    Status codes are defined here.
""").strip()


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    emb = tmp_path / "emb"; emb.mkdir(); (emb / "config.json").write_text("{}")
    doc = tmp_path / "doc"; doc.mkdir()
    tok = tmp_path / "tok"; tok.mkdir()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tok))

    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""
        llm: {{base_url: "http://local/v1", api_key: "k", model: "m",
               default_headers: {{X-Auth: secret}}, temperature: 0}}
        embeddings: {{local_path: "{emb}", dim: 8, device: cpu}}
        docling: {{artifacts_path: "{doc}"}}
        chunking: {{target_tokens: 40, max_tokens: 60, prepend_breadcrumb: true}}
        xref: {{enabled: true, expand_depth: 2, named_references: ["Status Code Definitions"]}}
        mentioned_entities: {{enabled: true}}
        graphrag: {{entity_types: [component, function], extract_claims: true}}
        normalize: {{enabled: true}}
        paths: {{artifacts_dir: "{tmp_path / 'artifacts'}", workspace_dir: "{tmp_path / 'ws'}"}}
    """))
    monkeypatch.chdir(tmp_path)
    return load_settings(cfg)


def test_chunk_pipeline_writes_artifacts(settings):
    from specgraph.chunk.run import run_chunk_from_markdown

    md_path = settings.root / "spec.md"
    md_path.write_text(MD)
    chunks = run_chunk_from_markdown(md_path, settings)

    assert len(chunks) >= 3
    assert settings.manifest_path.exists()
    assert settings.xref_path.exists()
    # One chunk file per chunk.
    assert len(list(settings.chunks_dir.glob("*.md"))) == len(chunks)


def test_xref_edges_resolved_in_parquet(settings):
    import pandas as pd
    from specgraph.chunk.run import run_chunk_from_markdown

    md_path = settings.root / "spec.md"
    md_path.write_text(MD)
    run_chunk_from_markdown(md_path, settings)

    edges = pd.read_parquet(settings.xref_path)
    resolved = edges[edges["resolved"]]
    # Abort -> Section 5.1 (Identify) and -> Section 3 / named Status Code Definitions.
    assert len(resolved) >= 2
    assert set(["references_section"]).issubset(set(edges["edge_type"]))


def test_documents_dataframe_and_settings_render(settings):
    import yaml
    from specgraph.chunk.run import run_chunk_from_markdown
    from specgraph.chunk.manifest import documents_dataframe, load_manifest
    from specgraph.index.settings_render import build_settings_dict

    md_path = settings.root / "spec.md"
    md_path.write_text(MD)
    run_chunk_from_markdown(md_path, settings)

    df = documents_dataframe(load_manifest(settings.manifest_path))
    assert {"id", "title", "text", "raw_data"}.issubset(set(df.columns))
    assert df["id"].is_unique and len(df) >= 3

    cfg = build_settings_dict(settings)
    # Custom headers are delivered via the LiteLLM provider's call_args.
    chat = cfg["completion_models"]["default_completion_model"]
    assert chat["call_args"]["extra_headers"] == {"X-Auth": "secret"}
    assert cfg["embedding_models"]["default_embedding_model"]["type"] == "local_hf"
    assert cfg["extract_claims"]["enabled"] is True
    assert cfg["chunking"]["overlap"] == 0  # one chunk per document
    # Round-trips as valid YAML.
    assert yaml.safe_load(yaml.safe_dump(cfg))
