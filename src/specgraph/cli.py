"""``gridx`` — command-line interface for the specgraph pipeline.

Phases (each writes a durable artifact, so they are independently re-runnable):
    convert   PDF  -> Markdown            (Docling, offline)
    chunk     MD/dir -> chunks + manifest + cross_references.parquet
    index     chunks -> portable GraphRAG DB (parquet + lancedb)
    build     run the whole pipeline from --from {pdf,md,chunks}
    query     query the portable DB (Local Search ⊕ xref)
    doctor    preflight: validate config, paths, endpoint, GraphRAG API
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

import specgraph.offline  # noqa: F401  (enforce offline on import)
from specgraph.config import ConfigError, Settings, load_settings
from specgraph.logging import get_logger

app = typer.Typer(add_completion=False, help="Offline GraphRAG (Local Search) index builder.")
log = get_logger()

CONFIG_OPT = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml")


def _load(config: str) -> Settings:
    try:
        return load_settings(config)
    except ConfigError as exc:
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


# --------------------------------------------------------------------------- #
@app.command()
def convert(
    pdf: Path = typer.Argument(..., help="Path to the specification PDF."),
    config: str = CONFIG_OPT,
) -> None:
    """Phase A — convert a PDF to Markdown (Docling, offline)."""
    settings = _load(config)
    from specgraph.convert.docling_md import run_convert

    out = run_convert(pdf, settings)
    typer.echo(str(out))


@app.command()
def chunk(
    source: Path = typer.Argument(..., help="Markdown file (--from md) or chunks directory (--from chunks)."),
    from_: str = typer.Option("md", "--from", help="md | chunks"),
    config: str = CONFIG_OPT,
) -> None:
    """Phase B — section-chunk a Markdown file, or ingest a chunks directory."""
    settings = _load(config)
    from specgraph.chunk.run import run_chunk_from_directory, run_chunk_from_markdown

    if from_ == "chunks":
        chunks = run_chunk_from_directory(source, settings)
    elif from_ == "md":
        chunks = run_chunk_from_markdown(source, settings)
    else:
        typer.secho("--from must be 'md' or 'chunks'", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    typer.echo(f"{len(chunks)} chunks -> {settings.manifest_path}")


@app.command()
def index(config: str = CONFIG_OPT) -> None:
    """Phase C — build the portable GraphRAG DB from the chunk manifest."""
    settings = _load(config)
    from specgraph.index.build import run_build

    result = run_build(settings)
    typer.echo(f"Portable DB: {result['output_dir']}")


@app.command()
def build(
    source: Path = typer.Argument(..., help="PDF (--from pdf), Markdown file (--from md), or chunks dir (--from chunks)."),
    from_: str = typer.Option("pdf", "--from", help="pdf | md | chunks"),
    config: str = CONFIG_OPT,
) -> None:
    """Run the full pipeline end-to-end from the chosen entry point."""
    settings = _load(config)
    from specgraph.chunk.run import run_chunk_from_directory, run_chunk_from_markdown
    from specgraph.index.build import run_build

    if from_ == "pdf":
        from specgraph.convert.docling_md import run_convert

        md = run_convert(source, settings)
        run_chunk_from_markdown(md, settings)
    elif from_ == "md":
        run_chunk_from_markdown(source, settings)
    elif from_ == "chunks":
        run_chunk_from_directory(source, settings)
    else:
        typer.secho("--from must be 'pdf', 'md', or 'chunks'", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    result = run_build(settings)
    typer.secho(f"\n✔ Portable GraphRAG DB: {result['output_dir']}", fg=typer.colors.GREEN)


@app.command()
def query(
    text: str = typer.Argument(..., help="The query."),
    k: int = typer.Option(12, "--k", help="Max chunks to return."),
    db: Optional[Path] = typer.Option(None, "--db", help="Portable DB dir (defaults to build output)."),
    answer: bool = typer.Option(False, "--answer", help="Also synthesize an answer via the local LLM."),
    config: str = CONFIG_OPT,
) -> None:
    """Phase D — query the portable DB (Local Search ⊕ xref expansion)."""
    settings = _load(config)
    from specgraph.query.retriever import make_retriever

    retriever = make_retriever(settings, db_dir=db, k=k)
    docs = retriever.invoke(text)
    for d in docs:
        m = d.metadata
        via = f" via {m['via']}" if m.get("via") else ""
        head = m.get("breadcrumb") or m.get("title") or m.get("chunk_id")
        typer.secho(f"\n[{m.get('source')}{via} | hop={m.get('hop')} | score={m.get('score')}]",
                    fg=typer.colors.CYAN)
        typer.echo(f"  {head}")
        typer.echo("  " + " ".join(d.page_content.split())[:240] + "…")

    if answer:
        from ds_api import chat
        from langchain_core.prompts import ChatPromptTemplate

        ctx = "\n\n".join(f"## {d.metadata.get('breadcrumb')}\n{d.page_content}" for d in docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using ONLY the provided specification context; cite section breadcrumbs."),
            ("human", "Question: {q}\n\nContext:\n{ctx}"),
        ])
        typer.secho("\n=== Answer ===", fg=typer.colors.GREEN)
        typer.echo((prompt | chat).invoke({"q": text, "ctx": ctx}).content)


@app.command()
def doctor(config: str = CONFIG_OPT) -> None:
    """Preflight: validate config, local paths, the LLM endpoint, and GraphRAG."""
    ok = True

    def check(label: str, fn) -> None:
        nonlocal ok
        try:
            detail = fn() or "ok"
            typer.secho(f"  ✔ {label}: {detail}", fg=typer.colors.GREEN)
        except Exception as exc:  # noqa: BLE001
            ok = False
            typer.secho(f"  ✖ {label}: {exc}", fg=typer.colors.RED)

    typer.echo("specgraph doctor — offline preflight\n")
    try:
        settings = load_settings(config)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"  ✖ config: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("  ✔ config.yaml loaded", fg=typer.colors.GREEN)

    check("tiktoken cache", lambda: (settings.require_tiktoken(), "present")[1])
    check("embedding model path", lambda: (settings.require_embeddings(), settings.embeddings.local_path)[1])
    check("docling artifacts (only needed for PDF)", lambda: (settings.require_docling(), settings.docling.artifacts_path)[1])
    check("graphrag importable", _check_graphrag)
    check("custom-header registration", _check_register)
    check("LLM endpoint reachable", lambda: _check_llm(settings))

    if ok:
        typer.secho("\nAll checks passed.", fg=typer.colors.GREEN)
    else:
        typer.secho("\nSome checks failed — fix the items above before running.", fg=typer.colors.RED)
        raise typer.Exit(1)


def _check_graphrag() -> str:
    import importlib.metadata as m

    import graphrag  # noqa: F401
    return f"graphrag {m.version('graphrag')}"


def _check_register() -> str:
    from specgraph.llm.register import register_models

    register_models()
    return "local_hf embedding provider registered"


def _check_llm(settings: Settings) -> str:
    settings.require_llm()
    from specgraph.llm.client import ping_chat

    ping_chat(settings.llm)
    return f"{settings.llm.base_url} ({settings.llm.model})"


if __name__ == "__main__":
    app()
