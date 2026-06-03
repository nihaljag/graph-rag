"""Phase A — convert a specification PDF to Markdown with Docling.

Docling is chosen for its high-fidelity table reconstruction: it preserves nested
/ multi-row headers and stitches tables that span page breaks into a single GFM
table — critical for dense specifications like NVMe.

Strict offline: Docling loads its layout/table models from ``docling.artifacts_path``
and never downloads. If the models are missing, this raises with the prefetch
command to run on an online build box.
"""

from __future__ import annotations

from pathlib import Path

from specgraph.config import Settings
from specgraph.logging import get_logger, phase

log = get_logger()


def convert_pdf_to_markdown(pdf_path: Path, settings: Settings) -> str:
    """Return Markdown for a single PDF using Docling (offline)."""
    settings.require_docling()

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        artifacts_path=settings.docling.artifacts_path,
        do_table_structure=settings.docling.do_table_structure,
        do_ocr=settings.docling.do_ocr,
    )
    # Cell matching improves table cell accuracy (nested/merged headers).
    try:
        pipeline_options.table_structure_options.do_cell_matching = True
    except Exception:
        pass

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def run_convert(pdf_path: Path, settings: Settings) -> Path:
    """Convert ``pdf_path`` and write the Markdown artifact. Returns its path."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    settings.markdown_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.markdown_dir / f"{pdf_path.stem}.md"
    with phase(f"Docling PDF→Markdown ({pdf_path.name})"):
        markdown = convert_pdf_to_markdown(pdf_path, settings)
    out_path.write_text(markdown, encoding="utf-8")
    log.info("Wrote Markdown: %s (%d chars)", out_path, len(markdown))
    return out_path
