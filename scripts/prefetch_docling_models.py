"""Pre-cache Docling's offline models. RUN THIS ON A MACHINE WITH INTERNET.

It downloads the Docling layout/table-structure models to DOCLING_ARTIFACTS.
Then copy that directory to the offline target and point config.yaml/.env at it.
NOTE: do NOT import specgraph here — that package forces offline mode.

(No tokenizer prefetch is needed: specgraph estimates tokens from word counts,
so there is no tiktoken dependency.)

Usage:
    DOCLING_ARTIFACTS=/path/docling_models python scripts/prefetch_docling_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def prefetch_docling(artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"[docling] downloading models to {artifacts_dir} …")
    try:
        from docling.utils.model_downloader import download_models

        download_models(output_dir=artifacts_dir)
        print("[docling] done.")
    except Exception as exc:  # pragma: no cover
        print(f"[docling] python API failed ({exc}). Try the CLI instead:")
        print(f"    docling-tools models download -o {artifacts_dir}")
        raise


def main() -> int:
    artifacts = Path(os.environ.get("DOCLING_ARTIFACTS", "docling_models"))
    prefetch_docling(artifacts)
    print("\nCopy this to the offline machine and set DOCLING_ARTIFACTS in .env:")
    print(f"  DOCLING_ARTIFACTS={artifacts.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
