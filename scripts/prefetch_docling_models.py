"""Pre-cache offline model artifacts. RUN THIS ON A MACHINE WITH INTERNET.

It downloads:
  1. Docling layout/table-structure models -> DOCLING_ARTIFACTS
  2. The tiktoken cl100k_base encoding   -> TIKTOKEN_CACHE_DIR

Then copy both directories to the offline target and point config.yaml/.env at
them. NOTE: do NOT import specgraph here — that package forces offline mode.

Usage:
    DOCLING_ARTIFACTS=/path/docling_models \
    TIKTOKEN_CACHE_DIR=/path/tiktoken_cache \
    python scripts/prefetch_docling_models.py
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


def prefetch_tiktoken(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    print(f"[tiktoken] caching cl100k_base to {cache_dir} …")
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.encode("hello world")  # force materialization
    print("[tiktoken] done.")


def main() -> int:
    artifacts = Path(os.environ.get("DOCLING_ARTIFACTS", "docling_models"))
    cache = Path(os.environ.get("TIKTOKEN_CACHE_DIR", "tiktoken_cache"))
    prefetch_docling(artifacts)
    prefetch_tiktoken(cache)
    print("\nCopy these to the offline machine and set DOCLING_ARTIFACTS / "
          "TIKTOKEN_CACHE_DIR in .env:")
    print(f"  DOCLING_ARTIFACTS={artifacts.resolve()}")
    print(f"  TIKTOKEN_CACHE_DIR={cache.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
