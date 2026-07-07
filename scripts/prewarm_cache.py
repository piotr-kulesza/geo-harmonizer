#!/usr/bin/env python3
"""Pre-fetch the demo series into data/cache/ so the demo NEVER live-fetches.

Runs against real GEO (network). The author runs this locally before recording
(TICKETS 1.7). The final demo accessions are filled in after curation — the
default list below is a placeholder.

Usage:
    python scripts/prewarm_cache.py [ACCESSION ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import fetch_gse  # noqa: E402

DEFAULT_ACCESSIONS = ["GSE9891", "GSE26712", "GSE26193"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "accessions",
        nargs="*",
        default=DEFAULT_ACCESSIONS,
        help="GEO series accessions to warm (default: the placeholder demo set).",
    )
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cache_dir = Path(args.cache_dir)
    accessions = args.accessions or DEFAULT_ACCESSIONS

    print(f"Warming cache in {cache_dir}/ for: {', '.join(accessions)}")
    for accession in accessions:
        # A soft/matrix file already on disk means GEOparse reads from cache.
        before = _cache_signature(cache_dir, accession)
        result = fetch_gse(accession, cache_dir=str(cache_dir))
        after = _cache_signature(cache_dir, accession)
        hit = before and before == after
        state = "cache hit" if hit else "downloaded"
        print(f"  {result.summary()}  [{state}]")

    return 0


def _cache_signature(cache_dir: Path, accession: str) -> tuple:
    """A cheap fingerprint of an accession's cached files (name, size)."""
    if not cache_dir.exists():
        return ()
    files = sorted(cache_dir.glob(f"{accession}*"))
    return tuple((f.name, f.stat().st_size) for f in files)


if __name__ == "__main__":
    raise SystemExit(main())
