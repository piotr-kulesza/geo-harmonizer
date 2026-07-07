#!/usr/bin/env python3
"""Day-1 acceptance check: fetch the demo series and print each summary.

Runs against real GEO (network). Not run by the offline build sandbox — the
author runs this locally to confirm the demo series fetch or fall back cleanly.

Usage:
    python scripts/smoke_fetch.py [ACCESSION ...]
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
        help="GEO series accessions to fetch (default: the placeholder demo set).",
    )
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    accessions = args.accessions or DEFAULT_ACCESSIONS
    for accession in accessions:
        result = fetch_gse(accession, cache_dir=args.cache_dir)
        print(result.summary())
        if result.ok:
            print(
                f"    matrix: {result.n_features} features x "
                f"{result.n_samples} samples"
            )
            if result.metadata is not None:
                print(
                    f"    metadata: {result.metadata.shape[0]} samples x "
                    f"{result.metadata.shape[1]} fields"
                )
        elif result.metadata is not None:
            print(
                f"    metadata usable: {result.metadata.shape[0]} samples x "
                f"{result.metadata.shape[1]} fields (matrix needs upload)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
